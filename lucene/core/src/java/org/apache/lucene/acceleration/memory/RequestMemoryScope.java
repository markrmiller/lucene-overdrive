/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.acceleration.memory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

/**
 * Bounded native scratch owner. Worker scopes pin all backing allocations until workers exit.
 * Cancellation prevents allocation; it does not close memory beneath running workers. Borrowed
 * buffers must never escape the parent or a pinned worker's lifetime.
 *
 * @lucene.experimental
 */
public final class RequestMemoryScope implements AutoCloseable {
  /** Stable reason for failure before allocating native capacity. */
  public enum Failure {
    NODE_CAPACITY,
    REQUEST_LIMIT,
    CORE_LIMIT,
    SHUTDOWN,
    FAIR_SHARE,
    QUEUE_FULL,
    DEADLINE,
    PRESSURE,
    CANCELLED,
    CLOSED
  }

  /** Resource-limit failure; callers may fall back only to a separately admitted alternative. */
  public static final class AdmissionException extends IllegalStateException {
    private static final long serialVersionUID = 1L;
    private final Failure reason;

    public AdmissionException(Failure reason) {
      super("scratch allocation denied: " + reason);
      this.reason = reason;
    }

    public Failure reason() {
      return reason;
    }

    /** Capacity denials can select an admitted alternative; lifecycle failures cannot. */
    public boolean isCapacityFailure() {
      return switch (reason) {
        case NODE_CAPACITY, REQUEST_LIMIT, CORE_LIMIT, FAIR_SHARE, QUEUE_FULL, PRESSURE -> true;
        case SHUTDOWN, DEADLINE, CANCELLED, CLOSED -> false;
      };
    }
  }

  /** Host-owned foreground admission, including bounded cancellation-aware waiting if enabled. */
  @FunctionalInterface
  public interface AdmissionPolicy {
    MemoryLedger.Admission reserve(
        MemoryLedger.Owner owner,
        MemoryLedger.Kind kind,
        long bytes,
        MemoryLedger.Purpose purpose,
        BooleanSupplier cancelled);
  }

  private record Owned(NativeAllocator.Allocation allocation, MemoryLedger.Charge charge) {}

  private final Object lock = new Object();
  private final MemoryLedger ledger;
  private final MemoryLedger.Owner owner;
  private final NativeAllocator allocator;
  private final AdmissionPolicy admissionPolicy;
  private final Thread confinedThread;
  private final List<Owned> owned = new ArrayList<>();
  private final List<Owned> failedReclamations = new ArrayList<>();
  private int pins = 1;
  private boolean cancelled;
  private boolean closed;
  private volatile boolean alive = true;
  private RequestMemoryScope parent;
  private final List<Runnable> drainListeners = new ArrayList<>();
  private boolean drained;
  private int reclaimers;

  public RequestMemoryScope(
      MemoryLedger ledger, MemoryLedger.Owner owner, NativeAllocator allocator) {
    this(ledger, owner, allocator, null, null);
  }

  /** A host may install bounded admission; the default constructor remains fail-fast. */
  public RequestMemoryScope(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      NativeAllocator allocator,
      AdmissionPolicy admissionPolicy) {
    this(ledger, owner, allocator, null, Objects.requireNonNull(admissionPolicy));
  }

  private RequestMemoryScope(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      NativeAllocator allocator,
      Thread confinedThread,
      AdmissionPolicy admissionPolicy) {
    this.ledger = Objects.requireNonNull(ledger);
    this.owner = Objects.requireNonNull(owner);
    this.allocator = Objects.requireNonNull(allocator);
    this.confinedThread = confinedThread;
    this.admissionPolicy =
        admissionPolicy == null
            ? (who, kind, bytes, purpose, cancellation) ->
                ledger.tryReserve(who, kind, bytes, purpose)
            : admissionPolicy;
  }

  /**
   * Staging for synchronous calls. Allocation, worker pins and physical close stay on the creating
   * thread; cancellation may still arrive from another thread.
   */
  public static RequestMemoryScope threadConfined(
      MemoryLedger ledger, MemoryLedger.Owner owner, NativeAllocator allocator) {
    return threadConfined(ledger, owner, allocator, null);
  }

  /**
   * Synchronous staging with the host task's admission policy; null preserves fail-fast behavior.
   */
  public static RequestMemoryScope threadConfined(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      NativeAllocator allocator,
      AdmissionPolicy admissionPolicy) {
    return new RequestMemoryScope(
        ledger, owner, allocator, Thread.currentThread(), admissionPolicy);
  }

  private void checkThread() {
    if (confinedThread != null && confinedThread != Thread.currentThread())
      throw new IllegalStateException("synchronous staging belongs to another thread");
  }

  /** Ledger identity used to reject staging trees crossing runtime budget domains. */
  public MemoryLedger ledger() {
    return ledger;
  }

  /** Allocates native scratch; buffer indices are bounded by the Java ByteBuffer limit. */
  public ByteBuffer allocate(long bytes) {
    return allocate(bytes, null);
  }

  /** Checked sizing for integer scratch. */
  public ByteBuffer allocateInts(long count) {
    return allocate(MemoryLedger.bytes(count, Integer.BYTES));
  }

  /** Checked sizing for long scratch. */
  public ByteBuffer allocateLongs(long count) {
    return allocate(MemoryLedger.bytes(count, Long.BYTES));
  }

  /**
   * Reserves a caller-owned heap scratch peak before arrays are created. Close after dropping those
   * references. Zero managed-heap capacity preserves the unenrolled baseline policy.
   */
  public HeapReservation reserveHeap(long bytes) {
    if (bytes < 0) throw new IllegalArgumentException("negative heap capacity");
    WorkerScope pin = forkWorker();
    MemoryLedger.Reservation reservation = null;
    try {
      if (ledger.isEnrolled(MemoryLedger.Kind.HEAP))
        reservation = reserve(MemoryLedger.Kind.HEAP, bytes);
      else if (ledger.isShutdown()) throw new AdmissionException(Failure.SHUTDOWN);
      synchronized (lock) {
        checkOpen(null);
      }
      return new HeapReservation(reservation, pin);
    } catch (RuntimeException | Error failure) {
      if (reservation != null) reservation.close();
      pin.close();
      throw failure;
    }
  }

  /** Capacity credits and a worker pin for caller-owned heap scratch references. */
  public static final class HeapReservation implements AutoCloseable {
    private final MemoryLedger.Charge charge;
    private final WorkerScope pin;
    private final AtomicBoolean closed = new AtomicBoolean();

    private HeapReservation(MemoryLedger.Reservation reservation, WorkerScope pin) {
      charge =
          reservation == null
              ? null
              : reservation.commit(reservation.remaining(), MemoryLedger.State.ACTIVE);
      if (reservation != null) reservation.close();
      this.pin = pin;
    }

    /** Whether heap enrollment is explicitly enabled. */
    public boolean enrolled() {
      return charge != null;
    }

    @Override
    public void close() {
      pin.checkThread();
      if (closed.compareAndSet(false, true)) {
        try {
          if (charge != null) charge.releaseAfterDeallocation();
        } finally {
          pin.close();
        }
      }
    }
  }

  private ByteBuffer allocate(long bytes, WorkerScope worker) {
    checkThread();
    if (bytes < 0 || bytes > Integer.MAX_VALUE)
      throw new IllegalArgumentException("unsupported buffer size");
    synchronized (lock) {
      checkOpen(worker);
      pins =
          Math.incrementExact(pins); // protect allocation and publication against concurrent close
    }
    try {
      long peak = allocator.allocationSize(bytes);
      if (peak < bytes) throw new IllegalArgumentException("allocator underestimates capacity");
      try (MemoryLedger.Reservation reservation = reserve(MemoryLedger.Kind.NATIVE, peak)) {
        // Commit first, so even a failed close on validation failure stays fully charged.
        MemoryLedger.Charge charge = reservation.commit(peak, MemoryLedger.State.ACTIVE);
        NativeAllocator.Allocation allocation;
        try {
          allocation = Objects.requireNonNull(allocator.allocate(bytes));
        } catch (RuntimeException | Error failure) {
          charge.releaseAfterDeallocation(); // allocator must unwind failed creation
          throw failure;
        }
        Owned resource = new Owned(allocation, charge);
        try {
          ByteBuffer buffer = Objects.requireNonNull(allocation.buffer());
          if (allocation.capacity() > peak
              || allocation.capacity() < bytes
              || buffer.capacity() < bytes) {
            throw new IllegalStateException("allocator capacity violates reservation");
          }
          ByteBuffer view =
              buffer.duplicate().clear().limit((int) bytes).slice().order(ByteOrder.nativeOrder());
          synchronized (lock) {
            checkOpen(worker);
            owned.add(resource);
          }
          return view;
        } catch (RuntimeException | Error failure) {
          try {
            release(resource);
          } catch (RuntimeException closeFailure) {
            failure.addSuppressed(closeFailure);
          }
          throw failure;
        }
      }
    } finally {
      unpin();
    }
  }

  private MemoryLedger.Reservation reserve(MemoryLedger.Kind kind, long bytes) {
    MemoryLedger.Admission admission =
        admissionPolicy.reserve(
            owner,
            kind,
            bytes,
            MemoryLedger.Purpose.REQUEST_SCRATCH,
            () -> {
              synchronized (lock) {
                return closed || cancelled || parent != null && parent.isCancelled();
              }
            });
    if (!admission.granted())
      throw new AdmissionException(Failure.valueOf(admission.reason().name()));
    return admission.reservation();
  }

  private void checkOpen(WorkerScope worker) {
    if (cancelled || (parent != null && parent.isCancelled()))
      throw new AdmissionException(Failure.CANCELLED);
    if (closed || (worker != null && worker.closed.get()))
      throw new AdmissionException(Failure.CLOSED);
  }

  /** Registers a worker before scheduling it. The worker must close in the task's finally block. */
  public WorkerScope forkWorker() {
    checkThread();
    synchronized (lock) {
      checkOpen(null);
      pins = Math.incrementExact(pins);
      return new WorkerScope();
    }
  }

  /** Stops allocations; backing capacity remains alive until parent and workers close. */
  public void cancel() {
    synchronized (lock) {
      cancelled = true;
    }
  }

  public boolean isCancelled() {
    synchronized (lock) {
      return cancelled || (parent != null && parent.isCancelled());
    }
  }

  /** Independent staging owner sharing this request's quota and cancellation. */
  public RequestMemoryScope childScope() {
    WorkerScope pin = forkWorker();
    RequestMemoryScope child =
        new RequestMemoryScope(ledger, owner, allocator, confinedThread, admissionPolicy);
    child.parent = this;
    child.onDrained(pin::close);
    return child;
  }

  /** Runs once after all workers exit and all physical backing storage is released. */
  public void onDrained(Runnable listener) {
    Objects.requireNonNull(listener);
    synchronized (lock) {
      if (drained == false) {
        drainListeners.add(listener);
        return;
      }
    }
    listener.run();
  }

  /** Whether physical reclamation, including recovery from failed closes, has completed. */
  public boolean isDrained() {
    synchronized (lock) {
      return drained;
    }
  }

  private void notifyDrained() {
    List<Runnable> listeners;
    synchronized (lock) {
      if (drained
          || pins != 0
          || reclaimers != 0
          || owned.isEmpty() == false
          || failedReclamations.isEmpty() == false) return;
      drained = true;
      listeners = new ArrayList<>(drainListeners);
      drainListeners.clear();
    }
    RuntimeException failure = null;
    for (Runnable listener : listeners) {
      try {
        listener.run();
      } catch (RuntimeException exception) {
        if (failure == null) failure = exception;
        else failure.addSuppressed(exception);
      }
    }
    if (failure != null) throw failure;
  }

  /** Backing memory stays accessible while the parent or any worker remains pinned. */
  void checkAccessible() {
    if (alive == false) throw new IllegalStateException("released request storage");
  }

  private void release(Owned resource) {
    try {
      resource.allocation().close();
    } catch (RuntimeException | Error failure) {
      resource.charge().reclassify(MemoryLedger.State.FAILED_RECLAMATION);
      synchronized (lock) {
        failedReclamations.add(resource);
      }
      throw failure;
    }
    resource.charge().releaseAfterDeallocation();
  }

  /**
   * Explicit recovery after diagnosing failed physical closes. Live backing storage is closed
   * again; an allocator-confirmed dead allocation is refunded without attempting another physical
   * close. A thread-confined scope must be recovered on its creating thread before that thread
   * exits.
   */
  public void retryFailedReclamations() {
    List<Owned> retry;
    synchronized (lock) {
      if (failedReclamations.isEmpty()) return;
      checkThread();
      retry = new ArrayList<>(failedReclamations);
      failedReclamations.clear();
      reclaimers++;
    }
    RuntimeException failure = null;
    for (Owned resource : retry) {
      try {
        if (resource.allocation().isAlive()) release(resource);
        else resource.charge().releaseAfterDeallocation();
      } catch (RuntimeException closeFailure) {
        // A liveness probe may itself fail before release() can retain the handle. Keep every
        // failed attempt recoverable, without duplicating handles retained by release().
        synchronized (lock) {
          if (failedReclamations.contains(resource) == false) failedReclamations.add(resource);
        }
        if (failure == null) failure = closeFailure;
        else failure.addSuppressed(closeFailure);
      }
    }
    synchronized (lock) {
      reclaimers--;
    }
    notifyDrained();
    if (failure != null) throw failure;
  }

  private void unpin() {
    List<Owned> cleanup;
    synchronized (lock) {
      if (--pins != 0) return;
      alive = false;
      cleanup = new ArrayList<>(owned);
      owned.clear();
      reclaimers++;
    }
    RuntimeException failure = null;
    for (Owned resource : cleanup) {
      try {
        release(resource);
      } catch (RuntimeException closeFailure) {
        if (failure == null) failure = closeFailure;
        else failure.addSuppressed(closeFailure);
      }
    }
    synchronized (lock) {
      reclaimers--;
    }
    notifyDrained();
    if (failure != null) throw failure;
  }

  @Override
  public void close() {
    checkThread();
    synchronized (lock) {
      if (closed) return;
      closed = true;
    }
    unpin();
  }

  /** A worker pins the parent after parent close, until this handle closes. */
  public final class WorkerScope implements AutoCloseable {
    private final AtomicBoolean closed = new AtomicBoolean();

    private WorkerScope() {}

    private void checkThread() {
      RequestMemoryScope.this.checkThread();
    }

    public ByteBuffer allocate(long bytes) {
      return RequestMemoryScope.this.allocate(bytes, this);
    }

    public boolean isCancelled() {
      return RequestMemoryScope.this.isCancelled();
    }

    @Override
    public void close() {
      checkThread();
      if (closed.compareAndSet(false, true)) unpin();
    }
  }
}
