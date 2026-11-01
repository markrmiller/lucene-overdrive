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

import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.acceleration.AccelerationRuntime;

/**
 * Opt-in enrollment for existing heap and mapping owners; zero budgets preserve current behavior.
 * Heap charges describe retained references, not physical GC timing.
 *
 * @lucene.experimental
 */
public final class ManagedMemory {
  private ManagedMemory() {}

  private static final Cleaner CLEANER = Cleaner.create();
  private static final AtomicLong IDS = new AtomicLong();
  private static final Object OWNERS_LOCK = new Object();
  private static final Map<AccelerationRuntime, Map<Registration, Runnable>> OWNERS =
      new IdentityHashMap<>();
  private static final Object FAILED_LOCK = new Object();

  private record FailedReclamation(MemoryLedger.Charge charge, Object owner, Runnable retry) {}

  private static final Map<MemoryLedger, Map<MemoryLedger.Charge, FailedReclamation>> FAILED =
      new IdentityHashMap<>();

  static void retainFailedReclamation(SharedValueLease.Entry<?> entry) {
    retainFailedReclamation(entry.charge(), entry, entry::retryFailedReclamation);
  }

  static void retainFailedReclamation(MemoryLedger.Charge charge, Object owner, Runnable retry) {
    synchronized (FAILED_LOCK) {
      FAILED
          .computeIfAbsent(charge.ledger(), ignored -> new IdentityHashMap<>())
          .put(charge, new FailedReclamation(charge, owner, retry));
    }
  }

  static void forgetFailedReclamation(SharedValueLease.Entry<?> entry) {
    forgetFailedReclamation(entry.charge(), entry);
  }

  static void forgetFailedReclamation(MemoryLedger.Charge charge, Object owner) {
    synchronized (FAILED_LOCK) {
      MemoryLedger ledger = charge.ledger();
      Map<MemoryLedger.Charge, FailedReclamation> entries = FAILED.get(ledger);
      if (entries != null) {
        FailedReclamation failed = entries.get(charge);
        if (failed != null && failed.owner() == owner) entries.remove(charge);
        if (entries.isEmpty()) FAILED.remove(ledger);
      }
    }
  }

  private static List<FailedReclamation> failedEntries(MemoryLedger ledger, int maximum) {
    Objects.requireNonNull(ledger);
    if (maximum < 0) throw new IllegalArgumentException("negative diagnostic limit");
    ArrayList<FailedReclamation> snapshot = new ArrayList<>();
    synchronized (FAILED_LOCK) {
      Map<MemoryLedger.Charge, FailedReclamation> entries = FAILED.get(ledger);
      if (entries != null) {
        for (FailedReclamation entry : entries.values()) {
          if (snapshot.size() == maximum) break;
          snapshot.add(entry);
        }
      }
    }
    return snapshot;
  }

  /**
   * Bounded diagnostics for failed managed-store reclamation. The registry retains the allocation
   * and its close handle even after the cache and reader disappear; it never retries automatically.
   */
  public static List<MemoryLedger.AllocationRecord> failedReclamations(
      MemoryLedger ledger, int maximum) {
    ArrayList<MemoryLedger.AllocationRecord> records = new ArrayList<>();
    for (FailedReclamation entry : failedEntries(ledger, maximum)) {
      MemoryLedger.AllocationRecord record = entry.charge().snapshot();
      if (record.state() == MemoryLedger.State.FAILED_RECLAMATION) records.add(record);
    }
    return List.copyOf(records);
  }

  /**
   * Explicit operator retry of retained shared-value and standalone typed-store close failures.
   * Destructors run outside the registry lock; each entry serializes concurrent attempts and
   * refunds only after successful physical reclamation. Runtime failures remain retained and are
   * reported after all attempts.
   */
  public static void retryFailedReclamations(MemoryLedger ledger) {
    RuntimeException failure = null;
    for (FailedReclamation entry : failedEntries(ledger, Integer.MAX_VALUE)) {
      try {
        entry.retry().run();
      } catch (RuntimeException exception) {
        if (failure == null) failure = exception;
        else failure.addSuppressed(exception);
      }
    }
    if (failure != null) throw failure;
  }

  /** Iterator leases owned by one reader lifetime, without retaining the iterators themselves. */
  public static final class Readers implements AutoCloseable {
    private final Set<ReaderLease> leases = ConcurrentHashMap.newKeySet();
    private volatile boolean closed;

    /** The release action must retain only the lease, never the iterator used as referent. */
    public ReaderLease register(Object iterator, Runnable release) {
      ReaderLease lease = new ReaderLease(this, release);
      try {
        lease.cleanable = CLEANER.register(iterator, lease);
        leases.add(lease);
        if (closed) lease.close();
        return lease;
      } finally {
        Reference.reachabilityFence(iterator);
      }
    }

    /** Final reader closure, after its callers have stopped accessing borrowed storage. */
    @Override
    public void close() {
      closed = true;
      Throwable failure = null;
      for (ReaderLease lease : leases) {
        try {
          lease.close();
        } catch (RuntimeException | Error exception) {
          if (failure == null) failure = exception;
          else failure.addSuppressed(exception);
        }
      }
      if (failure instanceof RuntimeException exception) throw exception;
      if (failure instanceof Error error) throw error;
    }
  }

  /** An iterator's idempotent lease release, also invoked when the iterator becomes unreachable. */
  public static final class ReaderLease implements Runnable, AutoCloseable {
    private final Readers readers;
    private volatile boolean closed;
    private Runnable release;
    private Cleaner.Cleanable cleanable;

    private ReaderLease(Readers readers, Runnable release) {
      this.readers = readers;
      this.release = Objects.requireNonNull(release);
    }

    /** Whether exhaustion, abandonment or reader closure has released the lease. */
    public boolean isClosed() {
      return closed;
    }

    @Override
    public synchronized void run() {
      if (closed == false) {
        closed = true;
        Runnable action = release;
        release = null;
        try {
          action.run();
        } finally {
          readers.leases.remove(this);
        }
      }
    }

    @Override
    public void close() {
      try {
        run();
      } finally {
        cleanable.clean();
      }
    }
  }

  /** A shutdown retirement registration; unregister when the owner normally retires. */
  public static final class Registration implements AutoCloseable {
    private final AccelerationRuntime runtime;

    private Registration(AccelerationRuntime runtime) {
      this.runtime = runtime;
    }

    @Override
    public void close() {
      synchronized (OWNERS_LOCK) {
        Map<Registration, Runnable> registrations = OWNERS.get(runtime);
        if (registrations != null) {
          registrations.remove(this);
          if (registrations.isEmpty()) OWNERS.remove(runtime);
        }
      }
    }
  }

  /** Registers cache-owner retirement; physical mappings remain tied to actual reader closure. */
  public static Registration registerOwner(AccelerationRuntime runtime, Runnable retire) {
    Registration registration = new Registration(runtime);
    synchronized (OWNERS_LOCK) {
      if (runtime.ledger().isShutdown())
        throw new RequestMemoryScope.AdmissionException(RequestMemoryScope.Failure.SHUTDOWN);
      OWNERS.computeIfAbsent(runtime, ignored -> new IdentityHashMap<>()).put(registration, retire);
    }
    return registration;
  }

  /** Stops cache ownership outside the registry lock. Live leases remain charged until released. */
  public static void retireAll(AccelerationRuntime runtime) {
    ArrayList<Runnable> retire;
    synchronized (OWNERS_LOCK) {
      Map<Registration, Runnable> registrations = OWNERS.remove(runtime);
      if (registrations == null) return;
      retire = new ArrayList<>(registrations.values());
    }
    RuntimeException failure = null;
    for (Runnable action : retire) {
      try {
        action.run();
      } catch (RuntimeException exception) {
        if (failure == null) failure = exception;
        else failure.addSuppressed(exception);
      }
    }
    if (failure != null) throw failure;
  }

  /** A stable process-unique owner for reusable Lucene allocations. */
  public static MemoryLedger.Owner owner(String core, MemoryLedger.Purpose purpose) {
    String id = "lucene-" + purpose + "-" + IDS.incrementAndGet();
    return new MemoryLedger.Owner(id, core, 0, id);
  }

  /** Whether a dimension is explicitly enabled, independent of compute/storage policy. */
  public static boolean enabled(MemoryLedger.Kind kind) {
    return AccelerationRuntime.system().ledger().isEnrolled(kind);
  }

  /** Reserves an existing owner before allocating; null denotes unenrolled default behavior. */
  public static MemoryLedger.Reservation reserve(
      MemoryLedger.Owner owner, MemoryLedger.Kind kind, long bytes, MemoryLedger.Purpose purpose) {
    return reserve(AccelerationRuntime.system().ledger(), owner, kind, bytes, purpose);
  }

  /** Explicit ledger enrollment for embedded owners and isolated tests. */
  public static MemoryLedger.Reservation reserve(
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      MemoryLedger.Kind kind,
      long bytes,
      MemoryLedger.Purpose purpose) {
    if (ledger.isEnrolled(kind) == false) return null;
    if (ledger.isShutdown())
      throw new RequestMemoryScope.AdmissionException(RequestMemoryScope.Failure.SHUTDOWN);
    MemoryLedger.Admission admission = ledger.tryReserve(owner, kind, bytes, purpose);
    if (admission.granted() == false)
      throw new RequestMemoryScope.AdmissionException(
          RequestMemoryScope.Failure.valueOf(admission.reason().name()));
    return admission.reservation();
  }

  /** Retains the charge until the heap owner becomes unreachable, or explicit reference release. */
  public static Cleaner.Cleanable retainHeap(Object owner, MemoryLedger.Charge charge) {
    if (charge.snapshot().kind() != MemoryLedger.Kind.HEAP)
      throw new IllegalArgumentException("only managed heap can use reachability cleanup");
    charge.reclassify(MemoryLedger.State.PUBLISHED);
    return CLEANER.register(owner, charge::releaseAfterDeallocation);
  }
}
