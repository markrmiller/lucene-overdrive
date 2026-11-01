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

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * An idempotent operation-wide lease on an immutable value. Acquisition and retirement share one
 * atomic state, so removal never frees storage beneath a reader that won acquisition.
 *
 * @lucene.experimental
 */
public final class SharedValueLease<T> implements AutoCloseable {
  private final Entry<T> entry;
  private final AtomicBoolean closed = new AtomicBoolean();

  private SharedValueLease(Entry<T> entry) {
    this.entry = entry;
  }

  /** The borrowed value must not escape this lease's lifetime. */
  public T value() {
    if (closed.get()) throw new IllegalStateException("closed lease");
    return entry.value;
  }

  @Override
  public void close() {
    if (closed.compareAndSet(false, true)) entry.release();
  }

  /**
   * One nonreusable owner reference plus readers. Publish the initialized entry before sharing it;
   * transferring builder ownership to a cache does not alter its reference count.
   */
  public static final class Entry<T> {
    private static final long LIVE = Long.MIN_VALUE;
    private static final long REFS = Long.MAX_VALUE;
    private volatile T value;
    private Runnable destroy;
    private final MemoryLedger.Charge charge;
    private final AtomicLong state = new AtomicLong(LIVE | 1);
    private final Object reclamationLock = new Object();
    private boolean attempted;
    private boolean reclaiming;
    private boolean reclaimed;

    /**
     * The destructor must deterministically release the owned backing allocation. To support
     * explicit recovery, it must be safe to retry after a partially completed failed invocation.
     */
    public Entry(T value, Runnable destroy, MemoryLedger.Charge charge) {
      this.value = Objects.requireNonNull(value);
      this.destroy = Objects.requireNonNull(destroy);
      this.charge = Objects.requireNonNull(charge);
    }

    /** Returns null after retirement; no entry can be resurrected. */
    public SharedValueLease<T> tryAcquire() {
      while (true) {
        long previous = state.get();
        if ((previous & LIVE) == 0) return null;
        if ((previous & REFS) == REFS) throw new IllegalStateException("lease count overflow");
        if (state.compareAndSet(previous, previous + 1)) return new SharedValueLease<>(this);
      }
    }

    /** Closes admission and drops the owner's reference in the same CAS. */
    public void retire() {
      while (true) {
        long previous = state.get();
        if ((previous & LIVE) == 0) return;
        long remaining = (previous & REFS) - 1;
        if (state.compareAndSet(previous, remaining)) {
          if (remaining == 0) destroyOnce();
          else charge.reclassify(MemoryLedger.State.RETIRED_PINNED);
          return;
        }
      }
    }

    /** Diagnostic reader count; it is not a capacity multiplier. */
    public long activeLeases() {
      long observed = state.get();
      return (observed & REFS) - ((observed & LIVE) != 0 ? 1 : 0);
    }

    MemoryLedger.Charge charge() {
      return charge;
    }

    private void release() {
      long remaining = state.decrementAndGet();
      if (remaining == 0) destroyOnce();
    }

    private void destroyOnce() {
      synchronized (reclamationLock) {
        if (attempted) return;
        attempted = true;
        reclaiming = true;
      }
      reclaim();
    }

    /**
     * Retries a diagnosed failed close only after retirement and the final reader release. A
     * concurrent reclamation already owns the attempt; this method then returns without calling the
     * destructor. Successful physical close is the only operation that refunds the charge.
     */
    public void retryFailedReclamation() {
      synchronized (reclamationLock) {
        if (state.get() != 0) throw new IllegalStateException("value still has an owner or reader");
        if (reclaiming || reclaimed) return;
        MemoryLedger.State chargedState = charge.snapshot().state();
        if (attempted == false
            || (chargedState != MemoryLedger.State.FAILED_RECLAMATION
                && chargedState != MemoryLedger.State.RELEASED))
          throw new IllegalStateException("value has no failed reclamation to retry");
        reclaiming = true;
      }
      reclaim();
    }

    private void reclaim() {
      try {
        // No admission or entry lock is held while the external destructor runs.
        destroy.run();
        value = null;
        destroy = null;
        charge.releaseAfterDeallocation();
        synchronized (reclamationLock) {
          reclaimed = true;
        }
        ManagedMemory.forgetFailedReclamation(this);
      } catch (RuntimeException | Error failure) {
        charge.reclassify(MemoryLedger.State.FAILED_RECLAMATION);
        ManagedMemory.retainFailedReclamation(this);
        throw failure;
      } finally {
        synchronized (reclamationLock) {
          reclaiming = false;
        }
      }
    }
  }
}
