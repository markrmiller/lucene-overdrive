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
package org.apache.lucene.store;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.lucene.acceleration.memory.ManagedMemory;
import org.apache.lucene.acceleration.memory.MemoryLedger;

/** Charges a physical arena exactly once per mapped file, including shared-arena delayed unmap. */
@SuppressWarnings("preview")
final class ManagedMappingArena implements Arena {
  private static final class Tracker {
    final List<MemoryLedger.Charge> charges = new ArrayList<>();
  }

  private static final ConcurrentHashMap<MemorySegment.Scope, Tracker> TRACKERS =
      new ConcurrentHashMap<>();
  private final Arena delegate;
  private final Tracker tracker;
  private final MemoryLedger.Charge charge;
  private boolean closed;

  private ManagedMappingArena(Arena delegate, Tracker tracker, MemoryLedger.Charge charge) {
    this.delegate = delegate;
    this.tracker = tracker;
    this.charge = charge;
  }

  static Arena enroll(Arena arena, Path path, long capacity, MemoryLedger ledger) {
    MemoryLedger.Owner owner =
        ManagedMemory.owner(path.getParent().toString(), MemoryLedger.Purpose.CACHE_VALUE);
    try (MemoryLedger.Reservation reservation =
        ManagedMemory.reserve(
            ledger, owner, MemoryLedger.Kind.MAPPED, capacity, MemoryLedger.Purpose.CACHE_VALUE)) {
      if (reservation == null) return arena;
      MemoryLedger.Charge charge = reservation.commit(capacity, MemoryLedger.State.ACTIVE);
      Tracker tracker = TRACKERS.computeIfAbsent(arena.scope(), ignored -> new Tracker());
      synchronized (tracker) {
        tracker.charges.add(charge);
      }
      return new ManagedMappingArena(arena, tracker, charge);
    }
  }

  @Override
  public MemorySegment allocate(long bytes, long alignment) {
    return delegate.allocate(bytes, alignment);
  }

  @Override
  public MemorySegment.Scope scope() {
    return delegate.scope();
  }

  @Override
  public synchronized void close() {
    if (closed) return;
    try {
      delegate.close();
    } catch (RuntimeException | Error failure) {
      synchronized (tracker) {
        for (MemoryLedger.Charge item : tracker.charges)
          item.reclassify(MemoryLedger.State.FAILED_RECLAMATION);
      }
      throw failure;
    }
    closed = true;
    synchronized (tracker) {
      if (delegate.scope().isAlive()) charge.reclassify(MemoryLedger.State.RETIRED_PINNED);
      else {
        for (MemoryLedger.Charge item : tracker.charges) item.releaseAfterDeallocation();
        tracker.charges.clear();
        TRACKERS.remove(delegate.scope(), tracker);
      }
    }
  }
}
