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
package org.apache.lucene.acceleration;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.List;

/**
 * Counters every native call updates, kept in one cell per thread so a call does plain stores
 * instead of an atomic read-modify-write per counter (eight per call with {@code LongAdder}).
 *
 * <p>Only the owner thread writes its cell, with opaque stores; readers sum every cell with opaque
 * loads, so a snapshot may miss a call in progress but never sees a torn or decreasing count. When
 * a new cell is created, cells of exited threads are folded into a retired total; a thread's exit
 * happens-before {@code isAlive()} returns false, so its final counts are complete.
 */
final class CallCounters {
  static final int CALLS = 0,
      ITEMS = 1,
      STAGED = 2,
      BYTES = 3,
      NATIVE_NANOS = 4,
      BRIDGE_NANOS = 5,
      SCRATCH_HIGH_WATER = 6,
      BATCH_SIZES = 7,
      BUCKETS = 32,
      WIDTH = BATCH_SIZES + BUCKETS;

  private static final VarHandle LONGS = MethodHandles.arrayElementVarHandle(long[].class);

  private record Cell(Thread thread, long[] values) {}

  private final ThreadLocal<long[]> local = new ThreadLocal<>();
  private final List<Cell> cells = new ArrayList<>(); // guarded by this
  private final long[] retired = new long[WIDTH]; // guarded by this

  /** This thread's cell, created on its first call. */
  long[] cell() {
    long[] values = local.get();
    if (values == null) {
      values = new long[WIDTH];
      synchronized (this) {
        for (int i = cells.size() - 1; i >= 0; i--) {
          Cell cell = cells.get(i);
          if (!cell.thread().isAlive()) {
            fold(retired, cell.values());
            cells.remove(i);
          }
        }
        cells.add(new Cell(Thread.currentThread(), values));
      }
      local.set(values);
    }
    return values;
  }

  /** Adds {@code delta} to the owner's cell; only the owner thread may call this. */
  static void add(long[] cell, int index, long delta) {
    LONGS.setOpaque(cell, index, (long) LONGS.getOpaque(cell, index) + delta);
  }

  /** Raises the owner's high-water mark; only the owner thread may call this. */
  static void max(long[] cell, int index, long value) {
    if (value > (long) LONGS.getOpaque(cell, index)) LONGS.setOpaque(cell, index, value);
  }

  /** Every counter, summed (high-water marks maximized) over live and retired cells. */
  synchronized long[] snapshot() {
    long[] total = retired.clone();
    for (Cell cell : cells) fold(total, cell.values());
    return total;
  }

  private static void fold(long[] into, long[] cell) {
    for (int i = 0; i < WIDTH; i++) {
      long value = (long) LONGS.getOpaque(cell, i);
      into[i] = i == SCRATCH_HIGH_WATER ? Math.max(into[i], value) : into[i] + value;
    }
  }
}
