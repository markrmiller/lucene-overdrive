/* Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.acceleration;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Adaptive admission for short leaf calls; elapsed time cannot bound their first execution.
 *
 * @lucene.experimental
 */
public final class CriticalCallBudget {
  public static final long MAXIMUM_NANOS = 100_000;

  /** Completed-call counters and the current admission state. */
  public record Snapshot(long calls, long slowCalls, long maximumNanos, boolean disabled) {
    public static final Snapshot EMPTY = new Snapshot(0, 0, 0, false);
  }

  private final LongAdder calls = new LongAdder(), slowCalls = new LongAdder();
  private final AtomicLong maximumNanos = new AtomicLong();
  private volatile boolean disabled;

  public boolean disabled() {
    return disabled;
  }

  public boolean allows(Capability capability, long workBytes) {
    return !disabled && NativeBridge.criticalCallAllowed(capability, workBytes);
  }

  /** Records a completed critical call, including one that failed or threw. */
  public void record(long elapsedNanos) {
    if (elapsedNanos < 0) throw new IllegalArgumentException("negative critical-call duration");
    calls.increment();
    long previous = maximumNanos.get();
    while (elapsedNanos > previous && !maximumNanos.compareAndSet(previous, elapsedNanos))
      previous = maximumNanos.get();
    if (elapsedNanos > MAXIMUM_NANOS) {
      disabled = true;
      slowCalls.increment();
    }
  }

  public Snapshot snapshot() {
    return new Snapshot(calls.sum(), slowCalls.sum(), maximumNanos.get(), disabled);
  }
}
