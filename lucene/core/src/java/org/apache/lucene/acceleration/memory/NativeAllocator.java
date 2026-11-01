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

/**
 * Deterministic backing allocation contract. Implementations cannot retain buffers after close.
 * Calls occur outside admission locks and only after peak capacity has been reserved.
 *
 * @lucene.experimental
 */
public interface NativeAllocator {
  /** Returns the full rounded capacity to reserve, including controlled padding. */
  default long allocationSize(long requestedBytes) {
    return requestedBytes;
  }

  /** Allocates at most the capacity reported by {@link #allocationSize(long)}. */
  Allocation allocate(long requestedBytes);

  /** One backing allocation with deterministic physical close. */
  interface Allocation extends AutoCloseable {
    /** Borrowed view; its lifetime is bounded by the enclosing request/worker scope. */
    ByteBuffer buffer();

    /** Full backing capacity, counted exactly once (not the sum of slice lengths). */
    long capacity();

    /**
     * Whether physical storage remains live, for explicit recovery after a failed close. A default
     * conservative answer keeps capacity charged until close succeeds.
     */
    default boolean isAlive() {
      return true;
    }

    /**
     * Throws on failed physical reclamation; the ledger must retain its charge. Close is
     * idempotent.
     */
    @Override
    void close();
  }
}
