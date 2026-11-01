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

/**
 * Owned, bounds-checked int storage. Closing the owner invalidates every borrowed view.
 *
 * @lucene.experimental
 */
public interface IntStore extends AutoCloseable {
  /** Number of elements. */
  int length();

  /** Reads one element while the owner is alive. */
  int get(int index);

  /** Updates a mutable builder; frozen stores reject writes. */
  void set(int index, int value);

  /** Stops future writes and returns this storage. */
  IntStore freeze();

  /** Actual charged backing capacity, including allocator padding. */
  long capacityBytes();

  /** Storage accounting dimension. */
  MemoryLedger.Kind kind();

  /** Releases backing storage; idempotent after successful reclamation. */
  @Override
  void close();
}
