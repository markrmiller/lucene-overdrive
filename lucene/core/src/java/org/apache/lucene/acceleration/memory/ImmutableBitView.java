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
import java.util.Objects;

/**
 * Immutable dense bit view over explicitly admitted native storage. All reads require an enclosing
 * request/worker lifetime. No raw buffer escapes this view. This is an optional request-local
 * storage representation, not an index format; extending its lifetime requires retaining the owner
 * scope.
 *
 * @lucene.experimental
 */
public final class ImmutableBitView {
  private final ByteBuffer words;
  private final int length;
  private final long cardinality;
  private final RequestMemoryScope owner;

  private ImmutableBitView(
      ByteBuffer words, int length, long cardinality, RequestMemoryScope owner) {
    this.words = words.asReadOnlyBuffer().order(ByteOrder.nativeOrder());
    this.length = length;
    this.cardinality = cardinality;
    this.owner = owner;
  }

  /** Copies a heap baseline to admitted native storage, zeroing unused tail bits. */
  public static ImmutableBitView copyOf(long[] source, int length, RequestMemoryScope scope) {
    Objects.requireNonNull(source);
    Objects.requireNonNull(scope);
    if (length < 0) throw new IllegalArgumentException("negative bit length");
    int count = (int) (((long) length + 63) >>> 6);
    if (source.length < count) throw new IllegalArgumentException("insufficient source words");
    ByteBuffer storage = scope.allocateLongs(count);
    long cardinality = 0;
    for (int i = 0; i < count; i++) {
      long word = source[i];
      if (i == count - 1 && (length & 63) != 0) word &= (1L << (length & 63)) - 1;
      storage.putLong(i * Long.BYTES, word);
      cardinality += Long.bitCount(word);
    }
    return new ImmutableBitView(storage, length, cardinality, scope);
  }

  public int length() {
    owner.checkAccessible();
    return length;
  }

  public long cardinality() {
    owner.checkAccessible();
    return cardinality;
  }

  public boolean contains(int bit) {
    owner.checkAccessible();
    if (bit < 0 || bit >= length) return false;
    return (words.getLong((bit >>> 6) * Long.BYTES) & (1L << bit)) != 0;
  }

  /** Ordered iteration, with -1 after the last bit. */
  public int nextSetBit(int start) {
    owner.checkAccessible();
    if (start < 0) throw new IllegalArgumentException("negative start");
    if (start >= length) return -1;
    int wordIndex = start >>> 6;
    long word = words.getLong(wordIndex * Long.BYTES) & (-1L << start);
    int count = words.capacity() / Long.BYTES;
    while (true) {
      if (word != 0) return (wordIndex << 6) + Long.numberOfTrailingZeros(word);
      if (++wordIndex == count) return -1;
      word = words.getLong(wordIndex * Long.BYTES);
    }
  }

  public long intersectionCardinality(ImmutableBitView other) {
    checkLength(other);
    long result = 0;
    for (int i = 0; i < words.capacity(); i += Long.BYTES) {
      result += Long.bitCount(words.getLong(i) & other.words.getLong(i));
    }
    return result;
  }

  public long unionCardinality(ImmutableBitView other) {
    checkLength(other);
    long result = 0;
    for (int i = 0; i < words.capacity(); i += Long.BYTES) {
      result += Long.bitCount(words.getLong(i) | other.words.getLong(i));
    }
    return result;
  }

  public long differenceCardinality(ImmutableBitView other) {
    checkLength(other);
    long result = 0;
    for (int i = 0; i < words.capacity(); i += Long.BYTES) {
      result += Long.bitCount(words.getLong(i) & ~other.words.getLong(i));
    }
    return result;
  }

  private void checkLength(ImmutableBitView other) {
    owner.checkAccessible();
    other.owner.checkAccessible();
    if (other.length != length) throw new IllegalArgumentException("different bit universes");
  }
}
