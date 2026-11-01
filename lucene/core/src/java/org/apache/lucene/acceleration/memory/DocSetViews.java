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

import java.io.IOException;
import java.util.Objects;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * Owned document-set builders and operations. Every output is admitted while inputs remain live.
 *
 * @lucene.experimental
 */
public final class DocSetViews {
  private DocSetViews() {}

  /** Builds dense storage directly from an iterator, with no intermediate document array. */
  public static Owned dense(
      DocIdSetIterator iterator,
      int maxDoc,
      boolean nativeStorage,
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      NativeAllocator allocator)
      throws IOException {
    if (maxDoc < 0) throw new IllegalArgumentException("negative document bound");
    LongStore store =
        MemoryStores.longs(
            ledger,
            owner,
            nativeStorage ? MemoryLedger.Kind.NATIVE : MemoryLedger.Kind.HEAP,
            (int) (((long) maxDoc + 63) >>> 6),
            allocator,
            MemoryLedger.Purpose.CACHE_VALUE);
    try {
      for (int i = 0; i < store.length(); i++) store.set(i, 0);
      int count = 0;
      for (int doc = iterator.nextDoc();
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = iterator.nextDoc()) {
        Objects.checkIndex(doc, maxDoc);
        long word = store.get(doc >>> 6), bit = 1L << doc;
        if ((word & bit) == 0) count++;
        store.set(doc >>> 6, word | bit);
      }
      store.freeze();
      return new Owned(new Dense(store, maxDoc, count), MemoryStores.charge(store));
    } catch (IOException | RuntimeException | Error failure) {
      store.close();
      throw failure;
    }
  }

  /** Cache-owned allocation; retirement closes admission but preserves active readers. */
  public static final class Owned implements AutoCloseable {
    private final SharedValueLease.Entry<DocSetView> entry;
    private final long capacity;
    private final MemoryLedger.Charge charge;

    Owned(DocSetView view, MemoryLedger.Charge charge) {
      this.charge = charge;
      capacity = view.capacityBytes();
      entry = new SharedValueLease.Entry<>(view, view::close, charge);
    }

    /** Acquires one operation-wide reader; null means already retired. */
    public DocSetLease tryAcquire() {
      SharedValueLease<DocSetView> lease = entry.tryAcquire();
      return lease == null ? null : new DocSetLease(lease);
    }

    /** Owned capacity remains charged after retirement while readers are pinned. */
    public long capacityBytes() {
      return capacity;
    }

    /** Transfers durable ownership without refunding or charging the node twice. */
    public boolean transferOwner(MemoryLedger.Owner owner) {
      return charge.transferOwner(owner, MemoryLedger.State.PUBLISHED);
    }

    /** Reader count for pressure diagnostics. */
    public long activeLeases() {
      return entry.activeLeases();
    }

    /** Explicit retry of diagnosed failed physical reclamation. */
    public void retryFailedReclamation() {
      entry.retryFailedReclamation();
    }

    @Override
    public void close() {
      entry.retire();
    }
  }

  /** Builds a selected representation from strictly increasing document IDs. */
  public static Owned copyOf(
      int[] docs,
      int maxDoc,
      DocSetView.Representation representation,
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      NativeAllocator allocator) {
    Objects.requireNonNull(docs);
    Objects.requireNonNull(representation);
    if (maxDoc < 0) throw new IllegalArgumentException("negative document bound");
    int previous = -1;
    long compressedBytes = 0;
    for (int doc : docs) {
      if (doc <= previous || doc >= maxDoc)
        throw new IllegalArgumentException("unsorted document IDs");
      int delta = doc - previous;
      do {
        compressedBytes++;
        delta >>>= 7;
      } while (delta != 0);
      previous = doc;
    }
    if (representation == DocSetView.Representation.SORTED) {
      IntStore store =
          MemoryStores.ints(
              ledger,
              owner,
              MemoryLedger.Kind.HEAP,
              docs.length,
              null,
              MemoryLedger.Purpose.CACHE_VALUE);
      try {
        for (int i = 0; i < docs.length; i++) store.set(i, docs[i]);
        store.freeze();
        return new Owned(new Sorted(store, maxDoc), MemoryStores.charge(store));
      } catch (RuntimeException | Error failure) {
        store.close();
        throw failure;
      }
    }
    if (representation == DocSetView.Representation.COMPRESSED) {
      MemoryStores.Bytes store =
          MemoryStores.bytes(
              ledger, owner, Math.toIntExact(compressedBytes), MemoryLedger.Purpose.CACHE_VALUE);
      try {
        previous = -1;
        int offset = 0;
        for (int doc : docs) {
          int delta = doc - previous;
          while ((delta & ~127) != 0) {
            store.set(offset++, delta | 128);
            delta >>>= 7;
          }
          store.set(offset++, delta);
          previous = doc;
        }
        store.freeze();
        return new Owned(new Compressed(store, maxDoc, docs.length), store.charge());
      } catch (RuntimeException | Error failure) {
        store.close();
        throw failure;
      }
    }
    MemoryLedger.Kind kind =
        representation == DocSetView.Representation.NATIVE_DENSE
            ? MemoryLedger.Kind.NATIVE
            : MemoryLedger.Kind.HEAP;
    LongStore store =
        MemoryStores.longs(
            ledger,
            owner,
            kind,
            (int) (((long) maxDoc + 63) >>> 6),
            allocator,
            MemoryLedger.Purpose.CACHE_VALUE);
    try {
      for (int i = 0; i < store.length(); i++) store.set(i, 0);
      for (int doc : docs) store.set(doc >>> 6, store.get(doc >>> 6) | (1L << doc));
      store.freeze();
      return new Owned(new Dense(store, maxDoc, docs.length), MemoryStores.charge(store));
    } catch (RuntimeException | Error failure) {
      store.close();
      throw failure;
    }
  }

  /** Chooses by exact payload cost; native selection requires an explicit threshold and budget. */
  public static Owned adaptive(
      int[] docs,
      int maxDoc,
      boolean allowNative,
      long minimumNativeBytes,
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      NativeAllocator allocator) {
    long dense = ((long) maxDoc + 63) / 64 * 8;
    long sorted = MemoryLedger.bytes(docs.length, 4);
    long compressed = 0;
    int previous = -1;
    for (int doc : docs) {
      int delta = doc - previous;
      do {
        compressed++;
        delta >>>= 7;
      } while (delta != 0);
      previous = doc;
    }
    DocSetView.Representation selected =
        compressed < Math.min(sorted, dense)
            ? DocSetView.Representation.COMPRESSED
            : sorted < dense ? DocSetView.Representation.SORTED : DocSetView.Representation.DENSE;
    if (allowNative && dense >= minimumNativeBytes && dense <= sorted) {
      try {
        return copyOf(
            docs, maxDoc, DocSetView.Representation.NATIVE_DENSE, ledger, owner, allocator);
      } catch (RequestMemoryScope.AdmissionException denied) {
        if (denied.reason() != RequestMemoryScope.Failure.NODE_CAPACITY) throw denied;
      }
    }
    return copyOf(docs, maxDoc, selected, ledger, owner, allocator);
  }

  /** Immutable owned union, intersection, or difference without an uncharged temporary array. */
  public static Owned combine(
      DocSetView left,
      DocSetView right,
      Operation operation,
      MemoryLedger ledger,
      MemoryLedger.Owner owner,
      NativeAllocator allocator,
      boolean nativeOutput) {
    if (left.maxDoc() != right.maxDoc())
      throw new IllegalArgumentException("different document bounds");
    int maxDoc = left.maxDoc();
    LongStore output =
        MemoryStores.longs(
            ledger,
            owner,
            nativeOutput ? MemoryLedger.Kind.NATIVE : MemoryLedger.Kind.HEAP,
            (int) (((long) maxDoc + 63) >>> 6),
            allocator,
            MemoryLedger.Purpose.REQUEST_SCRATCH);
    try {
      for (int word = 0; word < output.length(); word++) output.set(word, 0);
      DocIdSetIterator a = left.iterator(), b = right.iterator();
      int first = a.nextDoc(), second = b.nextDoc(), count = 0;
      switch (operation) {
        case UNION -> {
          while (first != DocIdSetIterator.NO_MORE_DOCS
              || second != DocIdSetIterator.NO_MORE_DOCS) {
            int doc = Math.min(first, second);
            output.set(doc >>> 6, output.get(doc >>> 6) | (1L << doc));
            count++;
            if (first == doc) first = a.nextDoc();
            if (second == doc) second = b.nextDoc();
          }
        }
        case INTERSECTION -> {
          while (first != DocIdSetIterator.NO_MORE_DOCS
              && second != DocIdSetIterator.NO_MORE_DOCS) {
            if (first < second) first = a.advance(second);
            else if (second < first) second = b.advance(first);
            else {
              output.set(first >>> 6, output.get(first >>> 6) | (1L << first));
              count++;
              first = a.nextDoc();
              second = b.nextDoc();
            }
          }
        }
        case DIFFERENCE -> {
          while (first != DocIdSetIterator.NO_MORE_DOCS) {
            if (second < first) second = b.advance(first);
            if (first != second) {
              output.set(first >>> 6, output.get(first >>> 6) | (1L << first));
              count++;
            }
            first = a.nextDoc();
          }
        }
      }
      output.freeze();
      return new Owned(new Dense(output, maxDoc, count), MemoryStores.charge(output));
    } catch (IOException failure) {
      output.close();
      throw new java.io.UncheckedIOException(failure);
    } catch (RuntimeException | Error failure) {
      output.close();
      throw failure;
    }
  }

  /** Set algebra operations. */
  public enum Operation {
    UNION,
    INTERSECTION,
    DIFFERENCE
  }

  private record Sorted(IntStore store, int maxDoc) implements DocSetView {
    @Override
    public int cardinality() {
      return store.length();
    }

    private int lowerBound(int target) {
      int lo = 0, hi = store.length();
      while (lo < hi) {
        int mid = (lo + hi) >>> 1;
        if (store.get(mid) < target) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }

    @Override
    public boolean contains(int doc) {
      int index = lowerBound(doc);
      return index < store.length() && store.get(index) == doc;
    }

    @Override
    public int nextDoc(int target) {
      int index = lowerBound(Math.max(0, target));
      return index == store.length() ? DocIdSetIterator.NO_MORE_DOCS : store.get(index);
    }

    @Override
    public DocIdSetIterator iterator() {
      return new DocIdSetIterator() {
        private int index = -1, doc = -1;

        @Override
        public int docID() {
          return doc;
        }

        @Override
        public long cost() {
          return cardinality();
        }

        @Override
        public int nextDoc() {
          if (doc == NO_MORE_DOCS) return doc;
          return doc = ++index < store.length() ? store.get(index) : NO_MORE_DOCS;
        }

        @Override
        public int advance(int target) {
          if (doc == NO_MORE_DOCS) return doc;
          index = lowerBound(Math.max(target, doc + 1));
          return doc = index < store.length() ? store.get(index) : NO_MORE_DOCS;
        }
      };
    }

    @Override
    public long capacityBytes() {
      return store.capacityBytes();
    }

    @Override
    public Representation representation() {
      return Representation.SORTED;
    }

    @Override
    public void close() {
      store.close();
    }
  }

  private record Compressed(MemoryStores.Bytes store, int maxDoc, int cardinality)
      implements DocSetView {
    @Override
    public int nextDoc(int target) {
      int previous = -1, offset = 0;
      for (int i = 0; i < cardinality; i++) {
        int delta = 0, shift = 0, value;
        do {
          value = store.get(offset++);
          delta |= (value & 127) << shift;
          shift += 7;
        } while ((value & 128) != 0);
        previous += delta;
        if (previous >= Math.max(0, target)) return previous;
      }
      return DocIdSetIterator.NO_MORE_DOCS;
    }

    @Override
    public boolean contains(int doc) {
      return doc >= 0 && doc < maxDoc && nextDoc(doc) == doc;
    }

    @Override
    public DocIdSetIterator iterator() {
      return new DocIdSetIterator() {
        private int offset, consumed, doc = -1;

        @Override
        public int docID() {
          return doc;
        }

        @Override
        public long cost() {
          return cardinality;
        }

        @Override
        public int nextDoc() {
          if (doc == NO_MORE_DOCS) return doc;
          if (consumed == cardinality) return doc = NO_MORE_DOCS;
          int delta = 0, shift = 0, value;
          do {
            value = store.get(offset++);
            delta |= (value & 127) << shift;
            shift += 7;
          } while ((value & 128) != 0);
          consumed++;
          return doc += delta;
        }

        @Override
        public int advance(int target) {
          if (doc == NO_MORE_DOCS) return doc;
          target = Math.max(target, doc + 1);
          do {
            nextDoc();
          } while (doc < target);
          return doc;
        }
      };
    }

    @Override
    public long capacityBytes() {
      return store.capacityBytes();
    }

    @Override
    public Representation representation() {
      return Representation.COMPRESSED;
    }

    @Override
    public void close() {
      store.close();
    }
  }

  private record Dense(LongStore store, int maxDoc, int cardinality) implements DocSetView {
    @Override
    public boolean contains(int doc) {
      return doc >= 0 && doc < maxDoc && (store.get(doc >>> 6) & (1L << doc)) != 0;
    }

    @Override
    public int nextDoc(int target) {
      target = Math.max(0, target);
      if (target >= maxDoc) return DocIdSetIterator.NO_MORE_DOCS;
      int word = target >>> 6;
      long bits = store.get(word) & (-1L << target);
      while (true) {
        if (bits != 0) return (word << 6) + Long.numberOfTrailingZeros(bits);
        if (++word == store.length()) return DocIdSetIterator.NO_MORE_DOCS;
        bits = store.get(word);
      }
    }

    @Override
    public long capacityBytes() {
      return store.capacityBytes();
    }

    @Override
    public Representation representation() {
      return store.kind() == MemoryLedger.Kind.NATIVE
          ? Representation.NATIVE_DENSE
          : Representation.DENSE;
    }

    @Override
    public void close() {
      store.close();
    }
  }
}
