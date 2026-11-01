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

import java.util.BitSet;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestDocSetViews extends LuceneTestCase {
  private static final MemoryLedger.Owner OWNER = new MemoryLedger.Owner("sets", "core", 0, "sets");

  private static MemoryLedger ledger() {
    return new MemoryLedger(new MemoryLedger.Limits(1 << 20, 1 << 20, 0), 2 << 20);
  }

  public void testRepresentationsAndOwnedAlgebra() throws Exception {
    MemoryLedger ledger = ledger();
    AtomicInteger live = new AtomicInteger();
    for (int iteration = 0; iteration < 20; iteration++) {
      int length = 1 + random().nextInt(513);
      BitSet expected = new BitSet(length);
      for (int doc = 0; doc < length; doc++) if (random().nextBoolean()) expected.set(doc);
      int[] docs = expected.stream().toArray();
      for (DocSetView.Representation representation : DocSetView.Representation.values()) {
        try (DocSetViews.Owned owned =
                DocSetViews.copyOf(
                    docs, length, representation, ledger, OWNER, TestMemoryStores.allocator(live));
            DocSetLease lease = owned.tryAcquire()) {
          assertEquals(docs.length, lease.view().cardinality());
          for (int doc = 0; doc <= length; doc++) {
            assertEquals(expected.get(doc), lease.view().contains(doc));
            int next = expected.nextSetBit(doc);
            assertEquals(
                next < 0 ? DocIdSetIterator.NO_MORE_DOCS : next, lease.view().nextDoc(doc));
          }
          try (DocSetViews.Owned difference =
                  DocSetViews.combine(
                      lease.view(),
                      lease.view(),
                      DocSetViews.Operation.DIFFERENCE,
                      ledger,
                      OWNER,
                      TestMemoryStores.allocator(live),
                      true);
              DocSetLease empty = difference.tryAcquire()) {
            assertEquals(0, empty.view().cardinality());
            assertEquals(DocIdSetIterator.NO_MORE_DOCS, empty.view().nextDoc(0));
          }
        }
      }
    }
    assertEquals(0, live.get());
    assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
    assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
  }

  public void testRetiredIteratorPinsUntilExhaustionOrReaderClose() throws Exception {
    MemoryLedger ledger = ledger();
    AtomicInteger live = new AtomicInteger();
    DocSetViews.Owned owned =
        DocSetViews.copyOf(
            new int[] {0, 64, 128},
            129,
            DocSetView.Representation.NATIVE_DENSE,
            ledger,
            OWNER,
            TestMemoryStores.allocator(live));
    MemoryDocIdSet set = new MemoryDocIdSet(owned);
    DocIdSetIterator first = set.iterator(), abandoned = set.iterator();
    assertEquals(0, first.nextDoc());
    set.close();
    assertNull(set.iterator());
    assertEquals(1, live.get());
    assertEquals(64, first.nextDoc());
    assertEquals(128, first.nextDoc());
    assertEquals(DocIdSetIterator.NO_MORE_DOCS, first.nextDoc());
    assertEquals(1, live.get());
    assertEquals(0, abandoned.nextDoc());
    set.closeReaders();
    assertEquals(0, live.get());
    assertEquals(DocIdSetIterator.NO_MORE_DOCS, abandoned.nextDoc());
    assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
  }

  public void testOutputAdmissionDoesNotRefundInputs() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(0, 24, 0), 24);
    AtomicInteger live = new AtomicInteger();
    try (DocSetViews.Owned owned =
            DocSetViews.copyOf(
                new int[] {0, 128},
                129,
                DocSetView.Representation.NATIVE_DENSE,
                ledger,
                OWNER,
                TestMemoryStores.allocator(live));
        DocSetLease lease = owned.tryAcquire()) {
      expectThrows(
          RequestMemoryScope.AdmissionException.class,
          () ->
              DocSetViews.combine(
                  lease.view(),
                  lease.view(),
                  DocSetViews.Operation.UNION,
                  ledger,
                  OWNER,
                  TestMemoryStores.allocator(live),
                  true));
      assertEquals(24, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
      assertEquals(128, lease.view().nextDoc(1));
    }
    assertEquals(0, live.get());
  }

  public void testCompressedCursorIsForwardOnlyAndLeaseOwnsStorage() throws Exception {
    MemoryLedger ledger = ledger();
    int[] docs = new int[4096];
    for (int i = 0; i < docs.length; i++) docs[i] = i * 17;
    DocSetViews.Owned owned =
        DocSetViews.copyOf(docs, 70000, DocSetView.Representation.COMPRESSED, ledger, OWNER, null);
    DocSetLease lease = owned.tryAcquire();
    DocSetLease.LeasedIterator iterator = lease.iterator();
    owned.close();
    assertEquals(0, iterator.nextDoc());
    assertEquals(17, iterator.nextDoc());
    assertEquals(1700, iterator.advance(1699));
    for (int i = 101; i < docs.length; i++) assertEquals(docs[i], iterator.nextDoc());
    assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
  }

  public void testCacheOwnerAndFinalIteratorFailuresRemainRecoverable() throws Exception {
    for (boolean pinned : new boolean[] {false, true}) {
      MemoryLedger ledger = ledger();
      AtomicInteger live = new AtomicInteger();
      AtomicInteger attempts = new AtomicInteger();
      MemoryDocIdSet set =
          new MemoryDocIdSet(
              DocSetViews.copyOf(
                  new int[] {0},
                  64,
                  DocSetView.Representation.NATIVE_DENSE,
                  ledger,
                  OWNER,
                  failingCloseAllocator(live, attempts)));
      if (pinned) {
        DocIdSetIterator iterator = set.iterator();
        set.close();
        assertEquals(0, iterator.nextDoc());
        assertEquals(0, attempts.get());
        expectThrows(IllegalStateException.class, iterator::nextDoc);
      } else {
        expectThrows(IllegalStateException.class, set::close);
        // Repeat cache/reader retirement must not silently retry a failed physical close.
        set.closeReaders();
        assertEquals(1, attempts.get());
      }
      assertEquals(1, live.get());
      assertEquals(8L, ledger.snapshot().failedReclamations());
      assertEquals(1, ManagedMemory.failedReclamations(ledger, 10).size());
      if (pinned) ManagedMemory.retryFailedReclamations(ledger);
      else set.retryFailedReclamation();
      assertEquals(2, attempts.get());
      assertEquals(0, live.get());
      assertEquals(0L, ledger.snapshot().failedReclamations());
      assertEquals(0, ManagedMemory.failedReclamations(ledger, 10).size());
      ManagedMemory.retryFailedReclamations(ledger);
      assertEquals(2, attempts.get());
    }
  }

  private static NativeAllocator failingCloseAllocator(AtomicInteger live, AtomicInteger attempts) {
    NativeAllocator delegate = TestMemoryStores.allocator(live);
    return bytes -> {
      NativeAllocator.Allocation allocation = delegate.allocate(bytes);
      return new NativeAllocator.Allocation() {
        @Override
        public java.nio.ByteBuffer buffer() {
          return allocation.buffer();
        }

        @Override
        public long capacity() {
          return allocation.capacity();
        }

        @Override
        public boolean isAlive() {
          return allocation.isAlive();
        }

        @Override
        public void close() {
          if (attempts.incrementAndGet() == 1)
            throw new IllegalStateException("injected physical close failure");
          allocation.close();
        }
      };
    };
  }
}
