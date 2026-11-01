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
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestMemoryStores extends LuceneTestCase {
  private static final MemoryLedger.Owner OWNER =
      new MemoryLedger.Owner("stores", "core", 0, "stores");

  private static MemoryLedger ledger() {
    return new MemoryLedger(new MemoryLedger.Limits(1 << 20, 1 << 20, 1 << 20), 3 << 20);
  }

  static NativeAllocator allocator(AtomicInteger live) {
    return bytes -> {
      live.incrementAndGet();
      return new NativeAllocator.Allocation() {
        final ByteBuffer buffer = ByteBuffer.allocate(Math.toIntExact(bytes));
        boolean closed;

        @Override
        public ByteBuffer buffer() {
          return buffer;
        }

        @Override
        public long capacity() {
          return bytes;
        }

        @Override
        public boolean isAlive() {
          return !closed;
        }

        @Override
        public void close() {
          if (!closed) {
            closed = true;
            live.decrementAndGet();
          }
        }
      };
    };
  }

  public void testTypedParityFreezeAndClose() {
    MemoryLedger ledger = ledger();
    AtomicInteger live = new AtomicInteger();
    for (MemoryLedger.Kind kind :
        new MemoryLedger.Kind[] {MemoryLedger.Kind.HEAP, MemoryLedger.Kind.NATIVE}) {
      IntStore ints =
          MemoryStores.ints(
              ledger, OWNER, kind, 129, allocator(live), MemoryLedger.Purpose.REQUEST_SCRATCH);
      LongStore longs =
          MemoryStores.longs(
              ledger, OWNER, kind, 129, allocator(live), MemoryLedger.Purpose.REQUEST_SCRATCH);
      FloatStore floats =
          MemoryStores.floats(
              ledger, OWNER, kind, 129, allocator(live), MemoryLedger.Purpose.REQUEST_SCRATCH);
      for (int i = 0; i < 129; i++) {
        ints.set(i, i * -31);
        longs.set(i, Long.MIN_VALUE + i);
        floats.set(i, Float.intBitsToFloat(i));
      }
      ints.freeze();
      longs.freeze();
      floats.freeze();
      for (int i = 0; i < 129; i++) {
        assertEquals(i * -31, ints.get(i));
        assertEquals(Long.MIN_VALUE + i, longs.get(i));
        assertEquals(i, Float.floatToRawIntBits(floats.get(i)));
      }
      expectThrows(IndexOutOfBoundsException.class, () -> ints.get(129));
      expectThrows(IllegalStateException.class, () -> longs.set(0, 42));
      ints.close();
      ints.close();
      longs.close();
      floats.close();
      expectThrows(IllegalStateException.class, () -> ints.get(0));
      assertEquals(0, ledger.snapshot().charged().get(kind).longValue());
    }
    assertEquals(0, live.get());
  }

  public void testMappedPhysicalCloseFailureRetainsCharge() {
    MemoryLedger ledger = ledger();
    MemoryLedger.Charge charge;
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.MAPPED, 4096, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      charge = reservation.commit(4096, MemoryLedger.State.PUBLISHED);
    }
    AtomicInteger attempts = new AtomicInteger();
    IntStore store =
        MemoryStores.mappedInts(
            ByteBuffer.allocate(16),
            charge,
            () -> {
              if (attempts.incrementAndGet() == 1)
                throw new IllegalStateException("injected unmap failure");
            });
    store.set(0, 73);
    store.freeze();
    expectThrows(IllegalStateException.class, store::close);
    assertEquals(MemoryLedger.State.FAILED_RECLAMATION, charge.snapshot().state());
    assertEquals(4096, ledger.snapshot().charged().get(MemoryLedger.Kind.MAPPED).longValue());
    store.close();
    store.close();
    assertEquals(2, attempts.get());
    assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.MAPPED).longValue());
  }

  public void testCreationFailureRollsBack() {
    MemoryLedger ledger = ledger();
    NativeAllocator allocator =
        bytes -> {
          throw new IllegalStateException("injected creation failure");
        };
    expectThrows(
        IllegalStateException.class,
        () ->
            MemoryStores.longs(
                ledger,
                OWNER,
                MemoryLedger.Kind.NATIVE,
                3,
                allocator,
                MemoryLedger.Purpose.CACHE_VALUE));
    assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
    assertEquals(0, ledger.snapshot().reserved().get(MemoryLedger.Kind.NATIVE).longValue());
  }

  public void testFailedContractUnwindRetainsHandleForExplicitRecovery() {
    MemoryLedger ledger = ledger();
    AtomicInteger closes = new AtomicInteger();
    NativeAllocator invalid =
        bytes ->
            new NativeAllocator.Allocation() {
              @Override
              public ByteBuffer buffer() {
                return ByteBuffer.allocate(Math.toIntExact(bytes));
              }

              @Override
              public long capacity() {
                return bytes + 1;
              }

              @Override
              public void close() {
                if (closes.incrementAndGet() == 1)
                  throw new IllegalStateException("injected close failure");
              }
            };
    expectThrows(
        IllegalStateException.class,
        () ->
            MemoryStores.ints(
                ledger,
                OWNER,
                MemoryLedger.Kind.NATIVE,
                4,
                invalid,
                MemoryLedger.Purpose.CACHE_VALUE));
    assertEquals(16, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
    MemoryStores.retryFailedReclamations(ledger);
    MemoryStores.retryFailedReclamations(ledger);
    assertEquals(2, closes.get());
    assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
  }

  public void testStandaloneTypedCloseFailuresRetainOperatorRecoveryHandles() throws Exception {
    for (int type = 0; type < 3; type++) {
      MemoryLedger ledger = ledger();
      AtomicInteger attempts = new AtomicInteger();
      AtomicInteger releases = new AtomicInteger();
      createFailedTypedStore(type, ledger, attempts, releases);
      assertEquals(1, attempts.get());
      assertEquals(1, ManagedMemory.failedReclamations(ledger, 10).size());
      long held = ledger.snapshot().failedReclamations();
      assertTrue(held > 0);
      expectThrows(IllegalStateException.class, () -> MemoryStores.retryFailedReclamations(ledger));
      assertEquals(2, attempts.get());
      assertEquals(held, ledger.snapshot().failedReclamations());
      assertEquals(1, ManagedMemory.failedReclamations(ledger, 10).size());
      ManagedMemory.retryFailedReclamations(ledger);
      assertEquals(3, attempts.get());
      assertEquals(1, releases.get());
      assertEquals(0L, ledger.snapshot().failedReclamations());
      assertEquals(0, ManagedMemory.failedReclamations(ledger, 10).size());
      MemoryStores.retryFailedReclamations(ledger);
      assertEquals(3, attempts.get());
    }
  }

  private void createFailedTypedStore(
      int type, MemoryLedger ledger, AtomicInteger attempts, AtomicInteger releases)
      throws Exception {
    NativeAllocator allocator =
        bytes ->
            new NativeAllocator.Allocation() {
              final ByteBuffer buffer = ByteBuffer.allocate(Math.toIntExact(bytes));

              @Override
              public ByteBuffer buffer() {
                return buffer;
              }

              @Override
              public long capacity() {
                return bytes;
              }

              @Override
              public void close() {
                if (attempts.incrementAndGet() <= 2)
                  throw new IllegalStateException("injected typed-store close failure");
                releases.incrementAndGet();
              }
            };
    AutoCloseable store =
        switch (type) {
          case 0 ->
              MemoryStores.ints(
                  ledger,
                  OWNER,
                  MemoryLedger.Kind.NATIVE,
                  4,
                  allocator,
                  MemoryLedger.Purpose.REQUEST_SCRATCH);
          case 1 ->
              MemoryStores.longs(
                  ledger,
                  OWNER,
                  MemoryLedger.Kind.NATIVE,
                  4,
                  allocator,
                  MemoryLedger.Purpose.REQUEST_SCRATCH);
          default ->
              MemoryStores.floats(
                  ledger,
                  OWNER,
                  MemoryLedger.Kind.NATIVE,
                  4,
                  allocator,
                  MemoryLedger.Purpose.REQUEST_SCRATCH);
        };
    expectThrows(IllegalStateException.class, store::close);
  }
}
