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
import java.util.BitSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestMemoryOwnership extends LuceneTestCase {
  private static final MemoryLedger.Owner OWNER =
      new MemoryLedger.Owner("query", "core", 1, "request");

  private static class Allocator implements NativeAllocator {
    final AtomicInteger live = new AtomicInteger();
    boolean failAllocation;
    boolean failClose;
    boolean failLivenessProbe;

    @Override
    public Allocation allocate(long bytes) {
      if (failAllocation) throw new IllegalStateException("injected allocation failure");
      live.incrementAndGet();
      return new Allocation() {
        private final ByteBuffer buffer = ByteBuffer.allocate((int) bytes);
        private boolean closed;

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
          if (failLivenessProbe) throw new IllegalStateException("injected liveness probe failure");
          return closed == false;
        }

        @Override
        public void close() {
          if (failClose) throw new IllegalStateException("injected close failure");
          if (closed == false) {
            closed = true;
            live.decrementAndGet();
          }
        }
      };
    }
  }

  private static MemoryLedger ledger() {
    return new MemoryLedger(new MemoryLedger.Limits(1024, 1024, 0), 1024);
  }

  public void testWorkerPinsAndCancellation() {
    MemoryLedger ledger = ledger();
    Allocator allocator = new Allocator();
    RequestMemoryScope parent = new RequestMemoryScope(ledger, OWNER, allocator);
    ByteBuffer buffer = parent.allocate(16);
    RequestMemoryScope.WorkerScope worker = parent.forkWorker();
    parent.cancel();
    assertEquals(
        RequestMemoryScope.Failure.CANCELLED,
        expectThrows(RequestMemoryScope.AdmissionException.class, () -> worker.allocate(1))
            .reason());
    parent.close();
    assertEquals(1, allocator.live.get());
    buffer.putInt(0, 42); // already-running task retains its storage after parent close
    assertEquals(42, buffer.getInt(0));
    worker.close();
    worker.close();
    assertEquals(0, allocator.live.get());
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testConfinedScopeRejectsForeignOwnersBeforeChangingPinsOrCharges() throws Exception {
    MemoryLedger ledger = ledger();
    Allocator allocator = new Allocator();
    RequestMemoryScope parent = RequestMemoryScope.threadConfined(ledger, OWNER, allocator);
    ByteBuffer buffer = parent.allocate(16);
    var worker = parent.forkWorker();
    var heap = parent.reserveHeap(32);
    var child = parent.childScope();
    child.allocate(8);
    Thread other =
        new Thread(
            () -> {
              expectThrows(IllegalStateException.class, () -> parent.allocate(1));
              expectThrows(IllegalStateException.class, parent::forkWorker);
              expectThrows(IllegalStateException.class, parent::childScope);
              expectThrows(IllegalStateException.class, parent::close);
              parent.retryFailedReclamations(); // no failed handles; no foreign physical close
              expectThrows(IllegalStateException.class, () -> worker.allocate(1));
              expectThrows(IllegalStateException.class, worker::close);
              expectThrows(IllegalStateException.class, heap::close);
              expectThrows(IllegalStateException.class, () -> child.allocate(1));
              expectThrows(IllegalStateException.class, child::close);
              parent.cancel();
            });
    other.start();
    other.join(10_000);
    assertFalse(other.isAlive());
    assertTrue(parent.isCancelled());
    assertTrue(child.isCancelled());
    assertEquals(24L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
    parent.close();
    buffer.putInt(0, 42);
    child.close();
    heap.close();
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP));
    assertEquals(16L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
    worker.close();
    assertTrue(parent.isDrained());
    assertEquals(0, allocator.live.get());
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testAllocationRollbackAndFailedCloseCharge() {
    MemoryLedger ledger = ledger();
    Allocator allocator = new Allocator();
    try (RequestMemoryScope scope = new RequestMemoryScope(ledger, OWNER, allocator)) {
      allocator.failAllocation = true;
      expectThrows(IllegalStateException.class, () -> scope.allocate(16));
      assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
      assertEquals(0L, (long) ledger.snapshot().reserved().get(MemoryLedger.Kind.NATIVE));
      allocator.failAllocation = false;
    }
    RequestMemoryScope scope = new RequestMemoryScope(ledger, OWNER, allocator);
    scope.allocate(16);
    allocator.failClose = true;
    expectThrows(IllegalStateException.class, scope::close);
    assertEquals(16L, ledger.snapshot().failedReclamations());
    assertEquals(1, allocator.live.get());
    allocator.failClose = false;
    scope.retryFailedReclamations();
    assertEquals(0, allocator.live.get());
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testConfinedCloseFailureRequiresOwnerThreadRecovery() throws Exception {
    MemoryLedger ledger = ledger();
    Allocator allocator = new Allocator();
    RequestMemoryScope scope = RequestMemoryScope.threadConfined(ledger, OWNER, allocator);
    scope.allocate(16);
    allocator.failClose = true;
    expectThrows(IllegalStateException.class, scope::close);
    assertFalse(scope.isDrained());
    assertEquals(16L, ledger.snapshot().failedReclamations());
    allocator.failClose = false;
    Thread other =
        new Thread(() -> expectThrows(IllegalStateException.class, scope::retryFailedReclamations));
    other.start();
    other.join(10_000);
    assertFalse(other.isAlive());
    assertEquals(16L, ledger.snapshot().failedReclamations());
    assertEquals(1, allocator.live.get());
    scope.retryFailedReclamations();
    assertTrue(scope.isDrained());
    assertEquals(0, allocator.live.get());
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testLivenessProbeFailurePreservesAllRecoveryHandles() {
    MemoryLedger ledger = ledger();
    Allocator allocator = new Allocator();
    RequestMemoryScope scope = new RequestMemoryScope(ledger, OWNER, allocator);
    scope.allocate(16);
    scope.allocate(32);
    allocator.failClose = true;
    expectThrows(IllegalStateException.class, scope::close);
    assertEquals(48L, ledger.snapshot().failedReclamations());
    allocator.failClose = false;
    allocator.failLivenessProbe = true;
    IllegalStateException failure =
        expectThrows(IllegalStateException.class, scope::retryFailedReclamations);
    assertEquals(1, failure.getSuppressed().length);
    assertEquals(2, allocator.live.get());
    assertEquals(48L, ledger.snapshot().failedReclamations());
    allocator.failLivenessProbe = false;
    scope.retryFailedReclamations();
    assertEquals(0, allocator.live.get());
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
    scope.retryFailedReclamations();
    assertEquals(0, allocator.live.get());
  }

  public void testCancellationDuringAllocationDoesNotPublish() throws Exception {
    MemoryLedger ledger = ledger();
    CountDownLatch entered = new CountDownLatch(1);
    CountDownLatch complete = new CountDownLatch(1);
    AtomicInteger rejected = new AtomicInteger();
    Allocator allocator =
        new Allocator() {
          @Override
          public Allocation allocate(long bytes) {
            Allocation allocation = super.allocate(bytes);
            entered.countDown();
            try {
              assertTrue(complete.await(30, TimeUnit.SECONDS));
            } catch (InterruptedException e) {
              throw new AssertionError(e);
            }
            return allocation;
          }
        };
    RequestMemoryScope scope = new RequestMemoryScope(ledger, OWNER, allocator);
    Thread builder =
        new Thread(
            () -> {
              try {
                scope.allocate(64);
              } catch (RequestMemoryScope.AdmissionException failure) {
                assertEquals(RequestMemoryScope.Failure.CANCELLED, failure.reason());
                rejected.incrementAndGet();
              }
            });
    builder.start();
    assertTrue(entered.await(30, TimeUnit.SECONDS));
    assertEquals(64L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
    scope.cancel();
    scope.close();
    assertEquals(1, allocator.live.get());
    complete.countDown();
    builder.join(30000);
    assertFalse(builder.isAlive());
    assertEquals(1, rejected.get());
    assertEquals(0, allocator.live.get());
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testLeaseRetirementAndDuplicateClose() {
    MemoryLedger ledger = ledger();
    AtomicInteger destruction = new AtomicInteger();
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 64, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      SharedValueLease.Entry<String> entry =
          new SharedValueLease.Entry<>(
              "immutable",
              destruction::incrementAndGet,
              reservation.commit(64, MemoryLedger.State.PUBLISHED));
      SharedValueLease<String> lease = entry.tryAcquire();
      entry.retire();
      entry.retire();
      assertNull(entry.tryAcquire());
      assertEquals("immutable", lease.value());
      assertEquals(1L, entry.activeLeases());
      assertEquals(
          64L,
          (long)
              ledger
                  .snapshot()
                  .lifecycle()
                  .get(MemoryLedger.Kind.NATIVE)
                  .get(MemoryLedger.State.RETIRED_PINNED));
      lease.close();
      lease.close();
      assertEquals(1, destruction.get());
      assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
      expectThrows(IllegalStateException.class, lease::value);
    }
  }

  public void testFailedLeaseReclamationRequiresExplicitRetry() {
    MemoryLedger ledger = ledger();
    AtomicInteger attempts = new AtomicInteger();
    AtomicInteger physicalReleases = new AtomicInteger();
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 64, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      SharedValueLease.Entry<String> entry =
          new SharedValueLease.Entry<>(
              "value",
              () -> {
                if (attempts.incrementAndGet() == 1)
                  throw new IllegalStateException("injected close failure");
                physicalReleases.incrementAndGet();
              },
              reservation.commit(64, MemoryLedger.State.PUBLISHED));
      expectThrows(IllegalStateException.class, entry::retryFailedReclamation);
      SharedValueLease<String> lease = entry.tryAcquire();
      entry.retire();
      expectThrows(IllegalStateException.class, entry::retryFailedReclamation);
      expectThrows(IllegalStateException.class, lease::close);
      assertEquals(64L, ledger.snapshot().failedReclamations());
      assertEquals(0, physicalReleases.get());
      assertNull(entry.tryAcquire());
      lease.close();
      assertEquals(1, attempts.get());
      entry.retryFailedReclamation();
      entry.retryFailedReclamation();
      assertEquals(2, attempts.get());
      assertEquals(1, physicalReleases.get());
      assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
    }
  }

  public void testConcurrentLeaseReclamationRetriesDestroyOnlyOnce() throws Exception {
    MemoryLedger ledger = ledger();
    AtomicInteger attempts = new AtomicInteger();
    AtomicInteger physicalReleases = new AtomicInteger();
    CountDownLatch entered = new CountDownLatch(1);
    CountDownLatch complete = new CountDownLatch(1);
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 64, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      SharedValueLease.Entry<String> entry =
          new SharedValueLease.Entry<>(
              "value",
              () -> {
                if (attempts.incrementAndGet() == 1)
                  throw new IllegalStateException("injected close failure");
                entered.countDown();
                try {
                  assertTrue(complete.await(30, TimeUnit.SECONDS));
                } catch (InterruptedException e) {
                  throw new AssertionError(e);
                }
                physicalReleases.incrementAndGet();
              },
              reservation.commit(64, MemoryLedger.State.PUBLISHED));
      expectThrows(IllegalStateException.class, entry::retire);
      Thread retry = new Thread(entry::retryFailedReclamation);
      retry.start();
      try {
        assertTrue(entered.await(30, TimeUnit.SECONDS));
        entry.retryFailedReclamation();
        assertEquals(2, attempts.get());
        assertEquals(0, physicalReleases.get());
        assertEquals(64L, ledger.snapshot().failedReclamations());
        assertNull(entry.tryAcquire());
      } finally {
        complete.countDown();
      }
      retry.join(30000);
      assertFalse(retry.isAlive());
      assertEquals(2, attempts.get());
      assertEquals(1, physicalReleases.get());
      assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
      entry.retryFailedReclamation();
      assertEquals(2, attempts.get());
    }
  }

  public void testFailedOwnerRemainsRecoverableWithoutCacheReference() {
    MemoryLedger ledger = ledger();
    AtomicInteger attempts = new AtomicInteger();
    createFailedOwner(ledger, attempts);
    assertEquals(1, attempts.get());
    assertEquals(0, ManagedMemory.failedReclamations(ledger, 0).size());
    assertEquals(1, ManagedMemory.failedReclamations(ledger, 1).size());
    assertEquals(64L, ledger.snapshot().failedReclamations());
    ManagedMemory.retryFailedReclamations(ledger);
    assertEquals(2, attempts.get());
    assertEquals(0, ManagedMemory.failedReclamations(ledger, 1).size());
    assertEquals(0L, ledger.snapshot().failedReclamations());
    ManagedMemory.retryFailedReclamations(ledger);
    assertEquals(2, attempts.get());
  }

  private void createFailedOwner(MemoryLedger ledger, AtomicInteger attempts) {
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 64, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      SharedValueLease.Entry<String> entry =
          new SharedValueLease.Entry<>(
              "value",
              () -> {
                if (attempts.incrementAndGet() == 1)
                  throw new IllegalStateException("injected owner close failure");
              },
              reservation.commit(64, MemoryLedger.State.PUBLISHED));
      expectThrows(IllegalStateException.class, entry::retire);
    }
  }

  public void testAcquireRetireRace() throws Exception {
    for (int iteration = 0; iteration < 50; iteration++) {
      MemoryLedger ledger = ledger();
      AtomicInteger destruction = new AtomicInteger();
      try (MemoryLedger.Reservation reservation =
          ledger
              .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 8, MemoryLedger.Purpose.CACHE_VALUE)
              .reservation()) {
        SharedValueLease.Entry<String> entry =
            new SharedValueLease.Entry<>(
                "value",
                destruction::incrementAndGet,
                reservation.commit(8, MemoryLedger.State.PUBLISHED));
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch acquired = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        Thread reader =
            new Thread(
                () -> {
                  try {
                    assertTrue(start.await(30, TimeUnit.SECONDS));
                    SharedValueLease<String> lease = entry.tryAcquire();
                    acquired.countDown();
                    assertTrue(finish.await(30, TimeUnit.SECONDS));
                    if (lease != null) {
                      assertEquals(0, destruction.get());
                      assertEquals("value", lease.value());
                      lease.close();
                    }
                  } catch (InterruptedException e) {
                    throw new AssertionError(e);
                  }
                });
        reader.start();
        start.countDown();
        entry.retire();
        assertTrue(acquired.await(30, TimeUnit.SECONDS));
        assertNull(entry.tryAcquire());
        finish.countDown();
        reader.join(30000);
        assertFalse(reader.isAlive());
        assertEquals(1, destruction.get());
        assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
      }
    }
  }

  public void testImmutableBitsAgainstHeapBaseline() {
    for (int iteration = 0; iteration < 30; iteration++) {
      int length = random().nextInt(1024);
      BitSet left = new BitSet(length);
      BitSet right = new BitSet(length);
      for (int bit = 0; bit < length; bit++) {
        if (random().nextBoolean()) left.set(bit);
        if (random().nextBoolean()) right.set(bit);
      }
      long[] leftWords = new long[(length + 63) >>> 6];
      long[] rightWords = new long[(length + 63) >>> 6];
      long[] leftSource = left.toLongArray();
      long[] rightSource = right.toLongArray();
      System.arraycopy(leftSource, 0, leftWords, 0, leftSource.length);
      System.arraycopy(rightSource, 0, rightWords, 0, rightSource.length);
      if ((length & 63) != 0) { // poisoned tail must never become a document
        leftWords[leftWords.length - 1] |= -1L << (length & 63);
        rightWords[rightWords.length - 1] |= -1L << (length & 63);
      }
      try (RequestMemoryScope scope = new RequestMemoryScope(ledger(), OWNER, new Allocator())) {
        ImmutableBitView a = ImmutableBitView.copyOf(leftWords, length, scope);
        ImmutableBitView b = ImmutableBitView.copyOf(rightWords, length, scope);
        assertEquals(left.cardinality(), a.cardinality());
        for (int bit = 0; bit < length; bit++) assertEquals(left.get(bit), a.contains(bit));
        assertFalse(a.contains(length));
        int actual = a.nextSetBit(0);
        for (int expected = left.nextSetBit(0);
            expected >= 0;
            expected = left.nextSetBit(expected + 1)) {
          assertEquals(expected, actual);
          actual = a.nextSetBit(actual + 1);
        }
        assertEquals(-1, actual);
        BitSet intersection = (BitSet) left.clone();
        intersection.and(right);
        BitSet union = (BitSet) left.clone();
        union.or(right);
        BitSet difference = (BitSet) left.clone();
        difference.andNot(right);
        assertEquals(intersection.cardinality(), a.intersectionCardinality(b));
        assertEquals(union.cardinality(), a.unionCardinality(b));
        assertEquals(difference.cardinality(), a.differenceCardinality(b));
      }
    }
  }

  public void testChildScopeSharesQuotaCancellationAndDrain() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(32, 32, 0), 32);
    Allocator allocator = new Allocator();
    RequestMemoryScope parent = new RequestMemoryScope(ledger, OWNER, allocator);
    AtomicInteger drained = new AtomicInteger();
    parent.onDrained(drained::incrementAndGet);
    RequestMemoryScope child = parent.childScope();
    parent.allocate(16);
    child.allocate(16);
    expectThrows(RequestMemoryScope.AdmissionException.class, () -> child.allocate(1));
    parent.close();
    assertEquals(0, drained.get());
    assertEquals(2, allocator.live.get());
    child.close();
    assertEquals(1, drained.get());
    assertTrue(parent.isDrained());
    assertEquals(0, allocator.live.get());
    RequestMemoryScope next = new RequestMemoryScope(ledger, OWNER, allocator);
    RequestMemoryScope nested = next.childScope();
    next.cancel();
    assertEquals(
        RequestMemoryScope.Failure.CANCELLED,
        expectThrows(RequestMemoryScope.AdmissionException.class, () -> nested.allocate(1))
            .reason());
    nested.close();
    next.close();
  }

  public void testFailedCloseDelaysDrainCallbackUntilRecovery() {
    MemoryLedger ledger = ledger();
    Allocator allocator = new Allocator();
    RequestMemoryScope scope = new RequestMemoryScope(ledger, OWNER, allocator);
    AtomicInteger drained = new AtomicInteger();
    scope.onDrained(drained::incrementAndGet);
    scope.allocate(16);
    allocator.failClose = true;
    expectThrows(IllegalStateException.class, scope::close);
    assertFalse(scope.isDrained());
    assertEquals(0, drained.get());
    allocator.failClose = false;
    scope.retryFailedReclamations();
    assertTrue(scope.isDrained());
    assertEquals(1, drained.get());
    scope.onDrained(drained::incrementAndGet);
    assertEquals(2, drained.get());
  }

  public void testHeapScratchSharesCombinedQuotaAndPinsScope() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(32, 32, 0), 32);
    RequestMemoryScope scope = new RequestMemoryScope(ledger, OWNER, new Allocator());
    scope.allocate(16);
    RequestMemoryScope.HeapReservation scratch = scope.reserveHeap(16);
    assertTrue(scratch.enrolled());
    expectThrows(RequestMemoryScope.AdmissionException.class, () -> scope.reserveHeap(1));
    scope.close();
    assertFalse(scope.isDrained());
    scratch.close();
    assertTrue(scope.isDrained());
    assertEquals(0, ledger.snapshot().reserved().get(MemoryLedger.Kind.HEAP).longValue());
  }

  public void testHostAdmissionCancellationCannotAllocateOrLeakCredits() throws Exception {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(0, 100, 0), 100);
    var allocator = new Allocator();
    var other = new MemoryLedger.Owner("other", "other", 0, "other");
    try (var occupied =
            ledger
                .tryReserve(other, MemoryLedger.Kind.NATIVE, 100, MemoryLedger.Purpose.CACHE_VALUE)
                .reservation();
        var scope =
            new RequestMemoryScope(
                ledger,
                OWNER,
                allocator,
                (owner, kind, bytes, purpose, cancelled) -> {
                  try {
                    return ledger.awaitReserve(
                        owner, kind, bytes, purpose, java.time.Duration.ofSeconds(5), cancelled);
                  } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(error);
                  }
                })) {
      assertEquals(100, occupied.remaining());
      var result = new java.util.concurrent.atomic.AtomicReference<Throwable>();
      Thread worker =
          new Thread(
              () -> {
                try {
                  scope.allocate(1);
                } catch (Throwable error) {
                  result.set(error);
                }
              });
      worker.start();
      long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
      while (ledger.telemetry().queuedRequests() == 0 && System.nanoTime() < until)
        Thread.onSpinWait();
      try {
        assertEquals(1, ledger.telemetry().queuedRequests());
      } finally {
        scope.cancel();
        worker.join(5000);
      }
      assertFalse(worker.isAlive());
      assertEquals(
          RequestMemoryScope.Failure.CANCELLED,
          ((RequestMemoryScope.AdmissionException) result.get()).reason());
      assertEquals(0, allocator.live.get());
      assertEquals(0, ledger.telemetry().queuedRequests());
      assertEquals(100, ledger.snapshot().reserved().get(MemoryLedger.Kind.NATIVE).longValue());
    }
    assertTrue(ledger.awaitDrained(java.time.Duration.ZERO));
  }

  public void testHostAdmissionPolicyAndRequestCapFollowChildWorkers() {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(0, 256, 0), 96);
    var allocator = new Allocator();
    var attempts = new AtomicInteger();
    try (var parent =
            new RequestMemoryScope(
                ledger,
                OWNER,
                allocator,
                (owner, kind, bytes, purpose, cancelled) -> {
                  attempts.incrementAndGet();
                  return ledger.tryReserve(owner, kind, bytes, purpose);
                });
        var child = parent.childScope()) {
      assertEquals(64, parent.allocate(64).remaining());
      assertEquals(
          RequestMemoryScope.Failure.REQUEST_LIMIT,
          expectThrows(RequestMemoryScope.AdmissionException.class, () -> child.allocate(64))
              .reason());
      assertEquals(2, attempts.get());
      assertEquals(1, allocator.live.get());
    }
    assertEquals(0, allocator.live.get());
    assertEquals(0L, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
  }
}
