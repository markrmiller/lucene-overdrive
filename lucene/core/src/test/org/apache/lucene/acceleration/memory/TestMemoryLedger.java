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

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.NamedThreadFactory;

public class TestMemoryLedger extends LuceneTestCase {
  private static final MemoryLedger.Owner OWNER =
      new MemoryLedger.Owner("query", "core", 1, "request");

  public void testOrdinalPartitionIncludesBuildAndRetainedWithoutDoubleCharge() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(1024, 0, 0), 1024);
    ledger.setOrdinalLimit(128);
    MemoryLedger.Charge retained;
    try (MemoryLedger.Reservation build =
            ledger
                .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 80, MemoryLedger.Purpose.ORDINAL_BUILD)
                .reservation();
        MemoryLedger.Reservation second =
            ledger
                .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 48, MemoryLedger.Purpose.ORDINAL_BUILD)
                .reservation();
        MemoryLedger.Reservation foreground =
            ledger
                .tryReserve(
                    OWNER, MemoryLedger.Kind.HEAP, 300, MemoryLedger.Purpose.REQUEST_SCRATCH)
                .reservation()) {
      assertNotNull(second);
      assertNotNull(foreground);
      retained = build.commit(64, MemoryLedger.State.PUBLISHED);
      assertEquals(128L, ledger.ordinalBytes());
      assertEquals(64L, ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
      assertEquals(364L, ledger.snapshot().reserved().get(MemoryLedger.Kind.HEAP).longValue());
      assertFalse(build.tryGrow(1));
      assertEquals(
          MemoryLedger.Reason.NODE_CAPACITY,
          ledger
              .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.ORDINAL_BUILD)
              .reason());
    }
    assertEquals(64L, ledger.ordinalBytes());
    ledger.setOrdinalLimit(32);
    retained.reclassify(MemoryLedger.State.RETIRED_PINNED);
    assertEquals(32L, ledger.telemetry().ordinalTargetBytes());
    assertEquals(64L, ledger.telemetry().ordinalUsedBytes());
    assertEquals(
        MemoryLedger.Reason.NODE_CAPACITY,
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.ORDINAL_BUILD)
            .reason());
    retained.releaseAfterDeallocation();
    assertEquals(0L, ledger.ordinalBytes());
    ledger.setOrdinalLimit(0);
    assertEquals(
        MemoryLedger.Reason.NODE_CAPACITY,
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.ORDINAL_BUILD)
            .reason());
  }

  public void testReservationCommitAndShrink() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(0, 128, 0), 1024);
    MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 64, MemoryLedger.Purpose.REQUEST_SCRATCH)
            .reservation();
    MemoryLedger.Charge charge = reservation.commit(64, MemoryLedger.State.ACTIVE);
    assertEquals(64L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
    assertEquals(0L, (long) ledger.snapshot().reserved().get(MemoryLedger.Kind.NATIVE));
    assertFalse(reservation.tryGrow(128)); // source + destination must fit simultaneously
    ledger.setTargetLimits(new MemoryLedger.Limits(0, 32, 0));
    assertEquals(128L, ledger.snapshot().installed().nativeBytes());
    assertEquals(32L, ledger.snapshot().target().nativeBytes());
    assertFalse(reservation.tryGrow(1));
    charge.releaseAfterDeallocation();
    charge.releaseAfterDeallocation();
    charge.reclassify(MemoryLedger.State.RETIRED_PINNED); // racing retirement cannot resurrect it
    assertEquals(MemoryLedger.State.RELEASED, charge.snapshot().state());
    assertEquals(32L, ledger.snapshot().installed().nativeBytes());
    reservation.close();
    assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testCombinedRequestAndCoreLimits() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(100, 100, 100), 30);
    try (MemoryLedger.Reservation heap =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 20, MemoryLedger.Purpose.REQUEST_SCRATCH)
            .reservation()) {
      assertNotNull(heap);
      MemoryLedger.Owner worker = new MemoryLedger.Owner("worker", "core", 2, "request");
      assertEquals(
          MemoryLedger.Reason.REQUEST_LIMIT,
          ledger
              .tryReserve(
                  worker, MemoryLedger.Kind.NATIVE, 11, MemoryLedger.Purpose.REQUEST_SCRATCH)
              .reason());
      try (MemoryLedger.Reservation nativeReservation =
          ledger
              .tryReserve(
                  worker, MemoryLedger.Kind.NATIVE, 10, MemoryLedger.Purpose.REQUEST_SCRATCH)
              .reservation()) {
        assertNotNull(nativeReservation);
      }
      ledger.setCoreLimit("core", 20);
      MemoryLedger.Owner other = new MemoryLedger.Owner("other", "core", 3, "anotherRequest");
      assertEquals(
          MemoryLedger.Reason.CORE_LIMIT,
          ledger
              .tryReserve(other, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.CACHE_VALUE)
              .reason());
    }
    ledger.shutdown();
    assertEquals(
        MemoryLedger.Reason.SHUTDOWN,
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.CACHE_VALUE)
            .reason());
    expectThrows(ArithmeticException.class, () -> MemoryLedger.bytes(Long.MAX_VALUE, 8));
    expectThrows(IllegalArgumentException.class, () -> MemoryLedger.bytes(-1, 8));
  }

  public void testExactLimitConcurrentAdmissions() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(0, 64, 0), 1024);
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch admitted = new CountDownLatch(2);
    CountDownLatch release = new CountDownLatch(1);
    AtomicInteger winners = new AtomicInteger();
    Thread[] threads = new Thread[2];
    for (int i = 0; i < threads.length; i++) {
      threads[i] =
          new Thread(
              () -> {
                try {
                  assertTrue(start.await(30, TimeUnit.SECONDS));
                  MemoryLedger.Admission admission =
                      ledger.tryReserve(
                          OWNER,
                          MemoryLedger.Kind.NATIVE,
                          64,
                          MemoryLedger.Purpose.REQUEST_SCRATCH);
                  if (admission.granted()) winners.incrementAndGet();
                  admitted.countDown();
                  assertTrue(release.await(30, TimeUnit.SECONDS));
                  if (admission.granted()) admission.reservation().close();
                } catch (InterruptedException e) {
                  throw new AssertionError(e);
                }
              });
      threads[i].start();
    }
    start.countDown();
    assertTrue(admitted.await(30, TimeUnit.SECONDS));
    assertEquals(1, winners.get());
    assertEquals(64L, (long) ledger.snapshot().reserved().get(MemoryLedger.Kind.NATIVE));
    release.countDown();
    for (Thread thread : threads) {
      thread.join(30000);
      assertFalse(thread.isAlive());
    }
    assertEquals(0L, (long) ledger.snapshot().reserved().get(MemoryLedger.Kind.NATIVE));
  }

  public void testTransferDoesNotRefundAndFailedReclamation() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(0, 64, 0), 64);
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 64, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      MemoryLedger.Charge charge = reservation.commit(64, MemoryLedger.State.BUILDING);
      assertTrue(
          charge.transferOwner(
              new MemoryLedger.Owner("cache", "core", 1, "cache"), MemoryLedger.State.PUBLISHED));
      assertEquals(64L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
      charge.reclassify(MemoryLedger.State.FAILED_RECLAMATION);
      charge.reclassify(
          MemoryLedger.State.RETIRED_PINNED); // retirement can lag a failed final close
      assertEquals(64L, ledger.snapshot().failedReclamations());
      assertEquals(
          MemoryLedger.Reason.NODE_CAPACITY,
          ledger
              .tryReserve(OWNER, MemoryLedger.Kind.NATIVE, 1, MemoryLedger.Purpose.REQUEST_SCRATCH)
              .reason());
      charge.releaseAfterDeallocation();
    }
  }

  public void testWeightedContentionBorrowsIdleCapacityAndShutdownDrains() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    MemoryLedger.Owner first = new MemoryLedger.Owner("first", "first", 0, "first");
    MemoryLedger.Owner second = new MemoryLedger.Owner("second", "second", 0, "second");
    ledger.setCoreWeight("first", 1);
    ledger.setCoreWeight("second", 1);
    MemoryLedger.Charge charge;
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(first, MemoryLedger.Kind.HEAP, 80, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      charge = reservation.commit(80, MemoryLedger.State.PUBLISHED);
    }
    java.util.concurrent.atomic.AtomicReference<MemoryLedger.Admission> result =
        new java.util.concurrent.atomic.AtomicReference<>();
    java.util.concurrent.atomic.AtomicReference<Throwable> failure =
        new java.util.concurrent.atomic.AtomicReference<>();
    Thread waiter =
        new Thread(
            () -> {
              try {
                result.set(
                    ledger.awaitReserve(
                        second,
                        MemoryLedger.Kind.HEAP,
                        30,
                        MemoryLedger.Purpose.CACHE_VALUE,
                        java.time.Duration.ofSeconds(10),
                        () -> false));
              } catch (Throwable exception) {
                failure.set(exception);
              }
            });
    waiter.start();
    long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(10);
    while (ledger.telemetry().queuedRequests() == 0 && System.nanoTime() < deadline)
      Thread.onSpinWait();
    assertEquals(1, ledger.telemetry().queuedRequests());
    assertEquals(
        MemoryLedger.Reason.FAIR_SHARE,
        ledger
            .tryReserve(first, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.CACHE_VALUE)
            .reason());
    charge.releaseAfterDeallocation();
    waiter.join(10000);
    assertFalse(waiter.isAlive());
    assertNull(failure.get());
    assertTrue(result.get().granted());
    ledger.shutdown();
    assertFalse(ledger.awaitDrained(java.time.Duration.ZERO));
    result.get().reservation().close();
    assertTrue(ledger.awaitDrained(java.time.Duration.ofSeconds(1)));
    assertTrue(ledger.telemetry().waitNanos() > 0);
  }

  public void testWarmingYieldsAndCancelledQueueCreatesNoReservation() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    MemoryLedger.Owner owner = new MemoryLedger.Owner("warm", "core", 0, "warm");
    assertEquals(
        MemoryLedger.Reason.PRESSURE,
        ledger
            .tryReserve(owner, MemoryLedger.Kind.HEAP, 81, MemoryLedger.Purpose.WARMED_VALUE)
            .reason());
    try (MemoryLedger.Reservation foreground =
        ledger
            .tryReserve(owner, MemoryLedger.Kind.HEAP, 100, MemoryLedger.Purpose.REQUEST_SCRATCH)
            .reservation()) {
      assertEquals(100, foreground.remaining());
      assertEquals(
          MemoryLedger.Reason.CANCELLED,
          ledger
              .awaitReserve(
                  owner,
                  MemoryLedger.Kind.HEAP,
                  1,
                  MemoryLedger.Purpose.CACHE_VALUE,
                  java.time.Duration.ofSeconds(1),
                  () -> true)
              .reason());
      assertEquals(100, ledger.snapshot().reserved().get(MemoryLedger.Kind.HEAP).longValue());
    }
    assertEquals(0, ledger.telemetry().queuedRequests());
  }

  public void testManagedEnrollmentSurvivesShrinkToZeroAndShutdown() {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    MemoryLedger.Owner owner = new MemoryLedger.Owner("shrink", "core", 0, "shrink");
    MemoryLedger.Charge charge;
    try (MemoryLedger.Reservation reservation =
        ManagedMemory.reserve(
            ledger, owner, MemoryLedger.Kind.HEAP, 50, MemoryLedger.Purpose.CACHE_VALUE)) {
      charge = reservation.commit(50, MemoryLedger.State.PUBLISHED);
    }
    ledger.setTargetLimits(new MemoryLedger.Limits(0, 0, 0));
    assertTrue(ledger.isEnrolled(MemoryLedger.Kind.HEAP));
    assertEquals(
        RequestMemoryScope.Failure.NODE_CAPACITY,
        expectThrows(
                RequestMemoryScope.AdmissionException.class,
                () ->
                    ManagedMemory.reserve(
                        ledger, owner, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.CACHE_VALUE))
            .reason());
    charge.releaseAfterDeallocation();
    assertEquals(0, ledger.snapshot().installed().heapBytes());
    expectThrows(
        RequestMemoryScope.AdmissionException.class,
        () ->
            ManagedMemory.reserve(
                ledger, owner, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.CACHE_VALUE));
    ledger.shutdown();
    assertEquals(
        RequestMemoryScope.Failure.SHUTDOWN,
        expectThrows(
                RequestMemoryScope.AdmissionException.class,
                () ->
                    ManagedMemory.reserve(
                        ledger, owner, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.CACHE_VALUE))
            .reason());
    // An unenrolled kind keeps the default unmanaged behavior even after shutdown.
    assertNull(
        ManagedMemory.reserve(
            ledger, owner, MemoryLedger.Kind.MAPPED, 1, MemoryLedger.Purpose.CACHE_VALUE));
  }

  public void testContendedSharesAreIndependentPerKind() throws Exception {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(100, 100, 0), 200);
    var first = new MemoryLedger.Owner("a", "a", 0, "a");
    var second = new MemoryLedger.Owner("b", "b", 0, "b");
    ledger.setCoreWeight("a", 1);
    ledger.setCoreWeight("b", 3);
    var cancel = new java.util.concurrent.atomic.AtomicBoolean();
    try (var nativeCredit =
            ledger
                .tryReserve(
                    first, MemoryLedger.Kind.NATIVE, 80, MemoryLedger.Purpose.REQUEST_SCRATCH)
                .reservation();
        var heapCredit =
            ledger
                .tryReserve(first, MemoryLedger.Kind.HEAP, 30, MemoryLedger.Purpose.REQUEST_SCRATCH)
                .reservation()) {
      assertEquals(80, nativeCredit.remaining());
      assertEquals(30, heapCredit.remaining());
      var result = new java.util.concurrent.atomic.AtomicReference<MemoryLedger.Admission>();
      var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
      Thread waiter =
          new Thread(
              () -> {
                try {
                  result.set(
                      ledger.awaitReserve(
                          second,
                          MemoryLedger.Kind.HEAP,
                          75,
                          MemoryLedger.Purpose.REQUEST_SCRATCH,
                          java.time.Duration.ofSeconds(5),
                          cancel::get));
                } catch (Throwable error) {
                  failure.set(error);
                }
              });
      waiter.start();
      try {
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (ledger.telemetry().queuedRequests() == 0 && System.nanoTime() < until)
          Thread.onSpinWait();
        assertEquals(1, ledger.telemetry().queuedRequests());
        assertEquals(
            MemoryLedger.Reason.FAIR_SHARE,
            ledger
                .tryReserve(first, MemoryLedger.Kind.HEAP, 1, MemoryLedger.Purpose.CACHE_VALUE)
                .reason());
        var unconfigured = new MemoryLedger.Owner("c", "c", 0, "c");
        assertEquals(
            "an unconfigured new core participates with default weight one",
            MemoryLedger.Reason.FAIR_SHARE,
            ledger
                .tryReserve(
                    unconfigured, MemoryLedger.Kind.HEAP, 25, MemoryLedger.Purpose.CACHE_VALUE)
                .reason());
        assertEquals(
            30L,
            ledger.coreUsage(256).get("a").admittedBytes().get(MemoryLedger.Kind.HEAP).longValue());
        assertEquals(1, ledger.coreUsage(256).get("b").queued());
        // Heap demand must not strand this core's independent native capacity.
        try (var extra =
            ledger
                .tryReserve(first, MemoryLedger.Kind.NATIVE, 1, MemoryLedger.Purpose.CACHE_VALUE)
                .reservation()) {
          assertNotNull(extra);
        }
      } finally {
        cancel.set(true);
        waiter.join(5000);
      }
      assertFalse(waiter.isAlive());
      assertNull(failure.get());
      assertEquals(MemoryLedger.Reason.CANCELLED, result.get().reason());
      assertEquals(0, ledger.telemetry().queuedRequests());
    }
    assertTrue(ledger.awaitDrained(java.time.Duration.ZERO));
  }

  public void testGrowthWithHeldCreditsDoesNotQueueOrReclaim() throws Exception {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    var calls = new AtomicInteger();
    try (var held =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 80, MemoryLedger.Purpose.REQUEST_SCRATCH)
            .reservation()) {
      assertEquals(80, held.remaining());
      var other = new MemoryLedger.Owner("other", "other", 0, "other");
      try (var occupied =
          ledger
              .tryReserve(other, MemoryLedger.Kind.HEAP, 20, MemoryLedger.Purpose.CACHE_VALUE)
              .reservation()) {
        assertEquals(20, occupied.remaining());
        assertEquals(
            MemoryLedger.Reason.PRESSURE,
            ledger
                .awaitReserve(
                    OWNER,
                    MemoryLedger.Kind.HEAP,
                    1,
                    MemoryLedger.Purpose.REQUEST_SCRATCH,
                    java.time.Duration.ofSeconds(5),
                    () -> false,
                    calls::incrementAndGet)
                .reason());
        assertEquals(0, calls.get());
        assertEquals(0, ledger.telemetry().queuedRequests());
      }
    }
    assertTrue(ledger.awaitDrained(java.time.Duration.ZERO));
  }

  public void testQueuedDemandPrecedesReclamationOutsideAdmissionLock() throws Exception {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    var optional =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 100, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation();
    var charge = optional.commit(100, MemoryLedger.State.IDLE);
    optional.close();
    var other = new MemoryLedger.Owner("other", "other", 0, "other");
    var observed = new CountDownLatch(1);
    var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
    try {
      var admission =
          ledger.awaitReserve(
              other,
              MemoryLedger.Kind.HEAP,
              30,
              MemoryLedger.Purpose.REQUEST_SCRATCH,
              java.time.Duration.ofSeconds(5),
              () -> false,
              () -> {
                Thread observer =
                    new Thread(
                        () -> {
                          try {
                            assertEquals(1, ledger.telemetry().queuedRequests());
                          } catch (Throwable error) {
                            failure.set(error);
                          } finally {
                            observed.countDown();
                          }
                        });
                observer.start();
                try {
                  assertTrue(
                      "reclamation must not hold the admission lock",
                      observed.await(5, TimeUnit.SECONDS));
                  observer.join(5000);
                  assertFalse(observer.isAlive());
                } catch (InterruptedException error) {
                  Thread.currentThread().interrupt();
                  throw new AssertionError(error);
                }
                charge.releaseAfterDeallocation();
              });
      try (var granted = admission.reservation()) {
        assertTrue(admission.granted());
        assertEquals(30, granted.remaining());
        assertNull(failure.get());
      }
    } finally {
      charge.releaseAfterDeallocation();
    }
    assertEquals(0, ledger.telemetry().queuedRequests());
    assertTrue(ledger.awaitDrained(java.time.Duration.ZERO));
  }

  public void testIndivisibleRequestsBeyondSoftSharesSerializeWithoutDeadlock() throws Exception {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    ledger.setCoreWeight("a", 1);
    ledger.setCoreWeight("b", 3);
    var held =
        ledger
            .tryReserve(OWNER, MemoryLedger.Kind.HEAP, 100, MemoryLedger.Purpose.CACHE_VALUE)
            .reservation();
    var a = new java.util.concurrent.atomic.AtomicReference<MemoryLedger.Admission>();
    var b = new java.util.concurrent.atomic.AtomicReference<MemoryLedger.Admission>();
    var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
    Thread first =
        new Thread(
            () -> {
              try {
                a.set(
                    ledger.awaitReserve(
                        new MemoryLedger.Owner("a", "a", 0, "a"),
                        MemoryLedger.Kind.HEAP,
                        80,
                        MemoryLedger.Purpose.REQUEST_SCRATCH,
                        java.time.Duration.ofSeconds(5),
                        () -> false));
              } catch (Throwable error) {
                failure.set(error);
              }
            });
    Thread second =
        new Thread(
            () -> {
              try {
                b.set(
                    ledger.awaitReserve(
                        new MemoryLedger.Owner("b", "b", 0, "b"),
                        MemoryLedger.Kind.HEAP,
                        80,
                        MemoryLedger.Purpose.REQUEST_SCRATCH,
                        java.time.Duration.ofSeconds(5),
                        () -> false));
              } catch (Throwable error) {
                failure.set(error);
              }
            });
    try {
      first.start();
      long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
      while (ledger.telemetry().queuedRequests() < 1 && System.nanoTime() < until)
        Thread.onSpinWait();
      assertEquals(1, ledger.telemetry().queuedRequests());
      second.start();
      while (ledger.telemetry().queuedRequests() < 2 && System.nanoTime() < until)
        Thread.onSpinWait();
      assertEquals(2, ledger.telemetry().queuedRequests());
      held.close();
      first.join(5000);
      assertFalse(first.isAlive());
      assertNull(failure.get());
      assertTrue(a.get().granted());
      assertNull("the second peak cannot fit concurrently", b.get());
      a.get().reservation().close();
      second.join(5000);
      assertFalse(second.isAlive());
      assertNull(failure.get());
      assertTrue(b.get().granted());
    } finally {
      held.close();
      ledger.shutdown();
      first.join(5000);
      second.join(5000);
      if (a.get() != null && a.get().granted()) a.get().reservation().close();
      if (b.get() != null && b.get().granted()) b.get().reservation().close();
    }
    assertEquals(0, ledger.telemetry().queuedRequests());
    assertTrue(ledger.awaitDrained(java.time.Duration.ZERO));
  }

  public void testUnenrolledReserveAndShutdownProbeTakeNoLedgerLock() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    MemoryLedger.Owner owner = new MemoryLedger.Owner("stock", "core", 0, "stock");
    ExecutorService reader = Executors.newSingleThreadExecutor(new NamedThreadFactory("stock"));
    try {
      synchronized (ledger.lock) {
        assertNull(
            reader
                .submit(
                    () ->
                        ManagedMemory.reserve(
                            ledger,
                            owner,
                            MemoryLedger.Kind.MAPPED,
                            1,
                            MemoryLedger.Purpose.CACHE_VALUE))
                .get(10, TimeUnit.SECONDS));
        assertFalse(reader.submit(ledger::isShutdown).get(10, TimeUnit.SECONDS));
      }
      ledger.shutdown();
      assertTrue(reader.submit(ledger::isShutdown).get(10, TimeUnit.SECONDS));
    } finally {
      reader.shutdownNow();
      assertTrue(reader.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  public void testIsEnrolledReadsWithoutLedgerLockAndObservesTargetLimits() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(100, 0, 0), 100);
    ExecutorService reader = Executors.newSingleThreadExecutor(new NamedThreadFactory("enrolled"));
    try {
      synchronized (ledger.lock) {
        assertTrue(
            reader
                .submit(() -> ledger.isEnrolled(MemoryLedger.Kind.HEAP))
                .get(10, TimeUnit.SECONDS));
        assertFalse(
            reader
                .submit(() -> ledger.isEnrolled(MemoryLedger.Kind.NATIVE))
                .get(10, TimeUnit.SECONDS));
      }
      ledger.setTargetLimits(new MemoryLedger.Limits(100, 64, 0));
      assertTrue(reader.submit(() -> ledger.isEnrolled(MemoryLedger.Kind.NATIVE)).get());
      assertFalse(reader.submit(() -> ledger.isEnrolled(MemoryLedger.Kind.MAPPED)).get());
      ledger.setTargetLimits(new MemoryLedger.Limits(0, 0, 0));
      assertTrue(reader.submit(() -> ledger.isEnrolled(MemoryLedger.Kind.HEAP)).get());
      assertTrue(reader.submit(() -> ledger.isEnrolled(MemoryLedger.Kind.NATIVE)).get());
      assertFalse(reader.submit(() -> ledger.isEnrolled(MemoryLedger.Kind.MAPPED)).get());
    } finally {
      reader.shutdownNow();
      assertTrue(reader.awaitTermination(10, TimeUnit.SECONDS));
    }
  }
}
