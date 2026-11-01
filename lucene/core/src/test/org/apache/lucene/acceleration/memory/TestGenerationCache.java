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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestGenerationCache extends LuceneTestCase {
  private static SharedValueLease.Entry<String> value(MemoryLedger ledger, AtomicInteger freed) {
    MemoryLedger.Owner owner = new MemoryLedger.Owner("shared", "core", 1, "build");
    try (MemoryLedger.Reservation reservation =
        ledger
            .tryReserve(owner, MemoryLedger.Kind.HEAP, 64, MemoryLedger.Purpose.ORDINAL_BUILD)
            .reservation()) {
      return new SharedValueLease.Entry<>(
          "value", freed::incrementAndGet, reservation.commit(64, MemoryLedger.State.PUBLISHED));
    }
  }

  public void testInvalidatedBuildCannotPublishAndRetriesNewGeneration() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(128, 0, 0), 128);
    GenerationCache<String, String> cache = new GenerationCache<>(1);
    CountDownLatch entered = new CountDownLatch(1), finish = new CountDownLatch(1);
    AtomicReference<Throwable> failure = new AtomicReference<>();
    AtomicInteger freed = new AtomicInteger();
    Thread leader =
        new Thread(
            () -> {
              try (SharedValueLease<String> lease =
                  cache.acquire(
                      "field",
                      1,
                      cancelled -> {
                        entered.countDown();
                        try {
                          if (!finish.await(10, TimeUnit.SECONDS))
                            throw new IOException("timed out");
                        } catch (InterruptedException interrupted) {
                          Thread.currentThread().interrupt();
                          throw new IOException(interrupted);
                        }
                        return value(ledger, freed);
                      },
                      () -> false)) {
                fail("invalidated value was published: " + lease.value());
              } catch (Throwable expected) {
                failure.set(expected);
              }
            });
    leader.start();
    assertTrue(entered.await(10, TimeUnit.SECONDS));
    cache.advanceGeneration(2);
    finish.countDown();
    leader.join(10000);
    assertFalse(leader.isAlive());
    assertTrue(failure.get() instanceof IOException);
    assertEquals(1, freed.get());
    assertEquals(1, cache.statistics().rejectedPublications());
    try (SharedValueLease<String> lease =
        cache.acquire("field", 2, cancelled -> value(ledger, freed), () -> false)) {
      cache.close();
      assertEquals("value", lease.value());
      assertEquals(64, ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
    }
    assertEquals(2, freed.get());
  }

  public void testCancelledWaiterDoesNotCancelAnotherWaiter() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(128, 0, 0), 128);
    GenerationCache<String, String> cache = new GenerationCache<>(1);
    CountDownLatch entered = new CountDownLatch(1), finish = new CountDownLatch(1);
    AtomicBoolean cancel = new AtomicBoolean();
    AtomicInteger builds = new AtomicInteger(), freed = new AtomicInteger();
    AtomicReference<Throwable> unexpected = new AtomicReference<>(),
        cancelled = new AtomicReference<>();
    Thread leader =
        new Thread(
            () -> {
              try (SharedValueLease<String> lease =
                  cache.acquire(
                      "field",
                      1,
                      stop -> {
                        builds.incrementAndGet();
                        entered.countDown();
                        try {
                          if (!finish.await(10, TimeUnit.SECONDS))
                            throw new IOException("timed out");
                        } catch (InterruptedException interrupted) {
                          Thread.currentThread().interrupt();
                          throw new IOException(interrupted);
                        }
                        if (stop.getAsBoolean())
                          throw new IOException("unexpected builder cancellation");
                        return value(ledger, freed);
                      },
                      () -> false)) {
                assertEquals("value", lease.value());
              } catch (Throwable failure) {
                unexpected.set(failure);
              }
            });
    leader.start();
    assertTrue(entered.await(10, TimeUnit.SECONDS));
    Thread waiter =
        new Thread(
            () -> {
              try (SharedValueLease<String> lease =
                  cache.acquire(
                      "field",
                      1,
                      stop -> {
                        builds.incrementAndGet();
                        return value(ledger, freed);
                      },
                      cancel::get)) {
                fail("cancelled waiter acquired " + lease.value());
              } catch (Throwable expected) {
                cancelled.set(expected);
              }
            });
    waiter.start();
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
    while (cache.statistics().sharedWaits() == 0 && System.nanoTime() < deadline)
      Thread.onSpinWait();
    assertEquals(1, cache.statistics().sharedWaits());
    cancel.set(true);
    waiter.join(10000);
    assertFalse(waiter.isAlive());
    assertTrue(cancelled.get() instanceof IOException);
    finish.countDown();
    leader.join(10000);
    assertFalse(leader.isAlive());
    assertNull(unexpected.get());
    assertEquals(1, builds.get());
    cache.close();
    assertEquals(1, freed.get());
  }
}
