/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
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

import java.io.InterruptedIOException;
import java.lang.reflect.Modifier;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Deadline translation must not turn cancellation, interruption or shutdown into a timeout. */
@SuppressWarnings("try")
public class TestTaskDeadlineMarker extends LuceneTestCase {
  private static final String MARKER =
      AccelerationRuntime.class.getName() + "$TaskDeadlineExceededException";

  private static AccelerationRuntime runtime() throws Exception {
    return new AccelerationRuntime(
        new AccelerationConfig(
            AccelerationConfig.Mode.OFF,
            null,
            Set.of(),
            Set.of(),
            0,
            0,
            1,
            false,
            AccelerationConfig.StoragePolicy.CURRENT));
  }

  private static AccelerationRuntime.TaskContext task(String id, long deadline) {
    return new AccelerationRuntime.TaskContext(
        new MemoryLedger.Owner("test", "deadline", 0, id), deadline);
  }

  private static long expired() {
    return System.nanoTime() - TimeUnit.SECONDS.toNanos(1);
  }

  public void testExpiredBoundTaskHasDistinctDeadlineTypeAndRestoresCaller() throws Exception {
    try (var runtime = runtime()) {
      var caller = task("caller", 0);
      try (var outer = runtime.enterTask(caller)) {
        var expired = task("expired", expired());
        try (var binding = runtime.enterTask(expired)) {
          var failure = expectThrows(InterruptedIOException.class, runtime::checkpoint);
          assertEquals(
              "only actual task deadline expiry receives the typed marker",
              MARKER,
              failure.getClass().getName());
          assertSame(expired, runtime.currentTaskContext());
          assertFalse(Thread.currentThread().isInterrupted());
        }
        assertSame(caller, runtime.currentTaskContext());
        runtime.checkpoint();
      }
      assertNull(runtime.currentTaskContext());
      assertNull(runtime.currentOwner());
      assertEquals(0, runtime.inventory().execution().activeScopes());
    }
  }

  public void testInterruptionWinsOverAnExpiredDeadlineAndKeepsItsFlag() throws Exception {
    try (var runtime = runtime();
        var binding = runtime.enterTask(task("interrupt", expired()))) {
      assertFalse(Thread.currentThread().isInterrupted());
      try {
        Thread.currentThread().interrupt();
        var failure = expectThrows(InterruptedIOException.class, runtime::checkpoint);
        assertEquals(InterruptedIOException.class, failure.getClass());
        assertTrue(Thread.currentThread().isInterrupted());
      } finally {
        Thread.interrupted();
      }
    }
  }

  public void testExplicitCancellationWinsOverAnExpiredDeadline() throws Exception {
    try (var runtime = runtime()) {
      var context = task("cancelled", expired());
      context.cancel();
      try (var binding = runtime.enterTask(context)) {
        var failure = expectThrows(InterruptedIOException.class, runtime::checkpoint);
        assertEquals(InterruptedIOException.class, failure.getClass());
        assertSame(context, runtime.currentTaskContext());
      }
      assertNull(runtime.currentTaskContext());
    }
  }

  public void testRuntimeShutdownWinsOverAnExpiredDeadline() throws Exception {
    try (var runtime = runtime()) {
      runtime.shutdown(System.nanoTime());
      try (var binding = runtime.enterTask(task("shutdown", expired()))) {
        var failure = expectThrows(InterruptedIOException.class, runtime::checkpoint);
        assertEquals(InterruptedIOException.class, failure.getClass());
      }
      assertNull(runtime.currentTaskContext());
    }
  }

  public void testUnboundZeroAndFutureDeadlineCheckpointsKeepNormalBehavior() throws Exception {
    try (var runtime = runtime()) {
      runtime.checkpoint();
      for (long deadline : new long[] {0, System.nanoTime() + TimeUnit.DAYS.toNanos(1)}) {
        var context = task("not-expired", deadline);
        try (var binding = runtime.enterTask(context)) {
          runtime.checkpoint();
          assertSame(context, runtime.currentTaskContext());
        }
        assertNull(runtime.currentTaskContext());
      }
    }
  }

  public void testMarkerIsFinalAndNotPubliclyConstructible() throws Exception {
    Class<?> marker = Class.forName(MARKER);
    assertTrue(Modifier.isPublic(marker.getModifiers()));
    assertTrue(Modifier.isFinal(marker.getModifiers()));
    assertTrue(InterruptedIOException.class.isAssignableFrom(marker));
    assertEquals(0, marker.getConstructors().length);
    assertEquals(1, marker.getDeclaredConstructors().length);
    assertTrue(Modifier.isPrivate(marker.getDeclaredConstructors()[0].getModifiers()));
  }
}
