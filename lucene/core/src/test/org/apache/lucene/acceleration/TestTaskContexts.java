/* Licensed to the Apache Software Foundation (ASF) under one or more
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
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestTaskContexts extends LuceneTestCase {
  public void testCombinedWorkerAdmissionAndCancellation() throws Exception {
    assumeTrue("bounded FFM scopes need JDK25", Runtime.version().feature() >= 25);
    var runtime =
        new AccelerationRuntime(
            new AccelerationConfig(
                AccelerationConfig.Mode.OFF,
                null,
                Set.of(),
                Set.of(),
                4096,
                1000,
                1,
                false,
                AccelerationConfig.StoragePolicy.SELECTIVE_OFF_HEAP));
    var context =
        new AccelerationRuntime.TaskContext(
            new MemoryLedger.Owner("task", "core", 42, "shared-request"), 0);
    AtomicReference<Throwable> failure = new AtomicReference<>();
    try (var binding = runtime.enterTask(context);
        var scope = runtime.newScope()) {
      assertNotNull(binding);
      var buffer = scope.allocate(600);
      buffer.put(0, (byte) 7);
      Thread worker =
          new Thread(
              () -> {
                try (var workerBinding = runtime.enterTask(context);
                    var workerScope = runtime.newScope()) {
                  assertNotNull(workerBinding);
                  var denied =
                      expectThrows(
                          RequestMemoryScope.AdmissionException.class,
                          () -> workerScope.allocate(600));
                  assertEquals(RequestMemoryScope.Failure.REQUEST_LIMIT, denied.reason());
                  assertEquals(280, workerScope.allocate(280).remaining());
                  context.cancel();
                  expectThrows(InterruptedIOException.class, runtime::checkpoint);
                } catch (Throwable t) {
                  failure.set(t);
                }
              });
      worker.start();
      worker.join();
      if (failure.get() != null) throw new AssertionError(failure.get());
      assertEquals(7, buffer.get(0)); // cancellation did not close a live parent's memory
      expectThrows(InterruptedIOException.class, runtime::newScope);
    }
    runtime.checkpoint(); // the thread binding was removed
    assertEquals(
        0L, runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE).longValue());
  }

  public void testHostPolicyAppliesToRuntimeHeapAndNativeScope() throws Exception {
    assumeTrue("bounded FFM scopes need JDK25", Runtime.version().feature() >= 25);
    var config =
        new AccelerationConfig(
            AccelerationConfig.Mode.OFF,
            null,
            Set.of(),
            Set.of(),
            4096,
            1000,
            1,
            false,
            AccelerationConfig.StoragePolicy.SELECTIVE_OFF_HEAP,
            new AccelerationConfig.MemoryPolicy(4096, 0, 0, 0, 0, 0),
            new AccelerationConfig.ShadowPolicy(false, 1000, 0, 16),
            new AccelerationConfig.Tuning(java.util.Map.of(), java.util.Map.of(), 0, 0));
    try (var runtime = new AccelerationRuntime(config)) {
      var attempts = new java.util.concurrent.atomic.AtomicInteger();
      var owner = new MemoryLedger.Owner("host", "core", 0, "shared-host");
      var task =
          new AccelerationRuntime.TaskContext(
              owner,
              0,
              (who, kind, bytes, purpose, cancelled) -> {
                assertEquals(owner, who);
                assertFalse(cancelled.getAsBoolean());
                attempts.incrementAndGet();
                return runtime.ledger().tryReserve(who, kind, bytes, purpose);
              });
      try (var binding = runtime.enterTask(task);
          var heap = runtime.reserveHeap(400);
          var scope = runtime.newScope()) {
        assertNotNull(binding);
        assertEquals(400, heap.remaining());
        assertEquals(128, scope.allocate(128).remaining());
        assertEquals(2, attempts.get());
        assertEquals(owner, runtime.currentOwner());
      }
      assertTrue(runtime.ledger().awaitDrained(java.time.Duration.ZERO));
    }
  }

  public void testNestedBindingsAndExpiredDeadline() throws Exception {
    var runtime =
        new AccelerationRuntime(
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
    var owner = new MemoryLedger.Owner("task", "core", 0, "request");
    try (var outer = runtime.enterTask(new AccelerationRuntime.TaskContext(owner, 0))) {
      assertNotNull(outer);
      try (var inner =
          runtime.enterTask(new AccelerationRuntime.TaskContext(owner, System.nanoTime() - 1))) {
        assertNotNull(inner);
        expectThrows(InterruptedIOException.class, runtime::checkpoint);
        expectThrows(IllegalStateException.class, outer::close);
      }
      runtime.checkpoint();
    }
    runtime.checkpoint();
    var same = new AccelerationRuntime.TaskContext(owner, 0);
    try (var outer = runtime.enterTask(same);
        var inner = runtime.enterTask(same)) {
      assertNotNull(inner);
      expectThrows(IllegalStateException.class, outer::close);
    }
    runtime.checkpoint();
  }
}
