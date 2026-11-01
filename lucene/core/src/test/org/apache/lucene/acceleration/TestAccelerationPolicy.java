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

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Independent diagnostic budgets, explicit task ownership and drain contracts. */
public class TestAccelerationPolicy extends LuceneTestCase {
  private AccelerationConfig policy(boolean shadow, long shadowBytes) {
    return new AccelerationConfig(
        AccelerationConfig.Mode.AUTO,
        TestAccelerationRuntime.nativePack(),
        Set.of(),
        Set.of(),
        16L << 20,
        8L << 20,
        1,
        true,
        AccelerationConfig.StoragePolicy.CURRENT,
        new AccelerationConfig.MemoryPolicy(8L << 20, 0, 0, 0, 0),
        new AccelerationConfig.ShadowPolicy(shadow, 1, shadowBytes, 4),
        new AccelerationConfig.Tuning(Map.of(), Map.of(), 0, 0));
  }

  public void testWorkspaceBoundsAndDescriptorAdmission() {
    var workspace = new AccelerationProvider.Workspace(16, 8, 4);
    assertEquals(104L, workspace.bytes(new AccelerationProvider.Shape(10, 2, 0)));
    expectThrows(
        ArithmeticException.class,
        () -> workspace.bytes(new AccelerationProvider.Shape(Long.MAX_VALUE, 1, 0)));
    var descriptor = AccelerationProvider.KernelDescriptor.reference(Capability.VECTOR_I8);
    assertTrue(descriptor.supports(new AccelerationProvider.Shape(4, 2, 64)));
    assertFalse(descriptor.supports(new AccelerationProvider.Shape(1L << 30, 2, 64)));
    expectThrows(
        IllegalArgumentException.class, () -> new AccelerationConfig.ShadowPolicy(true, 1, 0, 1));
    expectThrows(
        IllegalArgumentException.class,
        () -> new AccelerationConfig.Tuning(Map.of(), Map.of(Capability.VECTOR_I8, 0), 0, 0));
  }

  public void testOrdinalPartitionConfigurationCannotExceedHeap() {
    expectThrows(
        IllegalArgumentException.class,
        () -> new AccelerationConfig.MemoryPolicy(64, 0, 0, 0, 0, 65));
    assertEquals(32L, new AccelerationConfig.MemoryPolicy(64, 0, 0, 0, 0, 32).ordinalBytes());
    assertEquals(0L, new AccelerationConfig.MemoryPolicy(64, 0, 0, 0, 0).ordinalBytes());
  }

  public void testShadowAdmittedAndDeniedNeverChangesPublication() throws Exception {
    for (long budget : new long[] {1, 1L << 20}) {
      try (var runtime = AccelerationRuntime.forQualification(policy(true, budget))) {
        assertTrue(
            runtime.inventory().rejections().toString(), runtime.supports(Capability.TEST_COPY));
        long[] input = {Long.MIN_VALUE, 0, Long.MAX_VALUE, 0x76543210abcdef01L};
        assertArrayEquals(input, BlockOperations.copy(runtime, input));
        var counter = runtime.inventory().counters().get(Capability.TEST_COPY);
        assertEquals(1L, counter.nativeCalls());
        assertEquals(0L, counter.shadowMismatches());
        assertEquals(budget == 1 ? 0L : 1L, counter.shadowSamples());
        assertEquals(budget == 1 ? 1L : 0L, counter.shadowBudgetDenials());
        assertEquals(0L, runtime.inventory().execution().shadowBytes());
        assertEquals(
            Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
      }
    }
  }

  @SuppressWarnings(
      "try") // Parent retirement while a worker remains pinned is the behavior under test.
  public void testTaskCancellationStopsExistingScopesWithoutFreeingWorkers() throws Exception {
    try (var runtime = AccelerationRuntime.forQualification(policy(false, 0))) {
      var task =
          new AccelerationRuntime.TaskContext(
              new MemoryLedger.Owner("task", "core", 1, "request"), 0);
      try (var binding = runtime.enterTask(task);
          var scope = runtime.newScope()) {
        var worker = scope.forkWorker();
        var bytes = worker.allocate(64);
        bytes.putLong(0, 123L);
        task.cancel();
        scope.close();
        assertEquals(123L, bytes.getLong(0));
        assertFalse(scope.isDrained());
        expectThrows(RequestMemoryScope.AdmissionException.class, () -> worker.allocate(1));
        worker.close();
        assertTrue(scope.isDrained());
      }
      assertEquals(0L, runtime.inventory().execution().activeScopes());
    }
  }

  public void testShutdownDeadlineRetainsPinnedPhysicalCharge() throws Exception {
    var runtime = AccelerationRuntime.forQualification(policy(false, 0));
    var scope = runtime.newScope();
    var worker = scope.forkWorker();
    var bytes = worker.allocate(128);
    bytes.putLong(0, 456L);
    var failure = expectThrows(IOException.class, () -> runtime.shutdown(System.nanoTime()));
    assertTrue(failure.getMessage(), failure.getMessage().contains("pinned"));
    assertEquals(456L, bytes.getLong(0));
    assertTrue(runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE) >= 128);
    worker.close();
    scope.close();
    runtime.shutdown(System.nanoTime() + 1_000_000_000L);
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testShutdownNeverClosesCallerOwnedSpan() throws Exception {
    var runtime = AccelerationRuntime.forQualification(policy(false, 0));
    var scope = runtime.newScope();
    var bytes = scope.allocate(128);
    bytes.putLong(0, 789L);
    expectThrows(IOException.class, () -> runtime.shutdown(System.nanoTime()));
    assertEquals(789L, bytes.getLong(0));
    assertFalse(scope.isDrained());
    expectThrows(IOException.class, runtime::checkpoint);
    scope.close();
    runtime.shutdown(System.nanoTime() + 1_000_000_000L);
    assertEquals(0L, runtime.inventory().execution().activeScopes());
  }

  public void testPlatformVersionSuffixDoesNotBecomeAFeature() {
    assertEquals(0, NativeBridge.compareVersion("6.6.87.2-microsoft-standard-WSL2", "6.6.87.2"));
    assertTrue(NativeBridge.compareVersion("6.6.87.2-microsoft-standard-WSL2", "6.6.87.2.1") < 0);
    expectThrows(
        IllegalArgumentException.class, () -> NativeBridge.compareVersion("6.6", "9".repeat(40)));
  }
}
