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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Set;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestCriticalNativeCalls extends LuceneTestCase {
  @org.apache.lucene.util.SuppressForbidden(
      reason = "Injects an observed duration into a private FFM test kernel without a timing race")
  public void testRetirementUsesOrdinaryNativeAndDeclinesHeap() throws Exception {
    var manifest = PackLoader.verify(TestAccelerationRuntime.nativePack());
    var bridge =
        (NativeBridge)
            Class.forName("org.apache.lucene.internal.acceleration.FfmBridge")
                .getConstructor()
                .newInstance();
    var ledger = new MemoryLedger(new MemoryLedger.Limits(0, 16L << 20, 0), 8L << 20);
    var kernel =
        bridge
            .prepare(
                manifest.library(),
                new AccelerationProvider.Descriptor(
                    "critical", "1", 1, Set.of(Capability.DOCSET_INTERSECT)),
                ledger)
            .bind(Capability.DOCSET_INTERSECT);
    var field = kernel.getClass().getDeclaredField("criticalBudget");
    field.setAccessible(true);
    var budget = (CriticalCallBudget) field.get(kernel);
    budget.record(CriticalCallBudget.MAXIMUM_NANOS + 1);
    try (var scope =
        new RequestMemoryScope(
            ledger, new MemoryLedger.Owner("critical", "test", 0, "critical"), bridge)) {
      var parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      parameters.putLong(0, 1).putLong(16, 64);
      var a = scope.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 0xf0);
      var b = scope.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 0xcc);
      var out = scope.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
      var batch =
          new AccelerationProvider.Batch(
              Capability.DOCSET_INTERSECT,
              List.of(
                  new AccelerationProvider.Input(a, 1, AccelerationProvider.Encoding.I64),
                  new AccelerationProvider.Input(b, 1, AccelerationProvider.Encoding.I64)),
              new AccelerationProvider.Output(out, 1),
              parameters,
              scope.allocate(0),
              scope.allocate(256));
      assertEquals(AccelerationProvider.Status.OK, kernel.execute(batch).status());
      assertEquals(0xc0, out.getLong(0));
      assertEquals(1, kernel.criticalCalls().calls());
      var heapOutput = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 123);
      var heapBatch =
          new AccelerationProvider.Batch(
              Capability.DOCSET_INTERSECT,
              List.of(
                  new AccelerationProvider.Input(
                      ByteBuffer.allocate(8), 1, AccelerationProvider.Encoding.I64),
                  new AccelerationProvider.Input(
                      ByteBuffer.allocate(8), 1, AccelerationProvider.Encoding.I64)),
              new AccelerationProvider.Output(heapOutput, 1),
              parameters,
              scope.allocate(0),
              scope.allocate(256));
      assertEquals(AccelerationProvider.Status.UNSUPPORTED, kernel.execute(heapBatch).status());
      assertEquals(123, heapOutput.getLong(0));
      assertEquals(1, kernel.criticalCalls().calls());
      assertTrue(kernel.criticalCalls().disabled());
    }
    assertEquals(Long.valueOf(0), ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
  }
}
