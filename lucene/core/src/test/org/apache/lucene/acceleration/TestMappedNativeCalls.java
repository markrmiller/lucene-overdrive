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
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Set;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Cold mapped storage cannot enter a critical call, even for a short gathered batch. */
public class TestMappedNativeCalls extends LuceneTestCase {
  private NativeBridge bridge() throws Exception {
    TestAccelerationRuntime.nativePack();
    return (NativeBridge)
        Class.forName("org.apache.lucene.internal.acceleration.FfmBridge")
            .getConstructor()
            .newInstance();
  }

  private static byte[] bytes(ByteBuffer buffer) {
    byte[] bytes = new byte[buffer.remaining()];
    buffer.get(buffer.position(), bytes);
    return bytes;
  }

  private static boolean mapped(ByteBuffer buffer) throws Exception {
    var api = Class.forName("java.lang.foreign.MemorySegment");
    Object segment = api.getMethod("ofBuffer", java.nio.Buffer.class).invoke(null, buffer);
    return (boolean) api.getMethod("isMapped").invoke(segment);
  }

  public void testMMapDirectoryGatherRetainsMappedFlagAndUsesOrdinaryCall() throws Exception {
    NativeBridge bridge = bridge();
    var ledger = new MemoryLedger(new MemoryLedger.Limits(0, 16L << 20, 0), 8L << 20);
    var kernel =
        bridge
            .prepare(
                PackLoader.verify(TestAccelerationRuntime.nativePack()).library(),
                new AccelerationProvider.Descriptor(
                    "mapped", "1", 1, Set.of(Capability.OSQ_GATHER)),
                ledger)
            .bind(Capability.OSQ_GATHER);
    var fixture = KernelFixtures.fixtures(Capability.OSQ_GATHER).getFirst();
    try (var scope =
            new RequestMemoryScope(
                ledger, new MemoryLedger.Owner("mapped", "test", 0, "mapped"), bridge);
        var directory = new MMapDirectory(Path.of(createTempDir("mapped-gather").toString()))) {
      var base = fixture.operation().stage(scope);
      byte[] rows = bytes(base.inputs().get(1).bytes());
      try (var out = directory.createOutput("rows", IOContext.DEFAULT)) {
        out.writeBytes(rows, rows.length);
      }
      try (var input = directory.openInput("rows", IOContext.DEFAULT)) {
        ByteBuffer region = MappedSlices.slice(input, 0, rows.length);
        assertNotNull(region);
        assertTrue(
            "MappedSlices -> ByteBuffer -> MemorySegment must retain mapping", mapped(region));
        assertFalse(mapped(base.inputs().getFirst().bytes()));
        var inputs = new ArrayList<>(base.inputs());
        inputs.set(
            1,
            new AccelerationProvider.Input(
                region, base.inputs().get(1).elements(), AccelerationProvider.Encoding.BYTES));
        var nativeBatch =
            new AccelerationProvider.Batch(
                base.capability(),
                inputs,
                base.output(),
                base.parameters(),
                base.scratch(),
                base.metadata());
        assertTrue(bridge.isMapped(nativeBatch));
        assertEquals(AccelerationProvider.Status.OK, kernel.execute(nativeBatch).status());
        assertArrayEquals(fixture.expected(), bytes(base.output().bytes()));
        inputs.set(
            0,
            new AccelerationProvider.Input(
                ByteBuffer.wrap(bytes(inputs.getFirst().bytes())),
                inputs.getFirst().elements(),
                inputs.getFirst().encoding()));
        var heapBatch =
            new AccelerationProvider.Batch(
                base.capability(),
                inputs,
                base.output(),
                base.parameters(),
                base.scratch(),
                base.metadata());
        base.output().bytes().put(0, (byte) 0x5a);
        assertEquals(
            "A critical entry would accept this valid heap query; an ordinary entry must decline it",
            AccelerationProvider.Status.UNSUPPORTED,
            kernel.execute(heapBatch).status());
        assertEquals(0x5a, base.output().bytes().get(0));
      }
    }
    assertEquals(0L, ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
  }

  public void testMappedInputOutputAndWorkspacesAllPreventCriticalCall() throws Exception {
    NativeBridge bridge = bridge();
    var ledger = new MemoryLedger(new MemoryLedger.Limits(0, 16L << 20, 0), 8L << 20);
    var kernel =
        bridge
            .prepare(
                PackLoader.verify(TestAccelerationRuntime.nativePack()).library(),
                new AccelerationProvider.Descriptor(
                    "mapped-scopes", "1", 1, Set.of(Capability.OSQ_GATHER)),
                ledger)
            .bind(Capability.OSQ_GATHER);
    var fixture = KernelFixtures.fixtures(Capability.OSQ_GATHER).getFirst();
    try (var scope =
        new RequestMemoryScope(
            ledger, new MemoryLedger.Owner("mapped", "all-scopes", 0, "mapped"), bridge)) {
      var base = fixture.operation().stage(scope);
      assertFalse(bridge.isMapped(base));
      assertEquals(AccelerationProvider.Status.OK, kernel.execute(base).status());
      for (int slot = 0; slot < 7; slot++) {
        ByteBuffer original =
            slot < 3
                ? base.inputs().get(slot).bytes()
                : slot == 3
                    ? base.output().bytes()
                    : slot == 4 ? base.parameters() : slot == 5 ? base.scratch() : base.metadata();
        Path file = Path.of(createTempFile("mapped-slot-" + slot, ".bin").toString());
        Files.write(file, bytes(original));
        try (FileChannel channel =
            FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
          ByteBuffer view =
              channel
                  .map(FileChannel.MapMode.READ_WRITE, 0, Math.max(1, original.remaining()))
                  .order(ByteOrder.LITTLE_ENDIAN);
          assertTrue(mapped(view));
          var inputs = new ArrayList<>(base.inputs());
          int heapSlot = slot == 0 ? 2 : 0;
          var heap = inputs.get(heapSlot);
          inputs.set(
              heapSlot,
              new AccelerationProvider.Input(
                  ByteBuffer.wrap(bytes(heap.bytes())), heap.elements(), heap.encoding()));
          if (slot < 3) {
            var old = inputs.get(slot);
            inputs.set(slot, new AccelerationProvider.Input(view, old.elements(), old.encoding()));
          }
          var batch =
              new AccelerationProvider.Batch(
                  base.capability(),
                  inputs,
                  new AccelerationProvider.Output(
                      slot == 3 ? view : base.output().bytes(), base.output().elements()),
                  slot == 4 ? view : base.parameters(),
                  slot == 5 ? view : base.scratch(),
                  slot == 6 ? view : base.metadata());
          assertTrue("mapped slot=" + slot, bridge.isMapped(batch));
          var nativeInputs = new ArrayList<>(inputs);
          nativeInputs.set(heapSlot, base.inputs().get(heapSlot));
          var nativeBatch =
              new AccelerationProvider.Batch(
                  base.capability(),
                  nativeInputs,
                  batch.output(),
                  batch.parameters(),
                  batch.scratch(),
                  batch.metadata());
          assertEquals(
              "native mapped slot=" + slot,
              AccelerationProvider.Status.OK,
              kernel.execute(nativeBatch).status());
          assertEquals(
              "heap mapped slot=" + slot,
              AccelerationProvider.Status.UNSUPPORTED,
              kernel.execute(batch).status());
        }
      }
    }
  }
}
