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

/** Typed array inputs never need a ByteBuffer view or a materializing primary copy. */
public class TestBorrowedMaskedFacetInputs extends LuceneTestCase {
  private static final class Fixture implements AutoCloseable {
    final NativeBridge bridge;
    final RequestMemoryScope scope;
    final AccelerationProvider.Kernel kernel;

    Fixture() throws Exception {
      bridge =
          (NativeBridge)
              Class.forName("org.apache.lucene.internal.acceleration.FfmBridge")
                  .getConstructor()
                  .newInstance();
      var ledger = new MemoryLedger(new MemoryLedger.Limits(0, 64L << 20, 0), 64L << 20);
      kernel =
          bridge
              .prepare(
                  PackLoader.verify(TestAccelerationRuntime.nativePack()).library(),
                  new AccelerationProvider.Descriptor(
                      "borrowed-test", "1", 1, Set.of(Capability.MASKED_FACETS)),
                  ledger)
              .bind(Capability.MASKED_FACETS);
      scope =
          new RequestMemoryScope(
              ledger, new MemoryLedger.Owner("borrowed", "test", 0, "borrowed"), bridge);
    }

    @Override
    public void close() {
      scope.close();
    }
  }

  private static AccelerationProvider.BorrowedSpan arraySpan(Object array) throws Exception {
    var api = Class.forName("java.lang.foreign.MemorySegment");
    Object segment = api.getMethod("ofArray", array.getClass()).invoke(null, array);
    segment = api.getMethod("asReadOnly").invoke(segment);
    int width = array instanceof long[] ? 8 : array instanceof int[] ? 4 : 2;
    int length = java.lang.reflect.Array.getLength(array);
    Object readonly = segment;
    return new AccelerationProvider.BorrowedSpan() {
      @Override
      public long byteSize() {
        return (long) length * width;
      }

      @Override
      public boolean isNative() {
        return false;
      }

      @Override
      public Object foreignSegment() {
        return readonly;
      }

      @Override
      public byte byteAt(long offset) {
        java.util.Objects.checkIndex(offset, byteSize());
        long value =
            array instanceof long[] values
                ? values[(int) (offset / 8)]
                : array instanceof int[] values
                    ? Integer.toUnsignedLong(values[(int) (offset / 4)])
                    : Short.toUnsignedLong(((short[]) array)[(int) (offset / 2)]);
        return (byte) (value >>> ((offset % width) * 8));
      }
    };
  }

  private static AccelerationProvider.Batch batch(
      Fixture fixture,
      AccelerationProvider.BorrowedSpan words,
      AccelerationProvider.BorrowedSpan column,
      int documents,
      int width,
      ByteBuffer output) {
    ByteBuffer parameters = fixture.scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    parameters
        .putLong(0, (documents + 63L) / 64)
        .putLong(8, 9)
        .putLong(16, 0)
        .putLong(24, documents)
        .putLong(32, 0)
        .putLong(40, width)
        .putLong(48, documents);
    return new AccelerationProvider.Batch(
        Capability.MASKED_FACETS,
        List.of(
            AccelerationProvider.Input.borrowed(
                words, (documents + 63L) / 64, AccelerationProvider.Encoding.I64),
            AccelerationProvider.Input.borrowed(
                column, (long) documents * width, AccelerationProvider.Encoding.BYTES)),
        new AccelerationProvider.Output(output, 9),
        parameters,
        fixture.scope.allocate(0),
        fixture.scope.allocate(256));
  }

  public void testLongWordsAndShortIntColumnsWithoutByteBufferView() throws Exception {
    for (int width : new int[] {2, 4}) {
      try (var fixture = new Fixture()) {
        int documents = 1031;
        long[] words = new long[(documents + 63) / 64];
        for (int i = 0; i < words.length; i++) words[i] = random().nextLong();
        Object column = width == 2 ? new short[documents] : new int[documents];
        for (int i = 0; i < documents; i++)
          java.lang.reflect.Array.set(
              column, i, width == 2 ? (Object) (short) (i % 9) : (Object) (i % 9));
        ByteBuffer output = fixture.scope.allocate(36).order(ByteOrder.LITTLE_ENDIAN);
        var primary = batch(fixture, arraySpan(words), arraySpan(column), documents, width, output);
        assertNull(primary.inputs().getFirst().bytes());
        assertNull(primary.inputs().get(1).bytes());
        ByteBuffer reference = ByteBuffer.allocate(36).order(ByteOrder.LITTLE_ENDIAN);
        var oracle =
            new AccelerationProvider.Batch(
                primary.capability(),
                primary.inputs(),
                new AccelerationProvider.Output(reference, 9),
                primary.parameters(),
                primary.scratch(),
                primary.metadata());
        assertEquals(AccelerationProvider.Status.OK, JavaKernelOracle.execute(oracle).status());
        var guarded =
            primary.inputs().stream()
                .map(
                    input -> {
                      var span = input.span();
                      var noRead =
                          new AccelerationProvider.BorrowedSpan() {
                            @Override
                            public long byteSize() {
                              return span.byteSize();
                            }

                            @Override
                            public boolean isNative() {
                              return span.isNative();
                            }

                            @Override
                            public Object foreignSegment() {
                              return span.foreignSegment();
                            }

                            @Override
                            public byte byteAt(long offset) {
                              throw new AssertionError(
                                  "primary structural validation must not materialize or scan typed storage");
                            }
                          };
                      return AccelerationProvider.Input.borrowed(
                          noRead, input.elements(), input.encoding());
                    })
                .toList();
        var noScan =
            new AccelerationProvider.Batch(
                primary.capability(),
                guarded,
                primary.output(),
                primary.parameters(),
                primary.scratch(),
                primary.metadata());
        JavaKernelOracle.validate(noScan);
        assertEquals(AccelerationProvider.Status.OK, fixture.kernel.execute(noScan).status());
        for (int bucket = 0; bucket < 9; bucket++)
          assertEquals(reference.getInt(bucket * 4), output.getInt(bucket * 4));
        for (int repeat = 0; repeat < 3; repeat++) {
          var observation = fixture.kernel.criticalCalls();
          var result = fixture.kernel.execute(primary);
          if (observation.disabled()) {
            assertEquals(AccelerationProvider.Status.UNSUPPORTED, result.status());
            assertEquals(observation.calls(), fixture.kernel.criticalCalls().calls());
            for (int bucket = 0; bucket < 9; bucket++)
              assertEquals(reference.getInt(bucket * 4), output.getInt(bucket * 4));
            assertEquals(
                AccelerationProvider.Status.OK, JavaKernelOracle.execute(primary).status());
          } else {
            assertEquals(AccelerationProvider.Status.OK, result.status());
          }
          for (int bucket = 0; bucket < 9; bucket++)
            assertEquals(reference.getInt(bucket * 4), output.getInt(bucket * 4));
        }
      }
    }
  }

  public void testRuntimeShadowCopiesTypedInputsOnlyForDiagnostics() throws Exception {
    var config =
        new AccelerationConfig(
            AccelerationConfig.Mode.AUTO,
            TestAccelerationRuntime.nativePack(),
            Set.of(),
            Set.of(),
            64L << 20,
            16L << 20,
            1,
            true,
            AccelerationConfig.StoragePolicy.CURRENT,
            new AccelerationConfig.MemoryPolicy(16L << 20, 0, 0, 0, 0),
            new AccelerationConfig.ShadowPolicy(true, 1, 8L << 20, 4),
            new AccelerationConfig.Tuning(java.util.Map.of(), java.util.Map.of(), 0, 0));
    try (var runtime = AccelerationRuntime.forQualification(config);
        var fixture = new Fixture()) {
      int documents = 1024;
      long[] words = new long[16];
      java.util.Arrays.fill(words, -1);
      short[] column = new short[documents];
      for (int i = 0; i < documents; i++) column[i] = (short) (i % 9);
      ByteBuffer output = fixture.scope.allocate(36).order(ByteOrder.LITTLE_ENDIAN);
      var primary = batch(fixture, arraySpan(words), arraySpan(column), documents, 2, output);
      for (int i = 0; i < 9; i++) output.putInt(i * 4, 1234);
      boolean retired = runtime.criticalCalls().get(Capability.MASKED_FACETS).disabled();
      assertEquals(!retired, runtime.execute(primary, documents, 0));
      var counter = runtime.inventory().counters().get(Capability.MASKED_FACETS);
      assertEquals(retired ? 0L : 1L, counter.nativeCalls());
      assertEquals(retired ? 0L : 1L, counter.shadowSamples());
      assertEquals(0L, counter.shadowMismatches());
      assertEquals(0L, counter.stagedBytes());
      if (retired) {
        for (int i = 0; i < 9; i++) assertEquals(1234, output.getInt(i * 4));
        assertEquals(AccelerationProvider.Status.OK, JavaKernelOracle.execute(primary).status());
      }
      int total = 0;
      for (int i = 0; i < 9; i++) total += output.getInt(i * 4);
      assertEquals(documents, total);
    }
  }

  public void testLargeTypedHeapDeclinesOrdinaryCallWithoutPublishing() throws Exception {
    try (var fixture = new Fixture()) {
      int documents = 200000;
      long[] words = new long[(documents + 63) / 64];
      short[] column = new short[documents];
      ByteBuffer output = fixture.scope.allocate(36).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 9; i++) output.putInt(i * 4, 1234);
      var primary = batch(fixture, arraySpan(words), arraySpan(column), documents, 2, output);
      assertEquals(
          AccelerationProvider.Status.UNSUPPORTED, fixture.kernel.execute(primary).status());
      for (int i = 0; i < 9; i++) assertEquals(1234, output.getInt(i * 4));
    }
  }

  public void testOpaqueStorageAndMismatchedTraitsRejected() throws Exception {
    try (var fixture = new Fixture()) {
      var valid = arraySpan(new long[] {-1});
      var invalid =
          new AccelerationProvider.BorrowedSpan() {
            @Override
            public long byteSize() {
              return 8;
            }

            @Override
            public boolean isNative() {
              return true;
            }

            @Override
            public byte byteAt(long offset) {
              return 0;
            }

            @Override
            public Object foreignSegment() {
              return valid.foreignSegment();
            }
          };
      var primary =
          batch(fixture, invalid, arraySpan(new short[64]), 64, 2, fixture.scope.allocate(36));
      var invalidBatch = primary;
      expectThrows(IllegalArgumentException.class, () -> fixture.kernel.execute(invalidBatch));
      var opaque =
          new AccelerationProvider.BorrowedSpan() {
            @Override
            public long byteSize() {
              return 8;
            }

            @Override
            public boolean isNative() {
              return false;
            }

            @Override
            public byte byteAt(long offset) {
              return 0;
            }

            @Override
            public Object foreignSegment() {
              return "not a memory segment";
            }
          };
      primary = batch(fixture, opaque, arraySpan(new short[64]), 64, 2, fixture.scope.allocate(36));
      var opaqueBatch = primary;
      expectThrows(IllegalArgumentException.class, () -> fixture.kernel.execute(opaqueBatch));
    }
  }

  public void testClosedBorrowedNativeScopeRejectedBeforeEntry() throws Exception {
    try (var fixture = new Fixture()) {
      Class<?> arenaApi = Class.forName("java.lang.foreign.Arena");
      Class<?> segmentApi = Class.forName("java.lang.foreign.MemorySegment");
      Object arena = arenaApi.getMethod("ofShared").invoke(null);
      Object segment = arenaApi.getMethod("allocate", long.class, long.class).invoke(arena, 8L, 8L);
      segmentApi.getMethod("fill", byte.class).invoke(segment, (byte) -1);
      Object readonly = segmentApi.getMethod("asReadOnly").invoke(segment);
      var span =
          new AccelerationProvider.BorrowedSpan() {
            @Override
            public long byteSize() {
              return 8;
            }

            @Override
            public boolean isNative() {
              return true;
            }

            @Override
            public byte byteAt(long offset) {
              return -1;
            }

            @Override
            public Object foreignSegment() {
              return readonly;
            }
          };
      var staged =
          batch(fixture, span, arraySpan(new short[64]), 64, 2, fixture.scope.allocate(36));
      var nativeInputs = new java.util.ArrayList<>(staged.inputs());
      nativeInputs.set(
          1,
          new AccelerationProvider.Input(
              fixture.scope.allocate(128), 128, AccelerationProvider.Encoding.BYTES));
      var primary =
          new AccelerationProvider.Batch(
              staged.capability(),
              nativeInputs,
              staged.output(),
              staged.parameters(),
              staged.scratch(),
              staged.metadata());
      for (int i = 0; i < 3; i++)
        assertEquals(AccelerationProvider.Status.OK, fixture.kernel.execute(primary).status());
      arenaApi.getMethod("close").invoke(arena);
      expectThrows(IllegalStateException.class, () -> fixture.kernel.execute(primary));
    }
  }
}
