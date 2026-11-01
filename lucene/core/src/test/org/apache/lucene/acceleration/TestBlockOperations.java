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
package org.apache.lucene.acceleration;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.NativeAllocator;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.ScalarQuantizedVectorSimilarity;

public class TestBlockOperations extends LuceneTestCase {
  public void testOwnedOnlyVectorProfileDeclinesGenericBeforeNativeAdmission() throws Exception {
    checkGenericHelperContext(Capability.VECTOR_I8, OwnedByteBatchScorer.EXACT_OWNED_CONTEXT);
  }

  public void testJsonOnlyAggregateProfileDeclinesGenericBeforeNativeAdmission() throws Exception {
    checkGenericHelperContext(Capability.DOCVALUES_AGGREGATE, "solr.json-range.full-live.v1");
  }

  private void checkGenericHelperContext(Capability capability, String context) throws Exception {
    try (var runtime = helperRuntime(capability, context, false)) {
      assertTrue(runtime.supports(capability, context));
      assertFalse(runtime.supports(capability, "generic"));
      var before = runtime.inventory();
      var task =
          new AccelerationRuntime.TaskContext(
              new MemoryLedger.Owner("test", "generic-block-context", 0, capability.id),
              0,
              (owner, kind, bytes, purpose, cancelled) -> {
                if (kind == MemoryLedger.Kind.NATIVE && bytes > 0)
                  throw new AssertionError("generic helper crossed native admission barrier");
                return runtime.ledger().tryReserve(owner, kind, bytes, purpose);
              });
      try (var binding = runtime.enterTask(task)) {
        assertNotNull(binding);
        checkHelperResult(runtime, capability);
      } finally {
        task.cancel();
      }
      var after = runtime.inventory();
      assertEquals(before.memory(), after.memory());
      assertEquals(before.execution(), after.execution());
      var oldCounter = before.counters().get(capability);
      var counter = after.counters().get(capability);
      assertEquals(oldCounter.nativeCalls(), counter.nativeCalls());
      assertEquals(oldCounter.javaCalls() + 1, counter.javaCalls());
      assertEquals(
          oldCounter.fallbacks().getOrDefault(AccelerationRuntime.Reason.NO_MEASURED_WIN, 0L) + 1,
          counter.fallbacks().get(AccelerationRuntime.Reason.NO_MEASURED_WIN).longValue());
    }
  }

  public void testGenericAggregateProfileStillExecutesAndQualificationIgnoresContextRows()
      throws Exception {
    for (boolean qualification : new boolean[] {false, true}) {
      try (var runtime =
          helperRuntime(
              Capability.DOCVALUES_AGGREGATE,
              qualification ? "solr.json-range.full-live.v1" : "generic",
              qualification)) {
        var before = runtime.inventory();
        checkHelperResult(runtime, Capability.DOCVALUES_AGGREGATE);
        var counter = runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE);
        assertEquals(
            before.counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls() + 1,
            counter.nativeCalls());
        assertEquals(
            before.counters().get(Capability.DOCVALUES_AGGREGATE).javaCalls(), counter.javaCalls());
        assertEquals(before.memory().charged(), runtime.inventory().memory().charged());
        assertEquals(before.execution(), runtime.inventory().execution());
      }
    }
  }

  private static void checkHelperResult(AccelerationRuntime runtime, Capability capability)
      throws Exception {
    if (capability == Capability.VECTOR_I8) {
      byte[] target = new byte[384];
      Arrays.fill(target, (byte) 1);
      byte[][] docs = new byte[512][384];
      for (int i = 0; i < docs.length; i++) Arrays.fill(docs[i], (byte) (i % 251 - 125));
      float[] actual =
          BlockOperations.byteScores(runtime, target, docs, VectorSimilarityFunction.DOT_PRODUCT);
      for (int i = 0; i < actual.length; i++)
        assertEquals(VectorSimilarityFunction.DOT_PRODUCT.compare(target, docs[i]), actual[i], 0f);
    } else {
      byte[] encoded = new byte[1025 * 4];
      ByteBuffer bytes = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 1025; i++) bytes.putInt(i);
      assertEquals(
          new BlockOperations.Aggregate(1025, 524800, 0, 1024, true, false),
          BlockOperations.aggregate(
              runtime, encoded, 0, 1025, 32, 4, 1, 0, null, Long.MIN_VALUE, Long.MAX_VALUE));
    }
  }

  private static AccelerationRuntime helperRuntime(
      Capability capability, String context, boolean qualification) throws Exception {
    var pack = TestAccelerationRuntime.nativePack();
    var policy = TrustedPathFixtures.file("generic-block-context");
    boolean vector = capability == Capability.VECTOR_I8;
    String shape =
        vector
            ? "384\t512\t512\t1\t-"
            : "32\t1025\t1025\t" + Long.MIN_VALUE + "\t" + Long.MAX_VALUE;
    Files.writeString(
        policy,
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + capability.id
            + "\t"
            + shape
            + "\t"
            + (context.equals("generic") ? "native" : "mapped")
            + "\t200000\tTEST-ONLY-generic-helper-context\t"
            + context
            + "\n");
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    String key = "lucene.acceleration.native.qualificationPath";
    String previous = System.getProperty(key);
    System.setProperty(key, policy.toString());
    try {
      var config =
          TestAccelerationRuntime.config(AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1);
      return qualification
          ? AccelerationRuntime.forQualification(config)
          : new AccelerationRuntime(config);
    } finally {
      if (previous == null) System.clearProperty(key);
      else System.setProperty(key, previous);
    }
  }

  public void testDecodeAllWidthsOffsetsPaddingAndWrapping() throws Exception {
    for (int iteration = 0; iteration < 200; iteration++) {
      int bits = 8 * (1 + random().nextInt(8));
      int count = random().nextInt(100),
          offset = random().nextInt(7),
          stride = bits / 8 + random().nextInt(3);
      byte[] source = new byte[offset + count * stride + 9];
      random().nextBytes(source);
      long multiplier = random().nextLong(), base = random().nextLong();
      long[] actual =
          BlockOperations.decode(null, source, offset, count, bits, stride, multiplier, base);
      for (int i = 0; i < count; i++) {
        long expected = 0;
        for (int j = bits / 8 - 1; j >= 0; j--)
          expected = (expected << 8) | (source[offset + i * stride + j] & 255L);
        assertEquals(expected * multiplier + base, actual[i]);
      }
      long[] destination = new long[count + 2];
      Arrays.fill(destination, 42);
      BlockOperations.decodeInto(
          null, source, offset, count, bits, stride, multiplier, base, destination, 1);
      assertEquals(42L, destination[0]);
      assertEquals(42L, destination[destination.length - 1]);
      assertArrayEquals(
          actual, org.apache.lucene.util.ArrayUtil.copyOfSubArray(destination, 1, count + 1));
    }
    expectThrows(
        IllegalArgumentException.class,
        () -> BlockOperations.decode(null, new byte[4], 0, 1, 7, 1, 1, 0));
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> BlockOperations.decode(null, new byte[1], 0, 1, 16, 2, 1, 0));
  }

  public void testBitAlgebraAndSignedRange() throws Exception {
    for (int iteration = 0; iteration < 100; iteration++) {
      int length = random().nextInt(500), words = (length + 63) >>> 6;
      long[] a = new long[words], b = new long[words];
      for (int i = 0; i < words; i++) {
        a[i] = random().nextLong();
        b[i] = random().nextLong();
      }
      BitSet left = BitSet.valueOf(a), right = BitSet.valueOf(b);
      left.clear(length, Math.max(length, left.length()));
      right.clear(length, Math.max(length, right.length()));
      assertEquals(left.cardinality(), BlockOperations.cardinality(null, a, length));
      BitSet both = (BitSet) left.clone();
      both.and(right);
      assertEquals(both.cardinality(), BlockOperations.intersectionCount(null, a, b, length));
      for (Capability capability :
          List.of(
              Capability.DOCSET_INTERSECT, Capability.DOCSET_UNION, Capability.DOCSET_DIFFERENCE)) {
        BitSet expected = (BitSet) left.clone();
        switch (capability) {
          case DOCSET_INTERSECT -> expected.and(right);
          case DOCSET_UNION -> expected.or(right);
          case DOCSET_DIFFERENCE -> expected.andNot(right);
          // $CASES-OMITTED$
          default -> throw new AssertionError();
        }
        assertEquals(
            expected, BitSet.valueOf(BlockOperations.bitset(null, capability, a, b, length)));
      }
    }
    for (int iteration = 0; iteration < 100; iteration++) {
      int count = random().nextInt(300), words = (count + 63) >>> 6;
      long[] values = new long[count], selection = new long[words];
      for (int i = 0; i < count; i++) values[i] = random().nextInt(41) - 20;
      for (int w = 0; w < words; w++)
        selection[w] = random().nextBoolean() ? 0 : random().nextLong();
      long minimum = random().nextInt(41) - 20, maximum = random().nextInt(41) - 20;
      long[] expected = BlockOperations.range(null, values, minimum, maximum);
      for (int w = 0; w < words; w++) expected[w] &= selection[w];
      assertArrayEquals(
          expected, BlockOperations.rangeFilter(null, values, selection, minimum, maximum));
    }
    long[] boundaries = {Long.MIN_VALUE, -1, 0, 1, Long.MAX_VALUE};
    assertArrayEquals(new long[] {14}, BlockOperations.range(null, boundaries, -1, 1));
    assertArrayEquals(new long[] {0}, BlockOperations.range(null, boundaries, 1, -1));
    assertArrayEquals(new long[0], BlockOperations.range(null, new long[0], -1, 1));
  }

  public void testAggregateAdmissionLiveDocsAndOverflow() throws Exception {
    ByteBuffer input = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
    input.putLong(Long.MAX_VALUE).putLong(1).putLong(-7).putLong(Long.MIN_VALUE);
    var overflow =
        BlockOperations.aggregate(
            null, input.array(), 0, 4, 64, 8, 1, 0, new long[] {3}, Long.MIN_VALUE, Long.MAX_VALUE);
    assertEquals(
        new BlockOperations.Aggregate(2, Long.MIN_VALUE, 1, Long.MAX_VALUE, true, true), overflow);
    var filtered =
        BlockOperations.aggregate(null, input.array(), 0, 4, 64, 8, 1, 0, new long[] {6}, -7, 1);
    assertEquals(new BlockOperations.Aggregate(2, -6, -7, 1, true, false), filtered);
    var empty =
        BlockOperations.aggregate(
            null, input.array(), 0, 4, 64, 8, 1, 0, new long[] {0}, Long.MIN_VALUE, Long.MAX_VALUE);
    assertEquals(new BlockOperations.Aggregate(0, 0, 0, 0, false, false), empty);
  }

  public void testHistogramAndSplitLayout() throws Exception {
    assertArrayEquals(
        new int[] {1, 0, 2, 1, 0}, BlockOperations.histogram(null, new int[] {2, 0, -1, 2, 3}, 5));
    expectThrows(
        IllegalArgumentException.class, () -> BlockOperations.histogram(null, new int[] {-2}, 3));
    expectThrows(
        IllegalArgumentException.class, () -> BlockOperations.histogram(null, new int[] {3}, 3));
    assertArrayEquals(
        new int[] {0xab, 0x12, 0xcd, 0x34, 0xef, 0x56, 0x12, 0x78},
        BlockOperations.split(null, new int[] {0xabcdef12, 0x12345678}, 24, 8, 255, 255));
    for (int iteration = 0; iteration < 100; iteration++) {
      int count = random().nextInt(30),
          bits = 1 + random().nextInt(31),
          decrement = 1 + random().nextInt(31);
      int[] source = new int[count];
      for (int i = 0; i < count; i++) source[i] = random().nextInt();
      int mask = random().nextInt(), remainderMask = random().nextInt();
      int[] result = BlockOperations.split(null, source, bits, decrement, mask, remainderMask);
      int block = 0;
      for (int shift = bits; shift > 0; shift -= decrement, block++)
        for (int i = 0; i < count; i++)
          assertEquals((source[i] >>> shift) & mask, result[block * count + i]);
      for (int i = 0; i < count; i++)
        assertEquals(source[i] & remainderMask, result[block * count + i]);
    }
  }

  public void testQuantizedPackingAndTranspositionAgainstLucene() throws Exception {
    for (int iteration = 0; iteration < 100; iteration++) {
      byte[] source = new byte[random().nextInt(150)];
      for (int i = 0; i < source.length; i++) source[i] = (byte) random().nextInt(16);
      byte[] expected = new byte[4 * ((source.length + 7) >>> 3)];
      OptimizedScalarQuantizer.transposeHalfByte(source, expected);
      assertArrayEquals(expected, BlockOperations.transposeNibbles(null, source));
      if ((source.length & 1) == 0) {
        byte[] packed = BlockOperations.packNibbles(null, source);
        for (int i = 0; i < packed.length; i++) {
          assertEquals(source[i], (packed[i] & 255) >>> 4);
          assertEquals(source[i + packed.length], packed[i] & 15);
        }
      }
    }
    expectThrows(
        IllegalArgumentException.class, () -> BlockOperations.packNibbles(null, new byte[] {0}));
    expectThrows(
        IllegalArgumentException.class,
        () -> BlockOperations.transposeNibbles(null, new byte[] {16}));
  }

  public void testVectorAndQuantizedArithmeticBitForBit() throws Exception {
    for (int iteration = 0; iteration < 100; iteration++) {
      byte[] query = new byte[1 + random().nextInt(100)];
      byte[][] documents = new byte[3][query.length];
      random().nextBytes(query);
      for (byte[] document : documents) random().nextBytes(document);
      for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
        float[] scores = BlockOperations.byteScores(null, query, documents, similarity);
        for (int i = 0; i < scores.length; i++)
          assertEquals(
              Float.floatToRawIntBits(similarity.compare(query, documents[i])),
              Float.floatToRawIntBits(scores[i]));
      }
      for (int i = 0; i < query.length; i++) query[i] &= 127;
      for (byte[] document : documents)
        for (int i = 0; i < document.length; i++) document[i] &= 127;
      float[] corrections = {
        random().nextFloat() - .5f, random().nextFloat() - .5f, random().nextFloat() - .5f
      };
      float multiplier = random().nextFloat(), queryCorrection = random().nextFloat() - .5f;
      for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
        var oracle =
            ScalarQuantizedVectorSimilarity.fromVectorSimilarity(similarity, multiplier, (byte) 7);
        float[] actual =
            BlockOperations.quantizedRescore(
                null, query, documents, corrections, similarity, multiplier, queryCorrection);
        for (int i = 0; i < actual.length; i++)
          assertEquals(
              Float.floatToRawIntBits(
                  oracle.score(query, queryCorrection, documents[i], corrections[i])),
              Float.floatToRawIntBits(actual[i]));
      }
    }
  }

  public void testMaxSimEmptyAndExceptionalSemantics() {
    assertEquals(
        Float.floatToRawIntBits(Float.MIN_VALUE),
        Float.floatToRawIntBits(
            BlockOperations.maxSim(
                new float[][] {{1}, {2}}, new float[0][], VectorSimilarityFunction.EUCLIDEAN)));
    assertEquals(
        0f,
        BlockOperations.maxSim(
            new float[0][], new float[][] {{1}}, VectorSimilarityFunction.EUCLIDEAN),
        0f);
    float[][] query = {{1, 0}, {0, 1}}, document = {{1, 0}, {0, 1}};
    assertEquals(
        2f, BlockOperations.maxSim(query, document, VectorSimilarityFunction.DOT_PRODUCT), 0f);
    expectThrows(
        AssertionError.class,
        () ->
            BlockOperations.maxSim(
                new float[][] {{Float.NaN}},
                new float[][] {{1}},
                VectorSimilarityFunction.DOT_PRODUCT));
  }

  public void testOffUsesJavaWithoutNativeAllocation() throws Exception {
    AccelerationRuntime runtime =
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
    assertArrayEquals(
        new long[] {7, 10}, BlockOperations.decode(runtime, new byte[] {1, 2}, 0, 2, 8, 1, 3, 4));
    assertArrayEquals(new byte[] {0x12}, BlockOperations.packNibbles(runtime, new byte[] {1, 2}));
    assertEquals(0L, (long) runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
    assertEquals(1L, runtime.inventory().counters().get(Capability.DOCVALUES_DECODE).javaCalls());
  }

  private static RequestMemoryScope scope(MemoryLedger ledger, AtomicInteger live) {
    NativeAllocator allocator =
        bytes -> {
          live.incrementAndGet();
          return new NativeAllocator.Allocation() {
            private final ByteBuffer buffer = ByteBuffer.allocate(Math.toIntExact(bytes));
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
            public void close() {
              if (!closed) {
                closed = true;
                live.decrementAndGet();
              }
            }
          };
        };
    return new RequestMemoryScope(
        ledger, new MemoryLedger.Owner("fixtures", "standalone", 0, "fixtures"), allocator);
  }

  public void testStartupFixturesRejectCorruptionAndCloseOwner() throws Exception {
    for (Capability capability : Capability.values()) {
      AtomicInteger live = new AtomicInteger();
      MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(0, 1 << 20, 0), 1 << 20);
      List<KernelFixtures.Fixture> fixtures = KernelFixtures.fixtures(capability);
      AtomicInteger calls = new AtomicInteger();
      AccelerationProvider.Kernel exact =
          new AccelerationProvider.Kernel() {
            @Override
            public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
              return new AccelerationProvider.Estimate(true, 0, 0);
            }

            @Override
            public AccelerationProvider.Result execute(AccelerationProvider.Batch batch) {
              KernelFixtures.Fixture fixture = fixtures.get(calls.getAndIncrement());
              assertEquals(capability, batch.capability());
              batch.output().bytes().put(fixture.expected());
              return new AccelerationProvider.Result(
                  AccelerationProvider.Status.OK, fixture.operation().outputElements(), 0);
            }
          };
      assertEquals(
          !fixtures.isEmpty(), KernelFixtures.verify(capability, exact, () -> scope(ledger, live)));
      assertEquals(0, live.get());
      assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.NATIVE));
    }
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(0, 1024, 0), 1024);
    AtomicInteger live = new AtomicInteger();
    AccelerationProvider.Kernel corrupt =
        new AccelerationProvider.Kernel() {
          @Override
          public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
            return new AccelerationProvider.Estimate(true, 0, 0);
          }

          @Override
          public AccelerationProvider.Result execute(AccelerationProvider.Batch batch) {
            return new AccelerationProvider.Result(
                AccelerationProvider.Status.OK, batch.output().elements(), 0);
          }
        };
    assertFalse(KernelFixtures.verify(Capability.TEST_COPY, corrupt, () -> scope(ledger, live)));
    assertEquals(0, live.get());
    AccelerationProvider.Kernel failure =
        new AccelerationProvider.Kernel() {
          @Override
          public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
            return new AccelerationProvider.Estimate(true, 0, 0);
          }

          @Override
          public AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
              throws IOException {
            throw new IOException("fixture failure");
          }
        };
    expectThrows(
        IOException.class,
        () -> KernelFixtures.verify(Capability.TEST_COPY, failure, () -> scope(ledger, live)));
    assertEquals(0, live.get());
  }

  public void testRequiredFeatureMasks() throws Exception {
    assumeTrue("FFM feature helper requires JDK25", Runtime.version().feature() >= 25);
    var method =
        Class.forName("org.apache.lucene.internal.acceleration.FfmBridge")
            .getMethod("supportsRequiredFeatures", Set.class, long.class);
    for (long mask = 0; mask <= 3; mask++) {
      assertEquals(true, method.invoke(null, Set.of(), mask));
      assertEquals((mask & 1) != 0, method.invoke(null, Set.of("x86.avx2"), mask));
      assertEquals((mask & 1) != 0, method.invoke(null, Set.of("os.xsave-ymm"), mask));
      assertEquals((mask & 2) != 0, method.invoke(null, Set.of("x86.fma"), mask));
      assertEquals(
          mask == 3, method.invoke(null, Set.of("x86.avx2", "os.xsave-ymm", "x86.fma"), mask));
      assertEquals(false, method.invoke(null, Set.of("x86.avx512"), mask));
    }
    assertEquals(false, method.invoke(null, Set.of(), 4L));
  }

  public void testAdmissionTerminationDoesNotReplayJavaOrPublish() throws Exception {
    try (var runtime =
        AccelerationRuntime.forQualification(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                Set.of(),
                Set.of(),
                1))) {
      assertTrue(runtime.supports(Capability.TEST_COPY));
      AtomicInteger published = new AtomicInteger();
      for (var reason :
          new RequestMemoryScope.Failure[] {
            RequestMemoryScope.Failure.CANCELLED,
            RequestMemoryScope.Failure.CLOSED,
            RequestMemoryScope.Failure.SHUTDOWN,
            RequestMemoryScope.Failure.DEADLINE
          }) {
        var source =
            new BlockOperations.Source(
                1,
                AccelerationProvider.Encoding.I64,
                b -> {
                  throw new RequestMemoryScope.AdmissionException(reason);
                });
        var operation =
            new BlockOperations.Operation(
                Capability.TEST_COPY, 1, BlockOperations.parameters(1), List.of(source), 1, 8);
        IOException error =
            expectThrows(
                IOException.class,
                () ->
                    BlockOperations.nativeCall(
                        runtime, operation, b -> published.incrementAndGet()));
        assertSame(reason, ((RequestMemoryScope.AdmissionException) error.getCause()).reason());
      }
      assertEquals(0, published.get());
      assertEquals(0L, runtime.inventory().counters().get(Capability.TEST_COPY).nativeCalls());
      assertEquals(0L, runtime.inventory().counters().get(Capability.TEST_COPY).javaCalls());
      assertEquals(0L, runtime.inventory().execution().activeScopes());
      var source =
          new BlockOperations.Source(
              1,
              AccelerationProvider.Encoding.I64,
              b -> {
                throw new RequestMemoryScope.AdmissionException(
                    RequestMemoryScope.Failure.REQUEST_LIMIT);
              });
      var operation =
          new BlockOperations.Operation(
              Capability.TEST_COPY, 1, BlockOperations.parameters(1), List.of(source), 1, 8);
      assertFalse(BlockOperations.nativeCall(runtime, operation, b -> published.incrementAndGet()));
      assertEquals(0, published.get());
      assertEquals(
          1L,
          runtime
              .inventory()
              .counters()
              .get(Capability.TEST_COPY)
              .fallbacks()
              .get(AccelerationRuntime.Reason.BUDGET_DENIED)
              .longValue());
    }
  }

  public void testFloatingStatisticsPopulatedPrefixAndSelection() throws Exception {
    var initial =
        new BlockOperations.FloatingStats(0, 0, Double.MAX_VALUE, -Double.MAX_VALUE, 0, 0);
    double[] reusable = {1, 2, 3, Double.NaN, Double.POSITIVE_INFINITY};
    var expected = BlockOperations.floatingStats(null, new double[] {1, 2, 3}, null, initial);
    assertEquals(expected, BlockOperations.floatingStats(null, reusable, 3, null, initial));
    var selected = BlockOperations.floatingStats(null, reusable, 3, new long[] {5}, initial);
    assertEquals(2, selected.count());
    assertEquals(4, selected.sum(), 0);
    assertEquals(1, selected.min(), 0);
    assertEquals(3, selected.max(), 0);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> BlockOperations.floatingStats(null, reusable, 6, null, initial));
  }

  public void testReferenceFootprintIncludesZeroWordSetHeaders() throws Exception {
    ByteBuffer parameters = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    long sets = (16L << 20) + 1;
    parameters.putLong(8, sets);
    var batch =
        new AccelerationProvider.Batch(
            Capability.MULTI_INTERSECT,
            List.of(
                new AccelerationProvider.Input(
                    ByteBuffer.allocate(0), 0, AccelerationProvider.Encoding.I64)),
            new AccelerationProvider.Output(ByteBuffer.allocate(0), 0),
            parameters,
            ByteBuffer.allocate(0),
            ByteBuffer.allocate(256));
    assertTrue(JavaKernelOracle.referenceMemoryBytes(batch) >= Math.multiplyExact(sets, 64));
    assertEquals(AccelerationProvider.Status.UNSUPPORTED, JavaKernelOracle.execute(batch).status());
    parameters.putLong(8, Long.MAX_VALUE);
    expectThrows(ArithmeticException.class, () -> JavaKernelOracle.referenceMemoryBytes(batch));
  }

  public void testFloatDomainPreservesJavaFallbackAndIgnoresPadding() throws Exception {
    for (Capability capability : List.of(Capability.VECTOR_F32, Capability.MAXSIM)) {
      for (float value :
          new float[] {Float.NaN, Float.POSITIVE_INFINITY, Float.MAX_VALUE, 1e20f, 0f, 1e-30f}) {
        ByteBuffer q = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        q.putFloat(0, value).putFloat(4, value).putFloat(8, value);
        ByteBuffer rows = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        rows.putFloat(0, 1).putFloat(4, 2).putFloat(8, 3);
        ByteBuffer parameters = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        parameters
            .putLong(0, 1)
            .putLong(8, 3)
            .putLong(16, capability == Capability.VECTOR_F32 ? 3 : 1)
            .putLong(24, 2);
        List<AccelerationProvider.Input> inputs = new java.util.ArrayList<>();
        inputs.add(new AccelerationProvider.Input(q, 3, AccelerationProvider.Encoding.F32));
        inputs.add(new AccelerationProvider.Input(rows, 3, AccelerationProvider.Encoding.F32));
        if (capability == Capability.MAXSIM) {
          ByteBuffer offsets = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
          offsets.putLong(8, 1);
          inputs.add(new AccelerationProvider.Input(offsets, 2, AccelerationProvider.Encoding.I64));
          parameters.putLong(32, 1).putLong(40, 3);
        }
        var batch =
            new AccelerationProvider.Batch(
                capability,
                inputs,
                new AccelerationProvider.Output(ByteBuffer.allocate(4), 1),
                parameters,
                ByteBuffer.allocate(0),
                ByteBuffer.allocate(256));
        JavaKernelOracle.validate(batch);
        assertFalse(JavaKernelOracle.supportsFloatDomain(batch));
        assertEquals(
            value == value ? Float.floatToRawIntBits(value) : Float.floatToRawIntBits(Float.NaN),
            q.getInt(0));
      }
    }
    ByteBuffer q = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(0, 1);
    ByteBuffer rows = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
    rows.putFloat(0, 2).putFloat(4, Float.NaN).putFloat(8, Float.POSITIVE_INFINITY);
    ByteBuffer parameters = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    parameters.putLong(0, 1).putLong(8, 1).putLong(16, 3).putLong(24, 2);
    var batch =
        new AccelerationProvider.Batch(
            Capability.VECTOR_F32,
            List.of(
                new AccelerationProvider.Input(q, 1, AccelerationProvider.Encoding.F32),
                new AccelerationProvider.Input(rows, 3, AccelerationProvider.Encoding.F32)),
            new AccelerationProvider.Output(ByteBuffer.allocate(4), 1),
            parameters,
            ByteBuffer.allocate(0),
            ByteBuffer.allocate(256));
    JavaKernelOracle.validate(batch);
    assertTrue(JavaKernelOracle.supportsFloatDomain(batch));
  }
}
