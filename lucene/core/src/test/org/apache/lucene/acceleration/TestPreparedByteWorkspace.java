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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.NativeAllocator;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Physical workspace cost and ownership, including padded odd-sized ABI spans. */
public class TestPreparedByteWorkspace extends LuceneTestCase {
  private static final byte[] QUERY = {7, -11, 3};
  private static final byte[][] ROWS = {
    {1, 2, 3},
    {-5, 0, 2},
    {4, 1, -3},
    {7, 0, 1},
    {-1, -2, -3},
    {Byte.MIN_VALUE, 1, Byte.MAX_VALUE},
    {0, 3, 2}
  };

  private static AccelerationRuntime runtime(long bytes) throws IOException {
    return AccelerationRuntime.forQualification(
        new AccelerationConfig(
            AccelerationConfig.Mode.OFF,
            null,
            Set.of(),
            Set.of(),
            bytes,
            bytes,
            1,
            true,
            AccelerationConfig.StoragePolicy.CURRENT));
  }

  private static RequestMemoryScope scope(AccelerationRuntime runtime, TrackedAllocator allocator) {
    return RequestMemoryScope.threadConfined(
        runtime.ledger(), new MemoryLedger.Owner("test", "workspace", 0, "workspace"), allocator);
  }

  private static AccelerationProvider.Kernel kernel(long scratch, boolean inspect) {
    return new AccelerationProvider.Kernel() {
      @Override
      public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
        return new AccelerationProvider.Estimate(true, scratch, 0);
      }

      @Override
      public AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
          throws IOException {
        if (inspect) {
          List<ByteBuffer> views =
              List.of(
                  batch.inputs().get(0).bytes(),
                  batch.output().bytes(),
                  batch.parameters(),
                  batch.scratch(),
                  batch.metadata());
          int[] lengths = {3, ROWS.length * Float.BYTES, 64, 13, 256};
          assertTrue(views.getFirst().isReadOnly());
          for (int i = 0; i < views.size(); i++) {
            assertEquals(lengths[i], views.get(i).remaining());
            if (i > 0) {
              assertSame(views.get(1).array(), views.get(i).array());
              assertEquals(
                  "writable ABI spans have the expected disjoint workspace offsets",
                  new int[] {0, 8, 40, 104, 120}[i],
                  views.get(i).arrayOffset() + views.get(i).position());
            }
          }
          byte[] parameters = bytes(batch.parameters());
          fill(batch.scratch(), (byte) 0x5a);
          fill(batch.metadata(), (byte) 0xa5);
          fill(batch.output().bytes(), (byte) 0xcc);
          assertArrayEquals(QUERY, bytes(batch.inputs().get(0).bytes()));
          assertArrayEquals(parameters, bytes(batch.parameters()));
          var result = JavaKernelOracle.execute(batch);
          assertArrayEquals(parameters, bytes(batch.parameters()));
          assertArrayEquals(QUERY, bytes(batch.inputs().get(0).bytes()));
          for (int i = 0; i < 13; i++) assertEquals((byte) 0x5a, batch.scratch().get(i));
          for (int i = 0; i < 256; i++) assertEquals((byte) 0xa5, batch.metadata().get(i));
          return result;
        }
        return JavaKernelOracle.execute(batch);
      }
    };
  }

  private static byte[] bytes(ByteBuffer buffer) {
    byte[] bytes = new byte[buffer.remaining()];
    buffer.duplicate().get(bytes);
    return bytes;
  }

  private static void fill(ByteBuffer buffer, byte value) {
    for (int i = buffer.position(); i < buffer.limit(); i++) buffer.put(i, value);
  }

  public void testOneAlignedWorkspaceAndSeparateDiagnosticRows() throws Exception {
    var allocator = new TrackedAllocator();
    try (var runtime = runtime(8192)) {
      byte[] query = QUERY.clone();
      try (var frame =
          new PreparedByteScoring(
              runtime,
              kernel(13, true),
              scope(runtime, allocator),
              query,
              ROWS.length,
              VectorSimilarityFunction.DOT_PRODUCT)) {
        assertEquals(
            "one owned workspace plus separate diagnostic row staging",
            2,
            allocator.allocations.size());
        assertEquals(
            "all slot padding and rounded diagnostic rows are admitted",
            400,
            runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
        Arrays.fill(query, (byte) 0); // The prepared snapshot is independent of caller mutation.
        float[] scores = new float[ROWS.length + 2];
        Arrays.fill(scores, 42);
        float maximum = frame.score(ROWS, 0, ROWS.length, scores, 1);
        float expectedMaximum = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < ROWS.length; i++) {
          float expected = VectorSimilarityFunction.DOT_PRODUCT.compare(QUERY, ROWS[i]);
          assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(scores[i + 1]));
          expectedMaximum = Math.max(expectedMaximum, expected);
        }
        assertEquals(Float.floatToRawIntBits(expectedMaximum), Float.floatToRawIntBits(maximum));
        assertEquals(42, scores[0], 0);
        assertEquals(42, scores[scores.length - 1], 0);
        assertEquals(2, allocator.allocations.size());
      }
      assertDrained(runtime, allocator);
    }
  }

  public void testStagingFailureClosesTheAlreadyAllocatedWorkspace() throws Exception {
    var allocator = new TrackedAllocator();
    allocator.failAllocationAt = 2;
    try (var runtime = runtime(8192)) {
      var failure =
          expectThrows(
              IllegalStateException.class,
              () ->
                  new PreparedByteScoring(
                      runtime,
                      kernel(13, false),
                      scope(runtime, allocator),
                      QUERY,
                      ROWS.length,
                      VectorSimilarityFunction.DOT_PRODUCT));
      assertEquals("injected allocation failure", failure.getMessage());
      assertEquals(1, allocator.allocations.size());
      assertDrained(runtime, allocator);
    }
  }

  public void testPaddedCombinedPeakMustFitBeforeDiagnosticStaging() throws Exception {
    var allocator = new TrackedAllocator();
    try (var runtime = runtime(399)) {
      expectThrows(
          RequestMemoryScope.AdmissionException.class,
          () ->
              new PreparedByteScoring(
                  runtime,
                  kernel(13, false),
                  scope(runtime, allocator),
                  QUERY,
                  ROWS.length,
                  VectorSimilarityFunction.DOT_PRODUCT));
      assertEquals(
          "only the admitted workspace is physically allocated", 1, allocator.allocations.size());
      assertDrained(runtime, allocator);
    }
  }

  public void testAlignmentPaddingIsIncludedInThe128MiBConstructionCap() throws Exception {
    var allocator = new TrackedAllocator();
    allocator.failAllocationAt = 1;
    try (var runtime = runtime(256L << 20)) {
      long scratch =
          (128L << 20)
              - (QUERY.length + ROWS.length * QUERY.length + 320L + ROWS.length * Float.BYTES);
      expectThrows(
          IOException.class,
          () ->
              new PreparedByteScoring(
                  runtime,
                  kernel(scratch, false),
                  scope(runtime, allocator),
                  QUERY,
                  ROWS.length,
                  VectorSimilarityFunction.DOT_PRODUCT));
      assertEquals(0, allocator.attempts);
      assertDrained(runtime, allocator);
    }
  }

  public void testOverflowAndNegativeEstimatesFailBeforeAnyAllocation() throws Exception {
    var allocator = new TrackedAllocator();
    try (var runtime = runtime(8192)) {
      expectThrows(
          ArithmeticException.class,
          () ->
              new PreparedByteScoring(
                  runtime,
                  kernel(Long.MAX_VALUE, false),
                  scope(runtime, allocator),
                  QUERY,
                  ROWS.length,
                  VectorSimilarityFunction.DOT_PRODUCT));
      expectThrows(
          IllegalArgumentException.class,
          () ->
              new PreparedByteScoring(
                  runtime,
                  kernel(-1, false),
                  scope(runtime, allocator),
                  QUERY,
                  ROWS.length,
                  VectorSimilarityFunction.DOT_PRODUCT));
      assertEquals(0, allocator.attempts);
      assertDrained(runtime, allocator);
    }
  }

  public void testFailedWorkspaceCloseRetainsItsChargeUntilExplicitRecovery() throws Exception {
    var allocator = new TrackedAllocator();
    try (var runtime = runtime(8192)) {
      var frame =
          new PreparedByteScoring(
              runtime,
              kernel(13, false),
              scope(runtime, allocator),
              QUERY,
              ROWS.length,
              VectorSimilarityFunction.DOT_PRODUCT);
      allocator.failCloseFirst = true;
      expectThrows(IllegalStateException.class, frame::close);
      long held = allocator.allocations.get(0).capacity();
      assertEquals(held, runtime.ledger().snapshot().failedReclamations());
      assertEquals(
          held, runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
      assertTrue(allocator.allocations.get(0).isAlive());
      frame.close(); // Repeated retirement does not silently retry failed physical reclamation.
      assertTrue(allocator.allocations.get(0).isAlive());
      allocator.failCloseFirst = false;
      frame.retryFailedReclamations();
      assertDrained(runtime, allocator);
      expectThrows(IllegalStateException.class, () -> frame.score(ROWS, 0, 1, new float[1], 0));
    }
  }

  private static void assertDrained(AccelerationRuntime runtime, TrackedAllocator allocator) {
    assertEquals(
        0L, runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
    assertEquals(
        0L, runtime.ledger().snapshot().reserved().get(MemoryLedger.Kind.NATIVE).longValue());
    assertEquals(0L, runtime.ledger().snapshot().failedReclamations());
    for (var allocation : allocator.allocations) assertFalse(allocation.isAlive());
  }

  private static final class TrackedAllocator implements NativeAllocator {
    final List<Allocation> allocations = new ArrayList<>();
    int attempts;
    int failAllocationAt;
    boolean failCloseFirst;

    @Override
    public long allocationSize(long bytes) {
      return Math.addExact(Math.max(1L, bytes), 7) & ~7L;
    }

    @Override
    public Allocation allocate(long bytes) {
      attempts++;
      if (attempts == failAllocationAt)
        throw new IllegalStateException("injected allocation failure");
      int index = allocations.size();
      var allocation =
          new Allocation() {
            final ByteBuffer buffer =
                ByteBuffer.allocate(Math.toIntExact(allocationSize(bytes)))
                    .order(ByteOrder.LITTLE_ENDIAN);
            boolean closed;

            @Override
            public ByteBuffer buffer() {
              return buffer;
            }

            @Override
            public long capacity() {
              return buffer.capacity();
            }

            @Override
            public boolean isAlive() {
              return !closed;
            }

            @Override
            public void close() {
              if (index == 0 && failCloseFirst)
                throw new IllegalStateException("injected close failure");
              closed = true;
            }
          };
      allocations.add(allocation);
      return allocation;
    }
  }
}
