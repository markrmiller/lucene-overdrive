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

import java.util.Arrays;
import java.util.Set;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/** Index byte and corrective metadata parity for the complete optimized quantizer. */
public class TestAcceleratedQuantization extends LuceneTestCase {
  public void testNativeOptimizedIntervalsCorrectionsAndInputMutation() throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                Set.of(),
                Set.of(),
                1));
    assertTrue(
        runtime.inventory().rejections().toString(),
        runtime.supports(Capability.OPTIMIZED_QUANTIZE));
    for (int dimensions : new int[] {1, 7, 8, 9, 33, 65}) {
      for (var metric : VectorSimilarityFunction.values()) {
        for (byte bits : new byte[] {1, 2, 4, 7, 8}) {
          float[] vector = new float[dimensions], centroid = new float[dimensions];
          for (int i = 0; i < dimensions; i++) {
            vector[i] = (random().nextInt(512) - 256) / 128f;
            centroid[i] = (random().nextInt(128) - 64) / 128f;
          }
          if (vector[0] == 0) vector[0] = 0.5f;
          if (centroid[0] == 0) centroid[0] = 0.5f;
          if (metric == VectorSimilarityFunction.COSINE) {
            VectorUtil.l2normalize(vector);
            VectorUtil.l2normalize(centroid);
          }
          var quantizer = new OptimizedScalarQuantizer(metric, 0.1f, 5);
          float[] expectedCentered = vector.clone(), actualCentered = vector.clone();
          byte[] expectedBytes = new byte[dimensions + 7], actualBytes = new byte[dimensions + 7];
          Arrays.fill(expectedBytes, (byte) 0x55);
          Arrays.fill(actualBytes, (byte) 0x55);
          var expected =
              quantizer.scalarQuantizeJava(expectedCentered, expectedBytes, bits, centroid);
          var actual =
              quantizer.scalarQuantize(runtime, actualCentered, actualBytes, bits, centroid);
          assertArrayEquals(expectedBytes, actualBytes);
          for (int i = 0; i < dimensions; i++)
            assertEquals(
                Float.floatToRawIntBits(expectedCentered[i]),
                Float.floatToRawIntBits(actualCentered[i]));
          assertEquals(
              Float.floatToRawIntBits(expected.lowerInterval()),
              Float.floatToRawIntBits(actual.lowerInterval()));
          assertEquals(
              Float.floatToRawIntBits(expected.upperInterval()),
              Float.floatToRawIntBits(actual.upperInterval()));
          assertEquals(
              Float.floatToRawIntBits(expected.additionalCorrection()),
              Float.floatToRawIntBits(actual.additionalCorrection()));
          assertEquals(expected.quantizedComponentSum(), actual.quantizedComponentSum());
        }
      }
    }
    assertTrue(runtime.inventory().counters().get(Capability.OPTIMIZED_QUANTIZE).nativeCalls() > 0);
  }

  public void testInvalidQuantizerInputsRetainAssertionsBeforeNativePublication() throws Exception {
    try (var runtime =
        AccelerationRuntime.forQualification(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                Set.of(),
                Set.of(),
                1))) {
      assertTrue(runtime.supports(Capability.OPTIMIZED_QUANTIZE));
      var cosine = new OptimizedScalarQuantizer(VectorSimilarityFunction.COSINE);
      var euclidean = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
      for (int invalid = 0; invalid < 3; invalid++) {
        float[] vector = invalid == 0 ? new float[] {2f, 0f} : new float[] {1f, 0f};
        float[] centroid = invalid == 1 ? new float[] {2f, 0f} : new float[] {1f, 0f};
        byte[] destination = new byte[invalid == 2 ? 1 : 2];
        Arrays.fill(destination, (byte) 0x55);
        float[] originalVector = vector.clone();
        float[] originalCentroid = centroid.clone();
        byte[] originalDestination = destination.clone();
        var quantizer = invalid == 2 ? euclidean : cosine;
        expectThrows(
            AssertionError.class,
            () -> quantizer.scalarQuantize(runtime, vector, destination, (byte) 4, centroid));
        assertArrayEquals(originalDestination, destination);
        assertArrayEquals(originalVector, vector, 0f);
        assertArrayEquals(originalCentroid, centroid, 0f);
      }
      assertEquals(
          0L, runtime.inventory().counters().get(Capability.OPTIMIZED_QUANTIZE).nativeCalls());
    }
  }

  public void testPackingHooksRetainPaddedStripeOffsetsAndUntouchedBytes() {
    AccelerationRuntime runtime = AccelerationRuntime.system();
    long callsBefore =
        runtime.inventory().counters().get(Capability.QUANTIZE_TRANSPOSE).nativeCalls();
    for (int bits : new int[] {1, 2, 4}) {
      for (int dimensions : new int[] {1, 7, 9, 17, 65}) {
        int blocks = (dimensions + 7) / 8;
        for (int extraBytes : new int[] {3, 5}) {
          byte[] input = new byte[dimensions];
          for (int i = 0; i < input.length; i++) input[i] = (byte) random().nextInt(1 << bits);
          byte[] originalInput = input.clone();
          byte[] actual = new byte[blocks * bits + extraBytes];
          Arrays.fill(actual, (byte) 0x5A);
          byte[] expected = actual.clone();
          for (int plane = 0; plane < bits; plane++) {
            int stripeOffset = bits == 4 ? plane * actual.length / 4 : plane * actual.length / bits;
            for (int block = 0; block < blocks; block++) {
              int packedByte = 0;
              for (int lane = 0; lane < 8 && block * 8 + lane < dimensions; lane++)
                packedByte |= ((input[block * 8 + lane] >>> plane) & 1) << (7 - lane);
              expected[stripeOffset + block] = (byte) packedByte;
            }
          }
          switch (bits) {
            case 1 -> OptimizedScalarQuantizer.packAsBinary(input, actual);
            case 2 -> OptimizedScalarQuantizer.transposeDibit(input, actual);
            case 4 -> OptimizedScalarQuantizer.transposeHalfByte(input, actual);
            default -> throw new AssertionError(bits);
          }
          assertArrayEquals(expected, actual);
          assertArrayEquals(originalInput, input);
        }
      }
    }
    assertEquals(
        callsBefore,
        runtime.inventory().counters().get(Capability.QUANTIZE_TRANSPOSE).nativeCalls());
  }

  public void testAllNativePackingProfilesHaveZeroTailBits() throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                Set.of(),
                Set.of(),
                1));
    for (int bits : new int[] {1, 2, 4}) {
      for (int dimensions : new int[] {1, 7, 8, 9, 65}) {
        byte[] input = new byte[dimensions];
        for (int i = 0; i < input.length; i++) input[i] = (byte) random().nextInt(1 << bits);
        byte[] actual = BlockOperations.transposeBits(runtime, input, bits);
        int blocks = (dimensions + 7) / 8;
        byte[] expected = new byte[blocks * bits];
        for (int plane = 0; plane < bits; plane++)
          for (int i = 0; i < dimensions; i++)
            expected[plane * blocks + (i >>> 3)] |=
                (byte) (((input[i] >>> plane) & 1) << (7 - (i & 7)));
        assertArrayEquals(expected, actual);
      }
    }
  }
}
