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
package org.apache.lucene.codecs.lucene104;

import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.junit.BeforeClass;

/** With no pack, quantization, packing and vector flush ignore interrupts and task deadlines. */
public class TestStockQuantizationWithoutPack extends LuceneTestCase {

  @BeforeClass
  public static void requireNoPack() {
    AccelerationRuntime runtime = AccelerationRuntime.system();
    for (Capability capability :
        new Capability[] {
          Capability.OPTIMIZED_QUANTIZE,
          Capability.QUANTIZE_TRANSPOSE,
          Capability.QUANTIZE_BINARY,
          Capability.QUANTIZE_DIBIT,
          Capability.QUANTIZE_PACK,
          Capability.VECTOR_STATISTICS
        }) {
      assumeFalse("a pack registers " + capability, runtime.supports(capability));
    }
  }

  private static AccelerationRuntime.TaskContext expiredTask() {
    return new AccelerationRuntime.TaskContext(
        new MemoryLedger.Owner("stock", "core", 0, "stock"),
        System.nanoTime() - TimeUnit.SECONDS.toNanos(1));
  }

  private static byte[][] quantizeAndPack() {
    OptimizedScalarQuantizer quantizer =
        new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
    float[] centroid = new float[16];
    float[] vector = new float[16];
    byte[] nibbles = new byte[16];
    byte[] bits = new byte[16];
    byte[] dibits = new byte[16];
    for (int i = 0; i < 16; i++) {
      vector[i] = i - 8;
      nibbles[i] = (byte) i;
      bits[i] = (byte) (i & 1);
      dibits[i] = (byte) (i & 3);
    }
    byte[] quantized = new byte[16];
    quantizer.scalarQuantize(vector.clone(), quantized, (byte) 4, centroid);
    byte[][] multi = {new byte[16], new byte[16]};
    quantizer.multiScalarQuantize(vector.clone(), multi, new byte[] {1, 4}, centroid);
    byte[] transposed = new byte[8];
    OptimizedScalarQuantizer.transposeHalfByte(nibbles, transposed);
    byte[] binary = new byte[2];
    OptimizedScalarQuantizer.packAsBinary(bits, binary);
    byte[] dibit = new byte[4];
    OptimizedScalarQuantizer.transposeDibit(dibits, dibit);
    byte[] packed = new byte[8];
    OffHeapScalarQuantizedVectorValues.packNibbles(nibbles, packed);
    return new byte[][] {quantized, multi[0], multi[1], transposed, binary, dibit, packed};
  }

  public void testInterruptedQuantizationRunsLikeStock() {
    byte[][] expected = quantizeAndPack();
    Thread.currentThread().interrupt();
    try {
      byte[][] actual = quantizeAndPack();
      assertTrue(Thread.currentThread().isInterrupted());
      for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i]);
    } finally {
      Thread.interrupted();
    }
  }

  @SuppressWarnings("try")
  public void testQuantizationIgnoresExpiredTaskDeadline() throws Exception {
    byte[][] expected = quantizeAndPack();
    try (var binding = AccelerationRuntime.system().enterTask(expiredTask())) {
      byte[][] actual = quantizeAndPack();
      for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i]);
    }
  }

  @SuppressWarnings("try")
  public void testVectorFlushIgnoresExpiredTaskDeadline() throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(TestUtil.alwaysKnnVectorsFormat(new Lucene104ScalarQuantizedVectorsFormat()))
            .setMergePolicy(NoMergePolicy.INSTANCE);
    try (Directory directory = new ByteBuffersDirectory();
        IndexWriter writer = new IndexWriter(directory, config)) {
      try (var binding = AccelerationRuntime.system().enterTask(expiredTask())) {
        for (int i = 0; i < 10; i++) {
          Document document = new Document();
          document.add(
              new KnnFloatVectorField(
                  "vector", new float[] {i, 1, 2, 3}, VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(document);
        }
        writer.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(10, reader.leaves().get(0).reader().getFloatVectorValues("vector").size());
      }
    }
  }
}
