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
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestOwnedByteBatchScorer extends LuceneTestCase {
  private AccelerationRuntime runtime(Path pack, long bytes) throws IOException {
    return new AccelerationRuntime(
        new AccelerationConfig(
            AccelerationConfig.Mode.AUTO,
            pack,
            Set.of(),
            Set.of(),
            bytes,
            bytes,
            1,
            true,
            AccelerationConfig.StoragePolicy.CURRENT));
  }

  public void testJavaOnlyLifetimeAndQuerySnapshot() throws Exception {
    exercise(null, 16L << 20, false, 384, VectorSimilarityFunction.DOT_PRODUCT);
  }

  public void testOptionalPackMeasuredShapeAndFallbacks() throws Exception {
    String configured = System.getProperty("tests.acceleration.pack");
    assumeTrue("optional native pack not supplied", configured != null);
    var pack = Path.of(configured);
    for (var similarity : VectorSimilarityFunction.values())
      exercise(pack, 16L << 20, true, 384, similarity);
    exercise(pack, 16L << 20, true, 64, VectorSimilarityFunction.DOT_PRODUCT);
    exercise(pack, 16L << 20, true, 128, VectorSimilarityFunction.DOT_PRODUCT);
    exercise(pack, 16L << 20, false, 384, VectorSimilarityFunction.DOT_PRODUCT);
    exercise(pack, 16L << 10, true, 384, VectorSimilarityFunction.DOT_PRODUCT);
  }

  @SuppressWarnings("try")
  private void exercise(
      Path pack, long budget, boolean mapped, int dimensions, VectorSimilarityFunction similarity)
      throws Exception {
    byte[][] vectors = new byte[1024][dimensions];
    byte[] query = new byte[dimensions];
    random().nextBytes(query);
    byte[] snapshot = query.clone();
    for (byte[] row : vectors) random().nextBytes(row);
    try (Directory directory =
            mapped ? FSDirectory.open(createTempDir()) : new ByteBuffersDirectory();
        AccelerationRuntime runtime = runtime(pack, budget)) {
      try (IndexWriter writer =
          new IndexWriter(directory, new IndexWriterConfig().setCodec(new Lucene104Codec()))) {
        for (byte[] vector : vectors) {
          Document doc = new Document();
          doc.add(new KnnByteVectorField("v", vector, similarity));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        var leaf = reader.leaves().get(0).reader();
        int refs = leaf.getRefCount();
        long before = runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls();
        // Measured mapped DOT_PRODUCT shapes: 64 or 384 dimensions with 64 or 512 contiguous rows.
        boolean enrolled =
            mapped
                && (dimensions == 64 || dimensions == 384)
                && similarity == VectorSimilarityFunction.DOT_PRODUCT
                && budget >= 16L << 20
                && runtime.supports(Capability.VECTOR_I8);
        try (var scorer = OwnedByteBatchScorer.create(runtime, leaf, "v", query)) {
          Arrays.fill(query, (byte) 0);
          assertEquals(similarity.compare(snapshot, vectors[17]), scorer.score(17), 0);
          assertEquals(
              similarity.compare(snapshot, vectors[18]), scorer.forGraphTraversal().score(18), 0);
          for (int count : new int[] {0, 1, 64, 511, 512, 513}) {
            int[] nodes = new int[count];
            float[] scores = new float[count + 1];
            Arrays.fill(scores, -42);
            for (int i = 0; i < count; i++) nodes[i] = 128 + i;
            float maximum = scorer.bulkScore(nodes, scores, count);
            float expectedMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < count; i++) {
              float expected = similarity.compare(snapshot, vectors[nodes[i]]);
              assertEquals(
                  "metric=" + similarity + " count=" + count + " row=" + i + " mapped=" + mapped,
                  Float.floatToRawIntBits(expected),
                  Float.floatToRawIntBits(scores[i]));
              expectedMax = Math.max(expectedMax, expected);
            }
            assertEquals(expectedMax, maximum, 0);
            assertEquals(-42f, scores[count], 0);
            assertEquals(refs, leaf.getRefCount());
            long calls = runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls();
            if (count != 64 && count != 512) assertEquals(before, calls);
            else if (enrolled) assertEquals(before + 1, calls);
            before = calls;
          }
          int[] scattered = new int[512];
          float[] scores = new float[512];
          for (int i = 0; i < scattered.length; i++) scattered[i] = i % 7;
          scorer.bulkScore(scattered, scores, scattered.length);
          for (int i = 0; i < scattered.length; i++)
            assertEquals(similarity.compare(snapshot, vectors[scattered[i]]), scores[i], 0);
          assertEquals(
              before, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
          int[] bad = scattered.clone();
          bad[511] = 1024;
          Arrays.fill(scores, -42);
          expectThrows(IndexOutOfBoundsException.class, () -> scorer.bulkScore(bad, scores, 512));
          for (float score : scores) assertEquals(-42f, score, 0);
          Thread foreign =
              new Thread(() -> expectThrows(IllegalStateException.class, () -> scorer.score(0)));
          foreign.start();
          foreign.join();
          var context =
              new AccelerationRuntime.TaskContext(
                  new MemoryLedger.Owner("test", "owner", 0, "changed"), 0);
          try (var binding = runtime.enterTask(context)) {
            expectThrows(IOException.class, () -> scorer.bulkScore(scattered, scores, 512));
          }
          assertEquals(enrolled ? 2L : 0L, before);
        }
        assertEquals(refs, leaf.getRefCount());
      }
      assertEquals(0L, runtime.inventory().execution().activeScopes());
      for (long charged : runtime.ledger().snapshot().charged().values()) assertEquals(0L, charged);
    }
  }
}
