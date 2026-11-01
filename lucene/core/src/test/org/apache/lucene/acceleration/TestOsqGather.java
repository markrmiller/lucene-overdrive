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

import java.util.Set;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;

public class TestOsqGather extends LuceneTestCase {

  /** Bulk scores from mapped rows equal the Java scorer bit for bit for every encoding/metric. */
  public void testMappedGatherMatchesJava() throws Exception {
    try (var runtime =
        AccelerationRuntime.forQualification(
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                Set.of(),
                Set.of(),
                16L << 20,
                8L << 20,
                1,
                true,
                AccelerationConfig.StoragePolicy.CURRENT))) {
      assertTrue(runtime.supports(Capability.OSQ_GATHER));
      for (var encoding : ScalarEncoding.values())
        for (var metric : VectorSimilarityFunction.values())
          check(runtime, encoding, metric, random().nextBoolean() ? 384 : 1 + random().nextInt(70));
      assertTrue(runtime.inventory().counters().get(Capability.OSQ_GATHER).nativeCalls() > 0);
      assertEquals(
          0L,
          runtime.inventory().counters().get(Capability.OSQ_GATHER).fallbacks().values().stream()
              .mapToLong(Long::longValue)
              .sum());
    }
  }

  public void testProductionDefaultDispatchesQualified32RowGraphScorer() throws Exception {
    var pack = TestAccelerationRuntime.nativePack();
    var policy = TrustedPathFixtures.file("osq-functional-policy");
    java.nio.file.Files.writeString(
        policy,
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.OSQ_GATHER.id
            + "\t384\t32\t32\t1\t0\tmapped\t120960\tunit-default-dispatch\n");
    java.nio.file.Files.setPosixFilePermissions(
        policy, java.nio.file.attribute.PosixFilePermissions.fromString("rw-------"));
    var changes =
        java.util.Map.of(
            "lucene.acceleration.native.packPath", pack.toString(),
            "lucene.acceleration.native.qualificationPath", policy.toString(),
            "lucene.acceleration.experimental.enabled", "true",
            "lucene.acceleration.memory.nativeBytes", "16777216",
            "lucene.acceleration.memory.requestBytes", "8388608");
    var previous = new java.util.HashMap<String, String>();
    for (var entry : changes.entrySet()) {
      previous.put(entry.getKey(), System.getProperty(entry.getKey()));
      System.setProperty(entry.getKey(), entry.getValue());
    }
    String minimum = System.getProperty("lucene.acceleration.native.minimumBatch");
    System.clearProperty("lucene.acceleration.native.minimumBatch");
    try (var runtime = new AccelerationRuntime(AccelerationConfig.fromSystemProperties())) {
      assertEquals(1, runtime.config().minimumBatch());
      assertFalse(runtime.eligible(Capability.OSQ_GATHER, 31));
      check(runtime, ScalarEncoding.UNSIGNED_BYTE, VectorSimilarityFunction.DOT_PRODUCT, 384, true);
      assertTrue(runtime.inventory().counters().get(Capability.OSQ_GATHER).nativeCalls() > 0);
    } finally {
      for (var entry : previous.entrySet()) {
        if (entry.getValue() == null) System.clearProperty(entry.getKey());
        else System.setProperty(entry.getKey(), entry.getValue());
      }
      if (minimum == null) System.clearProperty("lucene.acceleration.native.minimumBatch");
      else System.setProperty("lucene.acceleration.native.minimumBatch", minimum);
    }
  }

  private void check(
      AccelerationRuntime runtime,
      ScalarEncoding encoding,
      VectorSimilarityFunction metric,
      int dimensions)
      throws Exception {
    check(runtime, encoding, metric, dimensions, false);
  }

  private void check(
      AccelerationRuntime runtime,
      ScalarEncoding encoding,
      VectorSimilarityFunction metric,
      int dimensions,
      boolean production)
      throws Exception {
    try (Directory directory = new MMapDirectory(createTempDir())) {
      int docs = 300;
      var config =
          new IndexWriterConfig()
              .setCodec(
                  TestUtil.alwaysKnnVectorsFormat(
                      new Lucene104ScalarQuantizedVectorsFormat(encoding)));
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int d = 0; d < docs; d++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("v", vector(dimensions), metric));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        var segment = (SegmentReader) reader.leaves().get(0).reader();
        KnnVectorsReader vectors = segment.getVectorReader();
        if (vectors instanceof PerFieldKnnVectorsFormat.FieldsReader fields)
          vectors = fields.getFieldReader("v");
        QuantizedByteVectorValues values =
            (QuantizedByteVectorValues)
                ((QuantizedVectorsReader) vectors).getQuantizedVectorValues("v");
        float[] query = vector(dimensions);
        var java =
            new Lucene104ScalarQuantizedVectorScorer(DefaultFlatVectorScorer.INSTANCE, null)
                .getRandomVectorScorer(metric, values, query);
        var accelerated =
            new Lucene104ScalarQuantizedVectorScorer(DefaultFlatVectorScorer.INSTANCE, runtime)
                .getRandomVectorScorer(metric, values.copy(), query);
        long before = runtime.inventory().counters().get(Capability.OSQ_GATHER).nativeCalls();
        for (int count : new int[] {1, 32, 64, 300}) {
          int[] nodes = new int[count];
          for (int i = 0; i < count; i++) nodes[i] = random().nextInt(docs);
          float[] expected = new float[count + 1], actual = new float[count + 1];
          actual[count] = -9;
          float expectedMax = java.bulkScore(nodes, expected, count);
          float actualMax = accelerated.bulkScore(nodes, actual, count);
          for (int i = 0; i < count; i++)
            assertEquals(
                encoding + " " + metric + " d=" + dimensions + " row " + i,
                Float.floatToRawIntBits(expected[i]),
                Float.floatToRawIntBits(actual[i]));
          assertEquals(Float.floatToRawIntBits(expectedMax), Float.floatToRawIntBits(actualMax));
          assertEquals(-9f, actual[count], 0f);
          assertEquals(java.score(nodes[0]), accelerated.score(nodes[0]), 0f);
        }
        assertEquals(
            before + (production ? 1 : 4),
            runtime.inventory().counters().get(Capability.OSQ_GATHER).nativeCalls());
        var supplier =
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
                metric, segment.getFloatVectorValues("v"));
        var graph = HnswGraphBuilder.create(supplier, 16, 40, 19).build(values.size());
        long graphBefore = runtime.inventory().counters().get(Capability.OSQ_GATHER).nativeCalls();
        var expectedHits =
            HnswGraphSearcher.search(java, 10, graph, null, Integer.MAX_VALUE).topDocs();
        var actualHits =
            HnswGraphSearcher.search(accelerated, 10, graph, null, Integer.MAX_VALUE).topDocs();
        assertEquals(expectedHits.totalHits, actualHits.totalHits);
        assertEquals(expectedHits.scoreDocs.length, actualHits.scoreDocs.length);
        for (int i = 0; i < expectedHits.scoreDocs.length; i++) {
          assertEquals(expectedHits.scoreDocs[i].doc, actualHits.scoreDocs[i].doc);
          assertEquals(
              Float.floatToRawIntBits(expectedHits.scoreDocs[i].score),
              Float.floatToRawIntBits(actualHits.scoreDocs[i].score));
        }
        assertSame(accelerated, accelerated.forGraphTraversal());
        if (!production)
          assertTrue(
              "HNSW must use the mapped gather bulk scorer",
              runtime.inventory().counters().get(Capability.OSQ_GATHER).nativeCalls()
                  > graphBefore);
      }
    }
  }

  private static float[] vector(int dimensions) {
    float[] v = new float[dimensions];
    for (int i = 0; i < dimensions; i++) v[i] = random().nextFloat() * 2 - 1;
    v[0] += 0.5f;
    return v;
  }
}
