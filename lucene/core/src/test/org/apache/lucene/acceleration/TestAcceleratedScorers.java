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
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.codecs.hnsw.AcceleratedFlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorScorer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.LegacyQuantizedByteVectorValues;
import org.apache.lucene.util.quantization.ScalarQuantizer;

/** Query-batch parity and the unchanged Java supplier/index-reader contract. */
public class TestAcceleratedScorers extends LuceneTestCase {
  private static AccelerationRuntime javaRuntime() throws IOException {
    return AccelerationRuntime.forQualification(
        TestAccelerationRuntime.config(
            AccelerationConfig.Mode.OFF, null, java.util.Set.of(), java.util.Set.of(), 1));
  }

  private static AccelerationRuntime nativeRuntime() throws IOException {
    var runtime =
        AccelerationRuntime.forQualification(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                java.util.Set.of(),
                java.util.Set.of(),
                1));
    assertTrue(runtime.inventory().rejections().toString(), runtime.supports(Capability.VECTOR_I8));
    return runtime;
  }

  private static AcceleratedFlatVectorsScorer wrap(AccelerationRuntime runtime) {
    return new AcceleratedFlatVectorsScorer(DefaultFlatVectorScorer.INSTANCE, runtime);
  }

  public void testJavaFallbackAndFloatDelegate() throws Exception {
    var runtime = javaRuntime();
    var wrapper = wrap(runtime);
    byte[][] rows = {{1, -2, 3}, {-3, 2, 1}, {0, 0, 0}, {127, -128, 4}};
    var bytes = new ScratchValues(rows, null, -1);
    assertParity(
        wrapper.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, bytes, rows[0]),
        DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
            VectorSimilarityFunction.EUCLIDEAN, bytes.copy(), rows[0]),
        new int[] {3, 0, 3, 2, 1});
    var floats =
        FloatVectorValues.fromFloats(List.of(new float[] {1, 2, 3}, new float[] {3, 2, 1}), 3);
    var expected =
        DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
            VectorSimilarityFunction.EUCLIDEAN, floats, new float[] {2, 1, 3});
    var actual =
        wrapper.getRandomVectorScorer(
            VectorSimilarityFunction.EUCLIDEAN, floats, new float[] {2, 1, 3});
    assertParity(actual, expected, new int[] {1, 0, 1});
    assertEquals(0L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
  }

  public void testNativeSignedByteMetricsTailsAndOrdinals() throws Exception {
    var runtime = nativeRuntime();
    var wrapper = wrap(runtime);
    for (int dimensions : new int[] {1, 3, 31, 128, 513}) {
      byte[][] rows = new byte[4][dimensions];
      byte[] query = new byte[dimensions];
      random().nextBytes(query);
      for (byte[] row : rows) random().nextBytes(row);
      var values = new ScratchValues(rows, new int[] {0, 2, 5, 7}, -1);
      for (var metric : VectorSimilarityFunction.values()) {
        var nativeScorer = wrapper.getRandomVectorScorer(metric, values, query);
        var javaScorer =
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(metric, values.copy(), query);
        assertParity(nativeScorer, javaScorer, new int[] {3, 1, 3, 0, 2, 1});
        assertMapping(nativeScorer, javaScorer);
        for (int ordinal = 0; ordinal < values.size(); ordinal++) {
          assertBits(javaScorer.score(ordinal), nativeScorer.score(ordinal));
        }
      }
    }
    assertEquals(20L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testNativeQuantizedRescoreCorrectionsAndMetrics() throws Exception {
    var runtime = nativeRuntime();
    assertTrue(runtime.supports(Capability.QUANTIZED_RESCORE));
    var oracle = new Lucene99ScalarQuantizedVectorScorer(DefaultFlatVectorScorer.INSTANCE);
    var wrapper = new AcceleratedFlatVectorsScorer(oracle, runtime);
    for (byte bits : new byte[] {4, 7}) {
      for (int dimensions : new int[] {3, 17, 129}) {
        byte[][] rows = new byte[4][dimensions];
        float[] query = new float[dimensions];
        for (int i = 0; i < dimensions; i++) query[i] = (i % 7 + 1) / 16f;
        for (byte[] row : rows) {
          random().nextBytes(row);
          for (int i = 0; i < row.length; i++) row[i] &= (byte) ((1 << bits) - 1);
        }
        var quantizer = new ScalarQuantizer(-1, 1, bits);
        var values = new QuantizedValues(rows, quantizer, new float[] {-0.25f, 0.25f, -0.75f, 0});
        for (var metric : VectorSimilarityFunction.values()) {
          assertParity(
              wrapper.getRandomVectorScorer(metric, values, query),
              oracle.getRandomVectorScorer(metric, values.copy(), query),
              new int[] {3, 1, 3, 0, 2, 1});
        }
      }
    }
    assertEquals(
        24L, runtime.inventory().counters().get(Capability.QUANTIZED_RESCORE).nativeCalls());
  }

  public void testGraphTraversalRetainsOriginalJavaScorer() throws Exception {
    var runtime = nativeRuntime();
    byte[][] rows = {{1, 2, 3}, {-5, 7, 2}, {7, -4, 2}};
    var values = new ScratchValues(rows, null, -1);
    var actual =
        wrap(runtime).getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, rows[0]);
    var expected =
        DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
            VectorSimilarityFunction.EUCLIDEAN, values.copy(), rows[0]);
    assertParity(actual.forGraphTraversal(), expected, new int[] {2, 0, 1, 2});
    assertEquals(0L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
    var supplier =
        DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
            VectorSimilarityFunction.EUCLIDEAN, values.copy());
    var graph = HnswGraphBuilder.create(supplier, 2, 10, 7).build(values.size());
    var actualHits = HnswGraphSearcher.search(actual, 1, graph, null, Integer.MAX_VALUE).topDocs();
    var expectedHits =
        HnswGraphSearcher.search(expected, 1, graph, null, Integer.MAX_VALUE).topDocs();
    assertEquals(expectedHits.totalHits, actualHits.totalHits);
    assertEquals(expectedHits.scoreDocs.length, actualHits.scoreDocs.length);
    for (int i = 0; i < expectedHits.scoreDocs.length; i++) {
      assertEquals(expectedHits.scoreDocs[i].doc, actualHits.scoreDocs[i].doc);
      assertBits(expectedHits.scoreDocs[i].score, actualHits.scoreDocs[i].score);
    }
    assertEquals(0L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
    assertSame(actual.forGraphTraversal(), actual.forGraphTraversal().forGraphTraversal());
    assertParity(actual, expected, new int[] {2, 0, 1, 2});
    assertEquals(1L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
  }

  public void testNativeFloatQueryMetricsAndMutableQuery() throws Exception {
    var runtime = nativeRuntime();
    assertTrue(
        runtime.inventory().rejections().toString(), runtime.supports(Capability.VECTOR_F32));
    for (int dimensions : new int[] {1, 3, 7, 8}) {
      List<float[]> rows = new java.util.ArrayList<>();
      for (int i = 0; i < 4; i++) {
        float[] row = new float[dimensions];
        for (int d = 0; d < dimensions; d++) row[d] = (random().nextInt(128) - 64) / 64f;
        if (row[0] == 0) row[0] = 0.5f;
        rows.add(row);
      }
      float[] query = new float[dimensions];
      Arrays.fill(query, 0.25f);
      for (var metric : VectorSimilarityFunction.values()) {
        var values = FloatVectorValues.fromFloats(rows, dimensions);
        var actual = wrap(runtime).getRandomVectorScorer(metric, values, query);
        var expected =
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(metric, values.copy(), query);
        assertParity(actual, expected, new int[] {3, 1, 3, 0, 2, 1});
        query[0] = -query[0];
        assertParity(actual, expected, new int[] {2, 1, 0});
      }
    }
    assertTrue(runtime.inventory().counters().get(Capability.VECTOR_F32).nativeCalls() > 0);
  }

  public void testUnqualifiedFloatReductionUsesOriginalJavaWithoutCopy() throws Exception {
    var runtime = nativeRuntime();
    int dimensions = VectorUtil.exactScalarFloatLimit() + 1;
    float[] query = new float[dimensions], row = new float[dimensions];
    Arrays.fill(query, 0.25f);
    Arrays.fill(row, -0.5f);
    FloatVectorValues values =
        new FloatVectorValues() {
          @Override
          public int dimension() {
            return dimensions;
          }

          @Override
          public int size() {
            return 1;
          }

          @Override
          public float[] vectorValue(int ordinal) {
            return row;
          }

          @Override
          public FloatVectorValues copy() throws IOException {
            throw new IOException(
                "unsupported arithmetic profile must not require a second reader");
          }

          @Override
          public DocIndexIterator iterator() {
            return createDenseIterator();
          }
        };
    for (var metric : VectorSimilarityFunction.values()) {
      var actual = wrap(runtime).getRandomVectorScorer(metric, values, query);
      var expected = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(metric, values, query);
      assertParity(actual, expected, new int[] {0, 0, 0});
    }
    assertEquals(0L, runtime.inventory().counters().get(Capability.VECTOR_F32).nativeCalls());
    assertTrue(runtime.supports(Capability.VECTOR_F32));
  }

  public void testPartitionFailureNeverPublishesAnEarlierNativeFrame() throws Exception {
    var runtime = nativeRuntime();
    byte[][] rows = new byte[4][513];
    for (byte[] row : rows) random().nextBytes(row);
    var values = new ScratchValues(rows, null, 2);
    var scorer =
        wrap(runtime).getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, rows[0]);
    int[] nodes = new int[4096];
    Arrays.fill(nodes, 1);
    nodes[nodes.length - 1] = 2;
    float[] scores = new float[nodes.length];
    Arrays.fill(scores, 123.5f);
    expectThrows(IOException.class, () -> scorer.bulkScore(nodes, scores, nodes.length));
    for (float score : scores) assertBits(123.5f, score);
    assertTrue(runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls() > 0);
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testByteQueryMutationsRemainVisibleToSingleAndBulkScores() throws Exception {
    var runtime = nativeRuntime();
    byte[][] rows = {{1, -2, 3}, {-128, 127, 7}, {0, 2, -4}};
    int[] nodes = {2, 0, 1, 2};
    for (var metric : VectorSimilarityFunction.values()) {
      byte[] query = {1, 2, 3};
      var values = new ScratchValues(rows, null, -1);
      var actual = wrap(runtime).getRandomVectorScorer(metric, values, query);
      var expected =
          DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(metric, values.copy(), query);
      for (byte first : new byte[] {-128, 127}) {
        query[0] = first;
        query[1] = (byte) -first;
        assertParity(actual, expected, nodes);
        for (int node : nodes) assertBits(expected.score(node), actual.score(node));
      }
    }
    assertEquals(8L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testUnsupportedDimensionsDoNotRequireAnExtraValuesCopy() throws Exception {
    var runtime = nativeRuntime();
    byte[] query = new byte[32769], row = new byte[query.length];
    java.util.Arrays.fill(query, (byte) 1);
    java.util.Arrays.fill(row, (byte) 2);
    var values =
        new ScratchValues(new byte[][] {row}, null, -1) {
          @Override
          public ByteVectorValues copy() throws IOException {
            throw new IOException("copy unavailable; original Java reader remains usable");
          }
        };
    var scorer =
        wrap(runtime).getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, query);
    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, row);
    assertBits(expected, scorer.score(0));
    float[] scores = new float[2];
    assertBits(expected, scorer.bulkScore(new int[] {0, 0}, scores, 2));
    assertBits(expected, scores[0]);
    assertBits(expected, scores[1]);
    assertEquals(0L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
  }

  public void testOversizedDenseBatchPartitionsWithoutQuarantining() throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                java.util.Set.of(),
                java.util.Set.of(),
                64L << 20,
                32L << 20,
                1,
                true,
                AccelerationConfig.StoragePolicy.CURRENT));
    assertTrue(runtime.inventory().rejections().toString(), runtime.supports(Capability.VECTOR_I8));
    byte[] query = new byte[32768], row = new byte[query.length];
    Arrays.fill(query, (byte) 1);
    Arrays.fill(row, (byte) 2);
    var scorer =
        wrap(runtime)
            .getRandomVectorScorer(
                VectorSimilarityFunction.EUCLIDEAN,
                new ScratchValues(new byte[][] {row}, null, -1),
                query);
    int[] nodes = new int[513];
    float[] scores = new float[nodes.length];
    float expected = VectorSimilarityFunction.EUCLIDEAN.compare(query, row);
    assertBits(expected, scorer.bulkScore(nodes, scores, nodes.length));
    for (float score : scores) assertBits(expected, score);
    assertTrue(runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls() > 1);
    assertTrue(runtime.supports(Capability.VECTOR_I8));
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testNativeZeroNormCosineAndSignedExtremes() throws Exception {
    var runtime = nativeRuntime();
    byte[][] rows = {{0, 0, 0}, {-128, -128, -128}, {127, 127, 127}, {1, 0, -1}};
    for (byte[] query : rows) {
      for (var metric : VectorSimilarityFunction.values()) {
        var values = new ScratchValues(rows, null, -1);
        assertParity(
            wrap(runtime).getRandomVectorScorer(metric, values, query),
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(metric, values.copy(), query),
            new int[] {0, 3, 1, 2, 0});
      }
    }
  }

  public void testSupplierCopiesRemainIndependentAndJavaBacked() throws Exception {
    // This is unconditional; a native inventory is also exercised in the explicit pack fixture.
    checkSupplier(javaRuntime());
  }

  public void testNativeInventoryDoesNotEnableIndexingSuppliers() throws Exception {
    var runtime = nativeRuntime();
    checkSupplier(runtime);
    assertEquals(0L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
  }

  private void checkSupplier(AccelerationRuntime runtime) throws Exception {
    byte[][] rows = {{1, -2, 3}, {-3, 2, 1}, {4, -5, 6}, {-7, 8, -9}};
    for (var metric : VectorSimilarityFunction.values()) {
      var values = new ScratchValues(rows, new int[] {0, 2, 5, 7}, -1);
      var supplier = wrap(runtime).getRandomVectorScorerSupplier(metric, values);
      var first = supplier.scorer();
      first.setScoringOrdinal(0);
      float before = first.score(1);
      var second = supplier.copy().scorer();
      second.setScoringOrdinal(2);
      assertBits(metric.compare(rows[2], rows[3]), second.score(3));
      assertBits(before, first.score(1));
      first.setScoringOrdinal(3);
      var oracle =
          DefaultFlatVectorScorer.INSTANCE
              .getRandomVectorScorerSupplier(metric, values.copy())
              .scorer();
      oracle.setScoringOrdinal(3);
      assertParity(first, oracle, new int[] {3, 1, 3, 0, 2});
      assertMapping(first, oracle);
    }
  }

  public void testStagingIOExceptionPreservesCallerOutput() throws Exception {
    var runtime = nativeRuntime();
    byte[][] rows = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    var values = new ScratchValues(rows, null, 2);
    var scorer =
        wrap(runtime)
            .getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, new byte[] {1, 1});
    float[] scores = {111, 222, 333};
    IOException failure =
        expectThrows(IOException.class, () -> scorer.bulkScore(new int[] {0, 2, 1}, scores, 3));
    assertEquals("injected vector read failure", failure.getMessage());
    assertArrayEquals(new float[] {111, 222, 333}, scores, 0f);
    assertEquals(0L, runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls());
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testBudgetSizedPartitionsUseTheSameJavaScores() throws Exception {
    var runtime =
        AccelerationRuntime.forQualification(
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                TestAccelerationRuntime.nativePack(),
                java.util.Set.of(),
                java.util.Set.of(),
                16L << 20,
                1L << 20,
                1,
                true,
                AccelerationConfig.StoragePolicy.CURRENT));
    assertTrue(runtime.inventory().rejections().toString(), runtime.supports(Capability.VECTOR_I8));
    byte[][] rows = new byte[4][513];
    for (byte[] row : rows) random().nextBytes(row);
    var values = new ScratchValues(rows, null, -1);
    int[] ordinals = new int[4096];
    for (int i = 0; i < ordinals.length; i++) ordinals[i] = i % 4;
    assertParity(
        wrap(runtime).getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, values, rows[0]),
        DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
            VectorSimilarityFunction.DOT_PRODUCT, values.copy(), rows[0]),
        ordinals);
    assertTrue(runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls() > 1);
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }

  public void testJavaWrittenIndexDeleteAppendMergeReopenNativeQuery() throws Exception {
    var runtime = nativeRuntime();
    try (Directory directory = newDirectory()) {
      try (var writer =
          new IndexWriter(directory, newIndexWriterConfig().setCodec(TestUtil.getDefaultCodec()))) {
        for (int id = 0; id < 24; id++) addDocument(writer, id);
        writer.commit();
      }
      try (var reader = DirectoryReader.open(directory)) {
        checkReader(reader, runtime);
      }
      try (var writer =
          new IndexWriter(directory, newIndexWriterConfig().setCodec(TestUtil.getDefaultCodec()))) {
        writer.deleteDocuments(new Term("id", "5"), new Term("id", "11"));
        for (int id = 24; id < 32; id++) addDocument(writer, id);
        writer.commit();
        try (var reader = DirectoryReader.open(writer)) {
          assertEquals(30, reader.numDocs());
          checkReader(reader, runtime);
        }
        writer.forceMerge(1);
        writer.commit();
      }
      try (var reader = DirectoryReader.open(directory)) {
        assertEquals(30, reader.numDocs());
        assertEquals(1, reader.leaves().size());
        checkReader(reader, runtime);
      }
    }
    assertTrue(runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls() > 0);
  }

  private void addDocument(IndexWriter writer, int id) throws IOException {
    Document document = new Document();
    document.add(new StringField("id", Integer.toString(id), Field.Store.NO));
    if (id % 3 != 0)
      document.add(
          new KnnByteVectorField(
              "vector",
              new byte[] {(byte) id, (byte) (id * -2), (byte) (id + 7)},
              VectorSimilarityFunction.EUCLIDEAN));
    writer.addDocument(document);
  }

  private void checkReader(DirectoryReader reader, AccelerationRuntime runtime) throws Exception {
    for (var context : reader.leaves()) {
      var leaf = context.reader();
      var values = leaf.getByteVectorValues("vector");
      if (values == null || values.size() == 0) continue;
      byte[] query = {4, -5, 6};
      var actual =
          wrap(runtime).getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, query);
      var expected =
          DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
              VectorSimilarityFunction.EUCLIDEAN, values.copy(), query);
      int[] ordinals = new int[values.size() + 2];
      for (int i = 0; i < values.size(); i++) ordinals[i] = values.size() - 1 - i;
      ordinals[values.size()] = 0;
      ordinals[values.size() + 1] = values.size() - 1;
      assertParity(actual, expected, ordinals);
      Bits actualLive = actual.getAcceptOrds(leaf.getLiveDocs());
      Bits expectedLive = expected.getAcceptOrds(leaf.getLiveDocs());
      for (int ordinal = 0; ordinal < values.size(); ordinal++) {
        assertEquals(expected.ordToDoc(ordinal), actual.ordToDoc(ordinal));
        if (expectedLive == null) assertNull(actualLive);
        else assertEquals(expectedLive.get(ordinal), actualLive.get(ordinal));
      }
    }
  }

  private static void assertParity(
      RandomVectorScorer actual, RandomVectorScorer expected, int[] ordinals) throws IOException {
    int[] original = ordinals.clone();
    float[] expectedScores = new float[ordinals.length + 2];
    float[] actualScores = new float[ordinals.length + 2];
    Arrays.fill(expectedScores, -456f);
    Arrays.fill(actualScores, -456f);
    assertBits(
        expected.bulkScore(ordinals, expectedScores, ordinals.length),
        actual.bulkScore(ordinals, actualScores, ordinals.length));
    for (int i = 0; i < expectedScores.length; i++) assertBits(expectedScores[i], actualScores[i]);
    assertArrayEquals(original, ordinals);
    assertBits(Float.NEGATIVE_INFINITY, actual.bulkScore(new int[0], new float[0], 0));
  }

  private static void assertMapping(RandomVectorScorer actual, RandomVectorScorer expected) {
    FixedBitSet docs = new FixedBitSet(8);
    docs.set(0);
    docs.set(5);
    assertEquals(expected.maxOrd(), actual.maxOrd());
    assertNull(actual.getAcceptOrds(null));
    Bits a = actual.getAcceptOrds(docs), b = expected.getAcceptOrds(docs);
    assertEquals(b.length(), a.length());
    for (int i = 0; i < actual.maxOrd(); i++) {
      assertEquals(expected.ordToDoc(i), actual.ordToDoc(i));
      assertEquals(b.get(i), a.get(i));
    }
  }

  private static void assertBits(float expected, float actual) {
    assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual));
  }

  private static final class QuantizedValues extends LegacyQuantizedByteVectorValues {
    private final byte[][] rows;
    private final ScalarQuantizer quantizer;
    private final float[] corrections;
    private final byte[] scratch;

    QuantizedValues(byte[][] rows, ScalarQuantizer quantizer, float[] corrections) {
      this.rows = rows;
      this.quantizer = quantizer;
      this.corrections = corrections;
      scratch = new byte[rows[0].length];
    }

    @Override
    public int dimension() {
      return scratch.length;
    }

    @Override
    public int size() {
      return rows.length;
    }

    @Override
    public ScalarQuantizer getScalarQuantizer() {
      return quantizer;
    }

    @Override
    public float getScoreCorrectionConstant(int ordinal) {
      return corrections[ordinal];
    }

    @Override
    public byte[] vectorValue(int ordinal) {
      System.arraycopy(rows[ordinal], 0, scratch, 0, scratch.length);
      return scratch;
    }

    @Override
    public LegacyQuantizedByteVectorValues copy() {
      return new QuantizedValues(rows, quantizer, corrections);
    }

    @Override
    public DocIndexIterator iterator() {
      return createDenseIterator();
    }
  }

  /** Models an IndexInput-backed cursor: returned vectors share mutable scratch within a copy. */
  private static class ScratchValues extends ByteVectorValues {
    private final byte[][] rows;
    private final int[] documents;
    private final int failingOrdinal;
    private final byte[] scratch;

    ScratchValues(byte[][] rows, int[] documents, int failingOrdinal) {
      this.rows = rows;
      this.documents = documents;
      this.failingOrdinal = failingOrdinal;
      scratch = new byte[rows[0].length];
    }

    @Override
    public int dimension() {
      return scratch.length;
    }

    @Override
    public int size() {
      return rows.length;
    }

    @Override
    public int ordToDoc(int ordinal) {
      return documents == null ? ordinal : documents[ordinal];
    }

    @Override
    public byte[] vectorValue(int ordinal) throws IOException {
      if (ordinal == failingOrdinal) throw new IOException("injected vector read failure");
      System.arraycopy(rows[ordinal], 0, scratch, 0, scratch.length);
      return scratch;
    }

    @Override
    public ByteVectorValues copy() throws IOException {
      return new ScratchValues(rows, documents, failingOrdinal);
    }

    @Override
    public DocIndexIterator iterator() {
      return createDenseIterator();
    }
  }
}
