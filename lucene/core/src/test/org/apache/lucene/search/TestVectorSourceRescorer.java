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
package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.ArrayUtil;

/** Built-in vector sources rescore per hit through the stock values path. */
public class TestVectorSourceRescorer extends LuceneTestCase {
  private static final float[] FLOAT_QUERY = {1, 2, 3};
  private static final byte[] BYTE_QUERY = {1, 2, 3};

  private static final class Counts {
    int approximateFactories, fullFactories, bulkFactories, scalarCalls, vectorReads;

    void reset() {
      approximateFactories = fullFactories = bulkFactories = scalarCalls = vectorReads = 0;
    }

    VectorScorer wrap(VectorScorer delegate) {
      if (delegate == null) return null;
      return new VectorScorer() {
        @Override
        public float score() throws IOException {
          scalarCalls++;
          return delegate.score();
        }

        @Override
        public DocIdSetIterator iterator() {
          return delegate.iterator();
        }

        @Override
        public Bulk bulk(DocIdSetIterator matches) throws IOException {
          bulkFactories++;
          return delegate.bulk(matches);
        }
      };
    }
  }

  private static final class CountingReader extends FilterDirectoryReader {
    CountingReader(DirectoryReader reader, Counts counts) throws IOException {
      super(
          reader,
          new SubReaderWrapper() {
            @Override
            public LeafReader wrap(LeafReader leaf) {
              return new FilterLeafReader(leaf) {
                @Override
                public FloatVectorValues getFloatVectorValues(String field) throws IOException {
                  var values = super.getFloatVectorValues(field);
                  if (values == null) return null;
                  return new FloatVectorValues() {
                    @Override
                    public int size() {
                      return values.size();
                    }

                    @Override
                    public int dimension() {
                      return values.dimension();
                    }

                    @Override
                    public float[] vectorValue(int ord) throws IOException {
                      counts.vectorReads++;
                      return values.vectorValue(ord);
                    }

                    @Override
                    public FloatVectorValues copy() throws IOException {
                      return values.copy();
                    }

                    @Override
                    public DocIndexIterator iterator() {
                      return values.iterator();
                    }

                    @Override
                    public VectorScorer scorer(float[] target) throws IOException {
                      counts.approximateFactories++;
                      return counts.wrap(values.scorer(target));
                    }

                    @Override
                    public VectorScorer rescorer(float[] target) throws IOException {
                      counts.fullFactories++;
                      return counts.wrap(values.rescorer(target));
                    }
                  };
                }

                @Override
                public ByteVectorValues getByteVectorValues(String field) throws IOException {
                  var values = super.getByteVectorValues(field);
                  if (values == null) return null;
                  return new ByteVectorValues() {
                    @Override
                    public int size() {
                      return values.size();
                    }

                    @Override
                    public int dimension() {
                      return values.dimension();
                    }

                    @Override
                    public byte[] vectorValue(int ord) throws IOException {
                      return values.vectorValue(ord);
                    }

                    @Override
                    public ByteVectorValues copy() throws IOException {
                      return values.copy();
                    }

                    @Override
                    public DocIndexIterator iterator() {
                      return values.iterator();
                    }

                    @Override
                    public VectorScorer scorer(byte[] target) throws IOException {
                      counts.approximateFactories++;
                      return counts.wrap(values.scorer(target));
                    }
                  };
                }

                @Override
                public CacheHelper getCoreCacheHelper() {
                  return leaf.getCoreCacheHelper();
                }

                @Override
                public CacheHelper getReaderCacheHelper() {
                  return leaf.getReaderCacheHelper();
                }
              };
            }
          });
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) {
      throw new UnsupportedOperationException("fixture reader is not reopened");
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
      return in.getReaderCacheHelper();
    }
  }

  private DirectoryReader write(Directory directory, Counts counts) throws Exception {
    var config =
        newIndexWriterConfig()
            .setCodec(TestUtil.alwaysKnnVectorsFormat(new Lucene99HnswVectorsFormat()))
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMaxBufferedDocs(10000);
    try (var writer = new IndexWriter(directory, config)) {
      for (int leaf = 0, doc = 0; leaf < 3; leaf++) {
        int size = leaf < 2 ? 80 : 7;
        for (int i = 0; i < size; i++, doc++) {
          var document = new Document();
          document.add(new StringField("id", Integer.toString(doc), Field.Store.NO));
          if (leaf == 0 || leaf == 1 && i % 3 != 0) {
            document.add(
                new KnnFloatVectorField(
                    "f",
                    new float[] {
                      doc % 5 + 1.17f, (doc % 7) * .31f + .12f, (doc % 11) * .17f + 1.23f
                    }));
            document.add(new KnnByteVectorField("b", new byte[] {(byte) (doc % 5 + 1), 2, 3}));
          }
          writer.addDocument(document);
        }
        writer.commit();
      }
      writer.deleteDocuments(new Term("id", "1"));
    }
    var reader = new CountingReader(DirectoryReader.open(directory), counts);
    assertEquals(3, reader.leaves().size());
    assertNotNull(reader.leaves().getFirst().reader().getLiveDocs());
    counts.reset();
    return reader;
  }

  private static TopDocs hits() {
    var hits = new ArrayList<ScoreDoc>();
    for (int doc = 0; doc < 167; doc++) if (doc != 1) hits.add(new ScoreDoc(doc, 0f));
    hits.add(new ScoreDoc(83, 2f));
    hits.add(hits.getFirst());
    java.util.Collections.reverse(hits);
    return new TopDocs(
        new TotalHits(166, TotalHits.Relation.EQUAL_TO), hits.toArray(ScoreDoc[]::new));
  }

  private static final class Recorder extends DoubleValuesSourceRescorer {
    final List<String> calls = new ArrayList<>();

    Recorder(DoubleValuesSource source) {
      super(source);
    }

    @Override
    protected float combine(float first, boolean present, double second) {
      calls.add(
          Float.floatToRawIntBits(first)
              + ":"
              + present
              + ":"
              + Double.doubleToRawLongBits(second));
      return first + (present ? (float) second : -1f);
    }
  }

  private static TopDocs copy(TopDocs input) {
    var copies = new java.util.IdentityHashMap<ScoreDoc, ScoreDoc>();
    ScoreDoc[] hits = new ScoreDoc[input.scoreDocs.length];
    for (int i = 0; i < hits.length; i++) {
      ScoreDoc original = input.scoreDocs[i];
      hits[i] = copies.computeIfAbsent(original, hit -> new ScoreDoc(hit.doc, hit.score));
    }
    return new TopDocs(input.totalHits, hits);
  }

  private static TopDocs scalar(
      IndexSearcher searcher, DoubleValuesSource source, TopDocs input, int topN, Recorder recorder)
      throws IOException {
    ScoreDoc[] hits = copy(input).scoreDocs;
    Arrays.sort(hits, (a, b) -> Integer.compare(a.doc, b.doc));
    for (ScoreDoc hit : hits) {
      LeafReaderContext leaf = null;
      for (var candidate : searcher.getIndexReader().leaves())
        if (hit.doc >= candidate.docBase
            && hit.doc < candidate.docBase + candidate.reader().maxDoc()) {
          leaf = candidate;
          break;
        }
      DoubleValues values = source.getValues(leaf, null);
      boolean present = values.advanceExact(hit.doc - leaf.docBase);
      hit.score = recorder.combine(hit.score, present, present ? values.doubleValue() : 0f);
    }
    Arrays.sort(hits, ScoreDoc.COMPARATOR);
    return new TopDocs(
        input.totalHits, ArrayUtil.copyOfSubArray(hits, 0, Math.min(topN, hits.length)));
  }

  private static void same(TopDocs expected, TopDocs actual) {
    assertEquals(expected.totalHits, actual.totalHits);
    assertEquals(expected.scoreDocs.length, actual.scoreDocs.length);
    for (int i = 0; i < expected.scoreDocs.length; i++) {
      assertEquals(expected.scoreDocs[i].doc, actual.scoreDocs[i].doc);
      assertEquals(
          Float.floatToRawIntBits(expected.scoreDocs[i].score),
          Float.floatToRawIntBits(actual.scoreDocs[i].score));
    }
  }

  public void testBuiltinSourcesScorePerHit() throws Exception {
    Counts counts = new Counts();
    DoubleValuesSource[] sources = {
      new FloatVectorSimilarityValuesSource(FLOAT_QUERY, "f"),
      new ByteVectorSimilarityValuesSource(BYTE_QUERY, "b"),
      new FullPrecisionFloatVectorSimilarityValuesSource(FLOAT_QUERY, "f")
    };
    try (Directory directory = newDirectory();
        var reader = write(directory, counts)) {
      var searcher = new IndexSearcher(reader);
      for (var source : sources) {
        TopDocs input = hits();
        var oracle = new Recorder(source);
        TopDocs expected = scalar(searcher, source, input, 13, oracle);
        counts.reset();
        var actual = new Recorder(source);
        same(expected, actual.rescore(searcher, copy(input), 13));
        assertEquals(oracle.calls, actual.calls);
        assertEquals(0, counts.bulkFactories);
        assertTrue(counts.approximateFactories + counts.fullFactories > 2);
        assertTrue(counts.scalarCalls > 0);
      }
    }
  }

  public void testExplicitSimilarityAndSubclassRemainScalar() throws Exception {
    Counts counts = new Counts();
    DoubleValuesSource[] sources = {
      new FullPrecisionFloatVectorSimilarityValuesSource(
          FLOAT_QUERY, "f", VectorSimilarityFunction.DOT_PRODUCT),
      new FullPrecisionFloatVectorSimilarityValuesSource(FLOAT_QUERY, "f") {}
    };
    try (Directory directory = newDirectory();
        var reader = write(directory, counts)) {
      var searcher = new IndexSearcher(reader);
      for (var source : sources) {
        var oracle = new Recorder(source);
        TopDocs input = hits();
        TopDocs expected = scalar(searcher, source, input, 1000, oracle);
        counts.reset();
        var actual = new Recorder(source);
        same(expected, actual.rescore(searcher, copy(input), 1000));
        assertEquals(oracle.calls, actual.calls);
        assertEquals(0, counts.bulkFactories);
        assertEquals(0, counts.approximateFactories);
        if (source == sources[0]) {
          assertEquals(0, counts.fullFactories);
          assertTrue(counts.vectorReads > 0);
        } else assertTrue(counts.fullFactories > 2);
      }
    }
  }

  public void testGenericSourceRetainsPerHitConstructionAndCombineOrdering() throws Exception {
    Counts counts = new Counts();
    int[] factories = {0};
    var source =
        new DoubleValuesSource() {
          @Override
          public DoubleValues getValues(LeafReaderContext leaf, DoubleValues scores) {
            double value = ++factories[0];
            return new DoubleValues() {
              @Override
              public double doubleValue() {
                return value;
              }

              @Override
              public boolean advanceExact(int doc) {
                return (doc & 3) != 0;
              }
            };
          }

          @Override
          public boolean needsScores() {
            return false;
          }

          @Override
          public boolean isCacheable(LeafReaderContext leaf) {
            return false;
          }

          @Override
          public DoubleValuesSource rewrite(IndexSearcher searcher) {
            return this;
          }

          @Override
          public boolean equals(Object other) {
            return other == this;
          }

          @Override
          public int hashCode() {
            return System.identityHashCode(this);
          }

          @Override
          public String toString() {
            return "per-hit-construction-control";
          }
        };
    try (Directory directory = newDirectory();
        var reader = write(directory, counts)) {
      var searcher = new IndexSearcher(reader);
      TopDocs input = hits();
      var oracle = new Recorder(source);
      TopDocs expected = scalar(searcher, source, input, 31, oracle);
      factories[0] = 0;
      var actual = new Recorder(source);
      same(expected, actual.rescore(searcher, copy(input), 31));
      assertEquals(oracle.calls, actual.calls);
      assertEquals(input.scoreDocs.length, factories[0]);
      assertEquals(0, counts.bulkFactories);
    }
  }
}
