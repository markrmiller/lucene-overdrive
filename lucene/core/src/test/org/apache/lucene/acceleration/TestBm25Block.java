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
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;

public class TestBm25Block extends LuceneTestCase {

  /** Every postings block of several terms scores bit-identically to the Java bulk path. */
  public void testMatchesJavaBulkScoringOnMappedNorms() throws Exception {
    try (var runtime =
            AccelerationRuntime.forQualification(config(TestAccelerationRuntime.nativePack()));
        Directory directory = new MMapDirectory(createTempDir())) {
      index(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        var leaf = reader.leaves().get(0).reader();
        var searcher = new IndexSearcher(reader);
        var similarity = new BM25Similarity();
        var collection = searcher.collectionStatistics("f");
        TermsEnum terms = leaf.terms("f").iterator();
        int checked = 0;
        while (terms.next() != null) {
          var term = new Term("f", terms.term());
          var stats = searcher.termStatistics(term, terms.docFreq(), terms.totalTermFreq());
          float boost = 1 + random().nextInt(3);
          var java = similarity.scorer(boost, collection, stats).asBulkSimScorer();
          try (var block =
              Bm25Block.prepare(runtime, leaf, "f", similarity, boost, collection, stats)) {
            assertNotNull(block);
            PostingsEnum postings = terms.postings(null, PostingsEnum.FREQS);
            var norms = leaf.getNormValues("f");
            var buffer = new DocAndFloatFeatureBuffer();
            postings.nextDoc();
            while (postings.docID() != DocIdSetIterator.NO_MORE_DOCS) {
              postings.nextPostings(DocIdSetIterator.NO_MORE_DOCS, buffer);
              int size = buffer.size;
              long[] normValues = new long[size];
              norms.longValues(size, buffer.docs, normValues, 1L);
              float[] expected = new float[size + 1], actual = new float[size + 1];
              actual[size] = -3;
              java.score(size, buffer.features.clone(), normValues, expected);
              block.score(size, buffer.docs, buffer.features, actual);
              for (int i = 0; i < size; i++)
                assertEquals(
                    Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(actual[i]));
              assertEquals(-3f, actual[size], 0f);
              checked += size;
            }
          }
        }
        assertTrue(checked > 1000);
      }
    }
  }

  /** Norms that are not mapped one-byte norms decline instead of failing. */
  public void testUnmappedNormsDecline() throws Exception {
    try (var runtime =
            AccelerationRuntime.forQualification(config(TestAccelerationRuntime.nativePack()));
        Directory directory = new ByteBuffersDirectory()) {
      index(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        var searcher = new IndexSearcher(reader);
        var stats = searcher.termStatistics(new Term("f", "t1"), 1, 1);
        assertNull(
            Bm25Block.prepare(
                runtime,
                reader.leaves().get(0).reader(),
                "f",
                new BM25Similarity(),
                1,
                searcher.collectionStatistics("f"),
                stats));
      }
    }
  }

  public void testPreparationClosesNewScorerWhenViewReleaseFails() throws Exception {
    try (var runtime =
            AccelerationRuntime.forQualification(config(TestAccelerationRuntime.nativePack()));
        Directory directory = new MMapDirectory(createTempDir())) {
      index(directory);
      try (DirectoryReader owner = DirectoryReader.open(directory)) {
        var leaf = owner.leaves().get(0).reader();
        var searcher = new IndexSearcher(owner);
        var collection = searcher.collectionStatistics("f");
        var stats = searcher.termStatistics(new Term("f", "t1"), 1, 1);
        var before = runtime.ledger().snapshot().charged();
        long scopes = runtime.inventory().execution().activeScopes();
        var closeFailure = new IOException("view resource close failed");
        var view =
            new FilterLeafReader(leaf) {
              @Override
              public NumericDocValues getNormValues(String field) throws IOException {
                close();
                return super.getNormValues(field);
              }

              @Override
              protected void doClose() throws IOException {
                assertEquals(scopes + 1, runtime.inventory().execution().activeScopes());
                throw closeFailure;
              }

              @Override
              public CacheHelper getCoreCacheHelper() {
                return in.getCoreCacheHelper();
              }

              @Override
              public CacheHelper getReaderCacheHelper() {
                return in.getReaderCacheHelper();
              }
            };
        assertSame(
            closeFailure,
            expectThrows(
                IOException.class,
                () ->
                    Bm25Block.prepare(
                        runtime, view, "f", new BM25Similarity(), 1, collection, stats)));
        assertEquals(0, view.getRefCount());
        assertEquals(1, leaf.getRefCount());
        assertEquals(
            "failed factory return must release the constructed scorer",
            scopes,
            runtime.inventory().execution().activeScopes());
        assertEquals(before, runtime.ledger().snapshot().charged());
      }
    }
  }

  private void index(Directory directory) throws Exception {
    try (IndexWriter writer =
        new IndexWriter(directory, new IndexWriterConfig().setCodec(TestUtil.getDefaultCodec()))) {
      for (int d = 0; d < 3000; d++) {
        StringBuilder text = new StringBuilder();
        int words = 1 + random().nextInt(d % 97 == 0 ? 400 : 30);
        for (int w = 0; w < words; w++) text.append(" t").append(random().nextInt(12));
        Document doc = new Document();
        doc.add(new TextField("f", text.toString(), Field.Store.NO));
        writer.addDocument(doc);
      }
      writer.forceMerge(1);
    }
  }

  private static AccelerationConfig config(java.nio.file.Path pack) {
    return new AccelerationConfig(
        AccelerationConfig.Mode.AUTO,
        pack,
        Set.of(),
        Set.of(),
        16L << 20,
        8L << 20,
        1,
        true,
        AccelerationConfig.StoragePolicy.CURRENT);
  }
}
