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
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestKnnExactScorerOwnership extends LuceneTestCase {
  public void testExactSuccessClosesAndRefundsOnce() throws Exception {
    verify(false, false);
  }

  public void testExactScoringFailureClosesAndRefundsOnce() throws Exception {
    verify(true, false);
  }

  public void testAlreadyExpiredTimeoutScoresOneBatchAndClosesOnce() throws Exception {
    verify(false, true);
  }

  public void testAlreadyExpiredTimeoutStillRejectsDimensionMismatch() throws Exception {
    try (var dir = new ByteBuffersDirectory()) {
      try (var writer = new IndexWriter(dir, new IndexWriterConfig())) {
        var value = new Document();
        value.add(new KnnByteVectorField("v", new byte[] {1, 2}));
        writer.addDocument(value);
      }
      try (var reader = DirectoryReader.open(dir)) {
        var query = new KnnByteVectorQuery("v", new byte[] {1}, 1);
        var leaf = reader.leaves().getFirst();
        expectThrows(
            IllegalArgumentException.class,
            () -> query.exactSearch(leaf, DocIdSetIterator.all(1), () -> true));
      }
    }
  }

  private void verify(boolean fail, boolean expired) throws Exception {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(4096, 0, 0), 4096);
    var opened = new AtomicInteger();
    var closed = new AtomicInteger();
    var drained = new AtomicInteger();
    try (var dir = new ByteBuffersDirectory()) {
      try (var writer = new IndexWriter(dir, new IndexWriterConfig())) {
        for (int doc = 0; doc < 8; doc++) {
          var value = new Document();
          value.add(
              new KnnByteVectorField(
                  "v", new byte[] {(byte) doc}, VectorSimilarityFunction.DOT_PRODUCT));
          writer.addDocument(value);
        }
        writer.forceMerge(1);
      }
      try (var reader = DirectoryReader.open(dir)) {
        var query =
            new KnnByteVectorQuery("v", new byte[] {1}, 8) {
              @Override
              ExactScorer createExactScorer(LeafReaderContext context, FieldInfo info, long cost)
                  throws IOException {
                opened.incrementAndGet();
                var scope =
                    new RequestMemoryScope(
                        ledger,
                        new MemoryLedger.Owner("test", "exact", 0, "one"),
                        bytes -> {
                          throw new AssertionError("this fixture owns heap only");
                        });
                scope.onDrained(drained::incrementAndGet);
                var heap = scope.reserveHeap(1024);
                VectorScorer scorer = createVectorScorer(context, info);
                if (fail) {
                  VectorScorer original = scorer;
                  scorer =
                      new VectorScorer() {
                        @Override
                        public float score() throws IOException {
                          return original.score();
                        }

                        @Override
                        public DocIdSetIterator iterator() {
                          return original.iterator();
                        }

                        @Override
                        public Bulk bulk(DocIdSetIterator docs) {
                          return (upTo, liveDocs, buffer) -> {
                            throw new IOException("owned-scoring-test");
                          };
                        }
                      };
                }
                return new ExactScorer(
                    scorer,
                    () -> {
                      closed.incrementAndGet();
                      try {
                        heap.close();
                      } finally {
                        scope.close();
                      }
                    });
              }
            };
        var leaf = reader.leaves().getFirst();
        int refs = leaf.reader().getRefCount();
        if (fail)
          expectThrows(
              IOException.class, () -> query.exactSearch(leaf, DocIdSetIterator.all(8), null));
        else {
          var hits = query.exactSearch(leaf, DocIdSetIterator.all(8), expired ? () -> true : null);
          assertEquals(expired ? 0 : 8, hits.scoreDocs.length);
          assertEquals(
              expired ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO,
              hits.totalHits.relation());
        }
        assertEquals(1, opened.get());
        assertEquals(opened.get(), closed.get());
        assertEquals(opened.get(), drained.get());
        assertEquals(0L, ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
        assertEquals(refs, leaf.reader().getRefCount());
      }
    }
  }
}
