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
package org.apache.lucene.benchmark.jmh;

import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestExactByteQueryBenchmark extends LuceneTestCase {
  public void testJava512BulkKeepsAllRowsAndJavaTail() throws Exception {
    DocIdSetIterator docs = DocIdSetIterator.all(513);
    VectorScorer source =
        new VectorScorer() {
          @Override
          public float score() {
            return (docs.docID() % 4) * 0.25f;
          }

          @Override
          public DocIdSetIterator iterator() {
            return docs;
          }
        };
    var bulk = ExactByteQueryBenchmark.java512(source).bulk(null);
    var buffer = new DocAndFloatFeatureBuffer();
    assertEquals(0.75f, bulk.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer), 0);
    assertEquals(512, buffer.size);
    assertEquals(512, buffer.docs.length);
    for (int i = 0; i < buffer.size; i++) {
      assertEquals(i, buffer.docs[i]);
      assertEquals((i % 4) * 0.25f, buffer.features[i], 0);
    }
    bulk.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
    assertEquals(1, buffer.size);
    assertEquals(512, buffer.docs[0]);
    assertEquals(0f, buffer.features[0], 0);
    bulk.nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);
    assertEquals(0, buffer.size);
  }

  public void testIndependentSignedOracleAndTieOrderRejectCorruption() {
    byte[] target = {Byte.MIN_VALUE, Byte.MAX_VALUE};
    byte[][] vectors = {{Byte.MIN_VALUE, Byte.MAX_VALUE}, {Byte.MIN_VALUE, Byte.MAX_VALUE}, {0, 0}};
    var oracle = ExactByteQueryBenchmark.oracle(target, vectors);
    assertEquals(0, oracle[0].doc);
    assertEquals(1, oracle[1].doc);
    assertEquals(2, oracle[2].doc);
    assertEquals(0.5f + (16384 + 16129) / 65536f, oracle[0].score, 0f);
    assertEquals(0.5f, oracle[2].score, 0f);
    ExactByteQueryBenchmark.verify(
        new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), oracle), oracle);
    ScoreDoc[] corrupt = {oracle[1], oracle[0], oracle[2]};
    expectThrows(
        IllegalStateException.class,
        () ->
            ExactByteQueryBenchmark.verify(
                new TopDocs(new TotalHits(3, TotalHits.Relation.EQUAL_TO), corrupt), oracle));
  }

  public void testWholeJava64AndJava512QueriesUseTheSameOracle() throws Exception {
    assumeTrue(
        "fresh-fork enrolled Java control", Boolean.getBoolean("tests.byteExact.javaControl"));
    for (String arm : new String[] {"JAVA64", "JAVA512"}) {
      var bench = new ExactByteQueryBenchmark();
      bench.implementation = arm;
      bench.setup();
      try {
        bench.beforeIteration();
        bench.query(new ExactByteQueryBenchmark.QueryOwner());
        bench.afterIteration();
      } finally {
        bench.close();
      }
    }
  }
}
