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
package org.apache.lucene.search;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import org.apache.lucene.acceleration.AccelerationConfig;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.BeforeClass;

/** A private functional policy proves producer version isolation, not deployment qualification. */
public class TestPackedStandaloneContext extends LuceneTestCase {
  @BeforeClass
  public static void requireDedicatedSystemPolicy() {
    assumeTrue("fresh SYSTEM policy fixture", Boolean.getBoolean("tests.bytePackedContext"));
    assertEquals(AccelerationConfig.Mode.AUTO, AccelerationRuntime.system().config().mode());
  }

  public void testNaturalQueryConsumesOnlyPackedProducerProfile() throws Exception {
    boolean legacy = Boolean.getBoolean("tests.bytePackedContext.legacyPolicy");
    String packed = "lucene.knn-byte.exact.standalone.packed.v2";
    String old = "lucene.knn-byte.exact.standalone.v1";
    var runtime = AccelerationRuntime.system();
    assertTrue(
        runtime.inventory().toString(),
        runtime.supports(Capability.VECTOR_I8, legacy ? old : packed));
    assertFalse(runtime.supports(Capability.VECTOR_I8, legacy ? packed : old));
    assertNull(runtime.currentOwner());
    byte[] query = new byte[384];
    Arrays.fill(query, (byte) 1);
    try (var directory = new MMapDirectory(createTempDir())) {
      try (var writer =
          new IndexWriter(
              directory,
              new IndexWriterConfig()
                  .setCodec(new Lucene104Codec())
                  .setMergePolicy(NoMergePolicy.INSTANCE)
                  .setMaxBufferedDocs(10000))) {
        for (int doc = 0; doc < 4096; doc++) {
          byte[] vector = new byte[384];
          Arrays.fill(vector, component(doc));
          var value = new Document();
          value.add(new KnnByteVectorField("v", vector, VectorSimilarityFunction.DOT_PRODUCT));
          value.add(
              new StringField(
                  "eligible", doc >= 1536 && doc < 2048 ? "yes" : "no", Field.Store.NO));
          writer.addDocument(value);
        }
      }
      var beforeReader = runtime.inventory().memory().charged();
      try (var reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        var leaf = reader.leaves().getFirst().reader();
        int refs = leaf.getRefCount();
        var before = runtime.inventory();
        var hits =
            new IndexSearcher(reader)
                .search(
                    new KnnByteVectorQuery(
                        "v", query, 512, new TermQuery(new Term("eligible", "yes"))),
                    512);
        var expected = new ArrayList<ScoreDoc>();
        for (int doc = 1536; doc < 2048; doc++)
          expected.add(
              new ScoreDoc(doc, 0.5f + (384 * component(doc)) / (float) (384 * (1 << 15))));
        expected.sort(
            Comparator.<ScoreDoc>comparingDouble(hit -> -hit.score)
                .thenComparingInt(hit -> hit.doc));
        assertEquals(512, hits.scoreDocs.length);
        for (int i = 0; i < expected.size(); i++) {
          assertEquals(expected.get(i).doc, hits.scoreDocs[i].doc);
          assertEquals(
              Float.floatToIntBits(expected.get(i).score),
              Float.floatToIntBits(hits.scoreDocs[i].score));
        }
        assertEquals(
            "natural query must use only its packed producer profile",
            legacy ? 0 : 1,
            runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls()
                - before.counters().get(Capability.VECTOR_I8).nativeCalls());
        assertEquals(refs, leaf.getRefCount());
        assertEquals(
            before.execution().activeScopes(), runtime.inventory().execution().activeScopes());
        assertNull(runtime.currentOwner());
      }
      assertEquals(beforeReader, runtime.inventory().memory().charged());
    }
  }

  private static byte component(int doc) {
    return switch (doc % 4) {
      case 0 -> Byte.MIN_VALUE;
      case 1 -> Byte.MAX_VALUE;
      case 2 -> 0;
      default -> 1;
    };
  }
}
