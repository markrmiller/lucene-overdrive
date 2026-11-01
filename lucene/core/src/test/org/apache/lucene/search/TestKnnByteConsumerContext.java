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
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.OwnedByteBatchScorer;
import org.apache.lucene.acceleration.memory.MemoryLedger;
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

/** Fresh-fork owned-only profile; its private row is not deployment approval. */
public class TestKnnByteConsumerContext extends LuceneTestCase {
  @BeforeClass
  public static void requireOwnedOnlyStartupPolicy() {
    assumeTrue(
        "dedicated owned-only SYSTEM fixture", Boolean.getBoolean("tests.byteExact.ownedOnly"));
    var runtime = AccelerationRuntime.system();
    assertTrue(
        runtime.inventory().rejections().toString(),
        runtime.supports(Capability.VECTOR_I8, OwnedByteBatchScorer.EXACT_OWNED_CONTEXT));
    assertFalse(runtime.supports(Capability.VECTOR_I8, "generic"));
    assertFalse(
        runtime.supports(Capability.VECTOR_I8, OwnedByteBatchScorer.EXACT_STANDALONE_CONTEXT));
  }

  public void testOwnedOnlyProfilePublishesNativeRankingUnderExistingOwner() throws Exception {
    check(true, false);
  }

  public void testOwnedOnlyProfileDeclinesStandaloneBeforeOptionalBuffers() throws Exception {
    check(false, false);
  }

  public void testOwnedOnlyProfileKeepsPublicGenericScorerJavaWithoutNativeAdmission()
      throws Exception {
    check(true, true);
  }

  private void check(boolean owner, boolean generic) throws Exception {
    var runtime = AccelerationRuntime.system();
    byte[] target = new byte[384];
    Arrays.fill(target, (byte) 1);
    try (var directory = new MMapDirectory(createTempDir())) {
      try (var writer =
          new IndexWriter(
              directory,
              new IndexWriterConfig()
                  .setCodec(new Lucene104Codec())
                  .setMergePolicy(NoMergePolicy.INSTANCE)
                  .setMaxBufferedDocs(10000))) {
        for (int doc = 0; doc < 512; doc++) {
          Document value = new Document();
          byte[] row = new byte[384];
          Arrays.fill(row, component(doc));
          value.add(new KnnByteVectorField("v", row, VectorSimilarityFunction.DOT_PRODUCT));
          value.add(new StringField("eligible", "yes", Field.Store.NO));
          writer.addDocument(value);
        }
      }
      var beforeReader = runtime.inventory().memory().charged();
      try (var reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        var leaf = reader.leaves().getFirst();
        int refs = leaf.reader().getRefCount();
        var before = runtime.inventory();
        var task =
            owner
                ? new AccelerationRuntime.TaskContext(
                    new MemoryLedger.Owner("test", "exact-owned-context", 0, "one"),
                    0,
                    (who, kind, bytes, purpose, cancelled) -> {
                      if (generic && kind == MemoryLedger.Kind.NATIVE && bytes > 0)
                        throw new AssertionError(
                            "generic caller must decline before native frame admission");
                      return runtime.ledger().tryReserve(who, kind, bytes, purpose);
                    })
                : null;
        try (var binding = task == null ? null : runtime.enterTask(task)) {
          if (task != null) assertNotNull(binding);
          var current = runtime.currentOwner();
          var values = leaf.reader().getByteVectorValues("v");
          assertFalse(
              OwnedByteBatchScorer.supportsExactMappedBatch(
                  runtime, values, VectorSimilarityFunction.DOT_PRODUCT));
          if (!owner) {
            try (var probe =
                ExactByteVectorScorer.tryCreate(
                    leaf,
                    leaf.reader().getFieldInfos().fieldInfo("v"),
                    values,
                    target,
                    512,
                    values.scorer(target))) {
              assertNull(
                  "standalone must decline before the optional wrapper/buffers exist", probe);
            }
          }
          if (generic) {
            int[] docs = new int[512];
            for (int i = 0; i < docs.length; i++) docs[i] = i;
            float[] output = new float[512];
            try (var scorer = OwnedByteBatchScorer.create(runtime, leaf.reader(), "v", target)) {
              scorer.bulkScore(docs, output, docs.length);
              for (int i = 0; i < output.length; i++) assertEquals(score(i), output[i], 0);
            }
          } else {
            var searcher = new IndexSearcher(reader);
            searcher.setQueryCache(null);
            var hits =
                searcher.search(
                    new KnnByteVectorQuery(
                        "v", target, 512, new TermQuery(new Term("eligible", "yes"))),
                    512);
            var expected = new ArrayList<ScoreDoc>();
            for (int doc = 0; doc < 512; doc++) expected.add(new ScoreDoc(doc, score(doc)));
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
          }
          assertEquals(current, runtime.currentOwner());
        } finally {
          if (task != null) task.cancel();
        }
        var completed = runtime.inventory();
        assertEquals(
            owner && !generic ? 1 : 0,
            completed.counters().get(Capability.VECTOR_I8).nativeCalls()
                - before.counters().get(Capability.VECTOR_I8).nativeCalls());
        if (owner && !generic) {
          assertEquals(
              before.counters().get(Capability.VECTOR_I8).javaCalls(),
              completed.counters().get(Capability.VECTOR_I8).javaCalls());
          assertEquals(
              before.counters().get(Capability.VECTOR_I8).fallbacks(),
              completed.counters().get(Capability.VECTOR_I8).fallbacks());
        }
        assertEquals(before.execution().activeScopes(), completed.execution().activeScopes());
        assertEquals(refs, leaf.reader().getRefCount());
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

  private static float score(int doc) {
    return 0.5f + (384 * component(doc)) / (float) (384 * (1 << 15));
  }
}
