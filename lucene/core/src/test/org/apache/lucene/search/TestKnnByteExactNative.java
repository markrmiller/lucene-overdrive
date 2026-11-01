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
import java.util.Comparator;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
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

/** Fresh-fork native functional fixture; its private policy does not approve deployment. */
public class TestKnnByteExactNative extends LuceneTestCase {
  private boolean denyBuffer;
  private boolean rejectNativeWorkspace;
  private boolean compactNativeBudget;

  @BeforeClass
  public static void selectPrivateFunctionalPolicy() throws Exception {
    assumeTrue(
        "dedicated plain-byte SYSTEM fixture", Boolean.getBoolean("tests.byteExact.functional"));
    String pack = System.getProperty("tests.acceleration.pack");
    assumeTrue("native pack and JDK25 required", pack != null && Runtime.version().feature() >= 25);
    assertNotNull(
        "private policy must be supplied before JVM startup",
        System.getProperty("lucene.acceleration.native.qualificationPath"));
    assertEquals(java.nio.file.Path.of(pack), AccelerationRuntime.system().config().packPath());
    assertTrue(
        AccelerationRuntime.system().inventory().rejections().toString(),
        AccelerationRuntime.system().supports(Capability.VECTOR_I8));
  }

  public void testNaturalExactQueryConsumesMapped512AndKeepsRankingAndTies() throws Exception {
    exercise(512, 1, false, false, false, 384, VectorSimilarityFunction.DOT_PRODUCT, 1);
  }

  public void testMappedQueryFitsA32KiBNativeRequestBudget() throws Exception {
    compactNativeBudget = true;
    try {
      exercise(512, 1, false, false, false, 384, VectorSimilarityFunction.DOT_PRODUCT, 1);
    } finally {
      compactNativeBudget = false;
    }
  }

  public void testNaturalExactQueryKeepsTheJavaTailInOrder() throws Exception {
    exercise(513, 1, false, false, false, 384, VectorSimilarityFunction.DOT_PRODUCT, 1);
  }

  public void testScatteredFilterKeepsJavaAndNeverAllocatesNativeWorkspace() throws Exception {
    rejectNativeWorkspace = true;
    try {
      exercise(1024, 1, true, false, false, 384, VectorSimilarityFunction.DOT_PRODUCT, 0);
    } finally {
      rejectNativeWorkspace = false;
    }
  }

  public void testMissingVectorsKeepOriginalJavaRanking() throws Exception {
    exercise(513, 1, false, true, false, 384, VectorSimilarityFunction.DOT_PRODUCT, 0);
  }

  public void testDeletedVectorKeepsOriginalJavaRanking() throws Exception {
    exercise(513, 1, false, false, true, 384, VectorSimilarityFunction.DOT_PRODUCT, 0);
  }

  public void testMultipleLeavesKeepJavaAndExactGlobalTieOrder() throws Exception {
    exercise(512, 2, false, false, false, 384, VectorSimilarityFunction.DOT_PRODUCT, 0);
  }

  public void testOtherDimensionAndSimilarityKeepTheCodecScorer() throws Exception {
    exercise(512, 1, false, false, false, 128, VectorSimilarityFunction.DOT_PRODUCT, 0);
    exercise(512, 1, false, false, false, 384, VectorSimilarityFunction.EUCLIDEAN, 0);
  }

  public void testCallerBudgetDeclinesBeforeBuffersAndKeepsItsOwner() throws Exception {
    denyBuffer = true;
    try {
      exercise(512, 1, false, false, false, 384, VectorSimilarityFunction.DOT_PRODUCT, 0);
    } finally {
      denyBuffer = false;
    }
  }

  private void exercise(
      int perLeaf,
      int segments,
      boolean scattered,
      boolean missing,
      boolean deleted,
      int dimension,
      VectorSimilarityFunction similarity,
      long nativeCalls)
      throws Exception {
    var runtime = AccelerationRuntime.system();
    int rows = perLeaf * segments;
    byte[] target = new byte[dimension];
    random().nextBytes(target);
    byte[][] values = new byte[rows][dimension];
    for (int doc = 0; doc < rows; doc++) {
      if (doc % 4 == 0) java.util.Arrays.fill(values[doc], Byte.MIN_VALUE);
      else if (doc % 4 == 1) java.util.Arrays.fill(values[doc], Byte.MAX_VALUE);
      else random().nextBytes(values[doc]);
    }
    try (var dir = new MMapDirectory(createTempDir())) {
      try (var writer =
          new IndexWriter(
              dir,
              new IndexWriterConfig()
                  .setCodec(new Lucene104Codec())
                  .setMergePolicy(NoMergePolicy.INSTANCE)
                  .setMaxBufferedDocs(10000))) {
        for (int doc = 0; doc < rows; doc++) {
          Document value = new Document();
          if (!missing || doc != 17)
            value.add(new KnnByteVectorField("v", values[doc], similarity));
          value.add(new StringField("id", "d" + doc, org.apache.lucene.document.Field.Store.NO));
          value.add(
              new StringField(
                  "eligible",
                  scattered && doc % 2 != 0 ? "no" : "yes",
                  org.apache.lucene.document.Field.Store.NO));
          writer.addDocument(value);
          if ((doc + 1) % perLeaf == 0) writer.commit();
        }
        if (deleted) writer.deleteDocuments(new Term("id", "d17"));
      }
      var readerLifetimeBefore = runtime.inventory().memory().charged();
      try (var reader = DirectoryReader.open(dir)) {
        assertEquals(segments, reader.leaves().size());
        var before = runtime.inventory();
        int[] refs = reader.leaves().stream().mapToInt(c -> c.reader().getRefCount()).toArray();
        var previousOwner = runtime.currentOwner();
        var deniedTask =
            denyBuffer || rejectNativeWorkspace || compactNativeBudget
                ? new AccelerationRuntime.TaskContext(
                    new MemoryLedger.Owner("test", "byte-exact", 0, "controlled"),
                    0,
                    (owner, kind, bytes, purpose, cancelled) -> {
                      if (rejectNativeWorkspace && kind == MemoryLedger.Kind.NATIVE && bytes > 0)
                        throw new AssertionError(
                            "scattered filters must not construct native workspace");
                      long ownedNative =
                          compactNativeBudget && kind == MemoryLedger.Kind.NATIVE
                              ? runtime.ledger().allocations(Integer.MAX_VALUE).stream()
                                  .filter(a -> a.owner().equals(owner) && a.kind() == kind)
                                  .mapToLong(MemoryLedger.AllocationRecord::capacity)
                                  .sum()
                              : 0;
                      return denyBuffer
                              || compactNativeBudget
                                  && kind == MemoryLedger.Kind.NATIVE
                                  && ownedNative + bytes > 32L * 1024
                          ? new MemoryLedger.Admission(MemoryLedger.Reason.REQUEST_LIMIT, null)
                          : runtime.ledger().tryReserve(owner, kind, bytes, purpose);
                    })
                : null;
        TopDocs actual;
        try (var binding = deniedTask == null ? null : runtime.enterTask(deniedTask)) {
          if (deniedTask != null) assertNotNull(binding);
          var callerOwner = runtime.currentOwner();
          actual =
              new IndexSearcher(reader)
                  .search(
                      new KnnByteVectorQuery(
                          "v", target, rows, new TermQuery(new Term("eligible", "yes"))),
                      rows);
          assertEquals(callerOwner, runtime.currentOwner());
        } finally {
          if (deniedTask != null) deniedTask.cancel();
        }
        assertEquals(previousOwner, runtime.currentOwner());
        var expected = new ArrayList<ScoreDoc>();
        for (int doc = 0; doc < rows; doc++) {
          if (scattered && doc % 2 != 0 || (missing || deleted) && doc == 17) continue;
          expected.add(new ScoreDoc(doc, similarity.compare(target, values[doc])));
        }
        expected.sort(
            Comparator.<ScoreDoc>comparingDouble(hit -> -hit.score)
                .thenComparingInt(hit -> hit.doc));
        assertEquals(expected.size(), actual.scoreDocs.length);
        for (int i = 0; i < expected.size(); i++) {
          assertEquals(expected.get(i).doc, actual.scoreDocs[i].doc);
          assertEquals(
              Float.floatToIntBits(expected.get(i).score),
              Float.floatToIntBits(actual.scoreDocs[i].score));
        }
        assertEquals(
            nativeCalls,
            runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls()
                - before.counters().get(Capability.VECTOR_I8).nativeCalls());
        if (nativeCalls > 0 && perLeaf == 512) {
          var completed = runtime.inventory().counters().get(Capability.VECTOR_I8);
          var started = before.counters().get(Capability.VECTOR_I8);
          assertEquals(started.javaCalls(), completed.javaCalls());
          assertEquals(started.fallbacks(), completed.fallbacks());
        }
        for (int i = 0; i < refs.length; i++)
          assertEquals(refs[i], reader.leaves().get(i).reader().getRefCount());
        assertEquals(
            before.execution().activeScopes(), runtime.inventory().execution().activeScopes());
        assertNull(runtime.currentOwner());
      }
      assertEquals(readerLifetimeBefore, runtime.inventory().memory().charged());
    }
  }
}
