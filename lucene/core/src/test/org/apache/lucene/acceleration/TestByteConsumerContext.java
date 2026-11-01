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

import java.nio.file.Files;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Set;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Capability inventory alone must never prepare a frame for another consumer. */
public class TestByteConsumerContext extends LuceneTestCase {
  public void testOwnedProfileDeclinesGenericAndStandaloneBeforePreparingAnyFrame()
      throws Exception {
    var pack = TestAccelerationRuntime.nativePack();
    var file = TrustedPathFixtures.file("byte-owned-only-context");
    String prefix =
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.VECTOR_I8.id
            + "\t384\t512\t512\t1\t-\tmapped\t199360\tTEST-ONLY-byte-owned-context";
    Files.writeString(file, prefix + "\t" + OwnedByteBatchScorer.EXACT_OWNED_CONTEXT + "\n");
    Files.setPosixFilePermissions(file, PosixFilePermissions.fromString("rw-------"));
    String key = "lucene.acceleration.native.qualificationPath";
    String old = System.getProperty(key);
    System.setProperty(key, file.toString());
    try (var runtime =
        new AccelerationRuntime(
            TestAccelerationRuntime.config(
                AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1))) {
      var before = runtime.inventory();
      var similarity = VectorSimilarityFunction.DOT_PRODUCT;
      assertTrue(runtime.supports(Capability.VECTOR_I8));
      assertEquals(0, runtime.measuredByteCapacity(384, similarity));
      assertNull(runtime.measuredByteKernel(384, 512, similarity));
      assertNull(PreparedByteScoring.prepareMeasured(runtime, new byte[384], similarity));
      String standalone = OwnedByteBatchScorer.EXACT_STANDALONE_CONTEXT;
      assertEquals(0, runtime.measuredByteCapacity(384, similarity, standalone));
      assertNull(runtime.measuredByteKernel(384, 512, similarity, standalone));
      assertNull(
          PreparedByteScoring.prepareMeasured(runtime, new byte[384], similarity, standalone));
      String packed = OwnedByteBatchScorer.EXACT_STANDALONE_PACKED_CONTEXT;
      assertEquals(0, runtime.measuredByteCapacity(384, similarity, packed));
      assertNull(runtime.measuredByteKernel(384, 512, similarity, packed));
      assertNull(PreparedByteScoring.prepareMeasured(runtime, new byte[384], similarity, packed));
      assertEquals(before.memory(), runtime.inventory().memory());
      assertEquals(before.execution(), runtime.inventory().execution());
      assertEquals(before.counters(), runtime.inventory().counters());
      String owned = OwnedByteBatchScorer.EXACT_OWNED_CONTEXT;
      assertEquals(512, runtime.measuredByteCapacity(384, similarity, owned));
      assertNotNull(runtime.measuredByteKernel(384, 512, similarity, owned));
      var task =
          new AccelerationRuntime.TaskContext(
              new org.apache.lucene.acceleration.memory.MemoryLedger.Owner(
                  "test", "byte-context", 0, "owned"),
              0);
      try (var binding = runtime.enterTask(task);
          var prepared =
              PreparedByteScoring.prepareMeasured(runtime, new byte[384], similarity, owned)) {
        assertNotNull(binding);
        assertNotNull(prepared);
        assertTrue(
            runtime
                    .inventory()
                    .memory()
                    .charged()
                    .get(org.apache.lucene.acceleration.memory.MemoryLedger.Kind.NATIVE)
                > 0);
      } finally {
        task.cancel();
      }
      assertEquals(before.memory().charged(), runtime.inventory().memory().charged());
      assertEquals(before.execution(), runtime.inventory().execution());
    } finally {
      if (old == null) System.clearProperty(key);
      else System.setProperty(key, old);
    }
  }

  public void testLegacyGenericProfileStillRunsThePublicOwnedScorer() throws Exception {
    var pack = TestAccelerationRuntime.nativePack();
    var file = TrustedPathFixtures.file("byte-legacy-generic-context");
    Files.writeString(
        file,
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.VECTOR_I8.id
            + "\t384\t512\t512\t1\t-\tmapped\t199360\tTEST-ONLY-byte-legacy-generic\n");
    Files.setPosixFilePermissions(file, PosixFilePermissions.fromString("rw-------"));
    String key = "lucene.acceleration.native.qualificationPath";
    String old = System.getProperty(key);
    System.setProperty(key, file.toString());
    try (var runtime =
            new AccelerationRuntime(
                TestAccelerationRuntime.config(
                    AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1));
        var directory = new org.apache.lucene.store.MMapDirectory(createTempDir())) {
      byte[] target = new byte[384];
      java.util.Arrays.fill(target, (byte) 1);
      try (var writer =
          new org.apache.lucene.index.IndexWriter(
              directory,
              new org.apache.lucene.index.IndexWriterConfig()
                  .setCodec(new org.apache.lucene.codecs.lucene104.Lucene104Codec())
                  .setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE)
                  .setMaxBufferedDocs(10000))) {
        for (int doc = 0; doc < 512; doc++) {
          var value = new org.apache.lucene.document.Document();
          byte[] vector = new byte[384];
          java.util.Arrays.fill(vector, (byte) (doc % 251 - 125));
          value.add(
              new org.apache.lucene.document.KnnByteVectorField(
                  "v", vector, VectorSimilarityFunction.DOT_PRODUCT));
          writer.addDocument(value);
        }
      }
      assertEquals(
          0,
          runtime.measuredByteCapacity(
              384, VectorSimilarityFunction.DOT_PRODUCT, OwnedByteBatchScorer.EXACT_OWNED_CONTEXT));
      try (var reader = org.apache.lucene.index.DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        var leaf = reader.leaves().getFirst().reader();
        int refs = leaf.getRefCount();
        int[] docs = new int[512];
        for (int i = 0; i < docs.length; i++) docs[i] = i;
        float[] output = new float[512];
        try (var scorer = OwnedByteBatchScorer.create(runtime, leaf, "v", target)) {
          scorer.bulkScore(docs, output, docs.length);
          for (int i = 0; i < output.length; i++)
            assertEquals(0.5f + (384 * (i % 251 - 125)) / (float) (384 * (1 << 15)), output[i], 0f);
        }
        assertEquals(refs, leaf.getRefCount());
      }
      var counter = runtime.inventory().counters().get(Capability.VECTOR_I8);
      assertEquals(1, counter.nativeCalls());
      assertEquals(0, counter.javaCalls());
      assertEquals(0, runtime.inventory().execution().activeScopes());
      for (long bytes : runtime.inventory().memory().charged().values()) assertEquals(0, bytes);
    } finally {
      if (old == null) System.clearProperty(key);
      else System.setProperty(key, old);
    }
  }
}
