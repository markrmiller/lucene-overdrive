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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.NativeAllocator;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestPreparedByteScoring extends LuceneTestCase {
  private static final MemoryLedger.Owner OWNER =
      new MemoryLedger.Owner("prepared", "test", 0, "prepared");

  private static AccelerationRuntime runtime(long bytes) throws IOException {
    return AccelerationRuntime.forQualification(
        new AccelerationConfig(
            AccelerationConfig.Mode.OFF,
            null,
            Set.of(),
            Set.of(),
            bytes,
            bytes,
            1,
            true,
            AccelerationConfig.StoragePolicy.CURRENT));
  }

  public void testProductionDoesNotLoadUnmeasuredPackAndScalarQualificationStaysJava()
      throws Exception {
    var malformed = createTempDir("unmeasured-pack");
    Files.writeString(malformed.resolve("manifest.json"), "not a manifest");
    for (var path : List.of(malformed, malformed.resolve("missing"))) {
      try (AccelerationRuntime runtime =
          new AccelerationRuntime(
              new AccelerationConfig(
                  AccelerationConfig.Mode.AUTO,
                  path,
                  Set.of(),
                  Set.of(),
                  1024,
                  1024,
                  1,
                  true,
                  AccelerationConfig.StoragePolicy.CURRENT))) {
        assertFalse(runtime.hasKernels());
        assertFalse(runtime.supports(Capability.VECTOR_I8));
        byte[] query = {1, -2};
        byte[][] rows = {{3, 4}, {-5, 6}};
        float[] scores =
            BlockOperations.byteScores(runtime, query, rows, VectorSimilarityFunction.DOT_PRODUCT);
        for (int i = 0; i < rows.length; i++)
          assertEquals(VectorSimilarityFunction.DOT_PRODUCT.compare(query, rows[i]), scores[i], 0);
        var counter = runtime.inventory().counters().get(Capability.VECTOR_I8);
        assertEquals(0L, counter.nativeCalls());
        assertEquals(
            1L, counter.fallbacks().get(AccelerationRuntime.Reason.NO_MEASURED_WIN).longValue());
        assertFalse(runtime.inventory().rejections().isEmpty());
        expectThrows(IOException.class, () -> runtime.diagnosticKernel(Capability.VECTOR_I8));
        expectThrows(IOException.class, runtime::newQualificationScope);
      }
    }
    try (AccelerationRuntime runtime =
        AccelerationRuntime.forQualification(
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                null,
                Set.of(),
                Set.of(),
                1024,
                1024,
                1,
                true,
                AccelerationConfig.StoragePolicy.CURRENT))) {
      assertTrue(runtime.isQualificationOnly());
      assertFalse(runtime.eligible(Capability.VECTOR_I8, 1));
      assertEquals(
          1L,
          runtime
              .inventory()
              .counters()
              .get(Capability.VECTOR_I8)
              .fallbacks()
              .get(AccelerationRuntime.Reason.NO_PACK)
              .longValue());
    }
  }

  public void testOptionalPackPreparedReaderScoringAndPhysicalDrain() throws Exception {
    String pack = System.getProperty("tests.acceleration.pack");
    assumeTrue("optional native reference pack was not supplied", pack != null);
    assumeTrue("native FFM tests require JDK25", Runtime.version().feature() >= 25);
    try (AccelerationRuntime runtime =
        AccelerationRuntime.forQualification(
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                Path.of(pack),
                Set.of(),
                Set.of(),
                16L << 20,
                16L << 20,
                1,
                true,
                AccelerationConfig.StoragePolicy.CURRENT))) {
      assertTrue(runtime.supports(Capability.VECTOR_I8));
      assertFalse(runtime.eligible(Capability.VECTOR_I8, 1));
      assertEquals(
          1L,
          runtime
              .inventory()
              .counters()
              .get(Capability.VECTOR_I8)
              .fallbacks()
              .get(AccelerationRuntime.Reason.SMALL_BATCH)
              .longValue());
      byte[] query = new byte[33];
      random().nextBytes(query);
      byte[][] documents = new byte[7][query.length];
      for (byte[] row : documents) random().nextBytes(row);
      int[] ordinals = {6, 0, 6, 2, 1};
      float[] output = new float[ordinals.length + 2];
      try (var directory = FSDirectory.open(createTempDir("prepared-vector-reader"))) {
        try (var writer = new IndexWriter(directory, new IndexWriterConfig())) {
          for (byte[] row : documents) {
            var document = new Document();
            document.add(new KnnByteVectorField("v", row, VectorSimilarityFunction.DOT_PRODUCT));
            writer.addDocument(document);
          }
          writer.forceMerge(1);
        }
        try (var reader = DirectoryReader.open(directory)) {
          var values = reader.leaves().getFirst().reader().getByteVectorValues("v");
          for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
            var javaScorer =
                FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
                    .getRandomVectorScorer(similarity, values.copy(), query)
                    .forGraphTraversal();
            float[] expected = new float[ordinals.length];
            try (PreparedByteScoring frame =
                PreparedByteScoring.prepare(runtime, query, ordinals.length, similarity)) {
              long held = runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE);
              assertTrue(held > 0);
              for (int iteration = 0; iteration < 10; iteration++) {
                Arrays.fill(output, 42);
                int count = iteration % (ordinals.length + 1);
                float maximum = javaScorer.bulkScore(ordinals, expected, count);
                assertEquals(
                    Float.floatToRawIntBits(maximum),
                    Float.floatToRawIntBits(frame.score(values, ordinals, count, output, 1)));
                for (int i = 0; i < count; i++)
                  assertEquals(
                      Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(output[i + 1]));
                assertEquals(42f, output[0], 0);
                assertEquals(42f, output[count + 1], 0);
                int[] contiguous = new int[count];
                for (int i = 0; i < count; i++) contiguous[i] = i;
                maximum = javaScorer.bulkScore(contiguous, expected, count);
                int references = reader.leaves().getFirst().reader().getRefCount();
                assertEquals(
                    Float.floatToRawIntBits(maximum),
                    Float.floatToRawIntBits(
                        frame.scoreContiguous(
                            reader.leaves().getFirst().reader(), values, 0, count, output, 1)));
                for (int i = 0; i < count; i++)
                  assertEquals(
                      Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(output[i + 1]));
                assertEquals(references, reader.leaves().getFirst().reader().getRefCount());
                assertEquals(
                    held,
                    runtime
                        .ledger()
                        .snapshot()
                        .charged()
                        .get(MemoryLedger.Kind.NATIVE)
                        .longValue());
              }
            }
            assertEquals(
                0L,
                runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
            assertEquals(0L, runtime.inventory().execution().activeScopes());
          }
        }
      }
    }
  }

  public void testLargeNativeCallSurvivesReaderCloseAndConcurrentMerge() throws Exception {
    int dimensions = 384, count = 1024;
    byte[] query = new byte[dimensions];
    random().nextBytes(query);
    byte[][] rows = new byte[count][dimensions];
    for (byte[] row : rows) random().nextBytes(row);
    try (var runtime =
            AccelerationRuntime.forQualification(
                TestAccelerationRuntime.config(
                    AccelerationConfig.Mode.AUTO,
                    TestAccelerationRuntime.nativePack(),
                    Set.of(),
                    Set.of(),
                    1));
        var directory = new org.apache.lucene.store.MMapDirectory(createTempDir());
        var writer =
            new IndexWriter(
                directory,
                new IndexWriterConfig()
                    .setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE)
                    .setCodec(new org.apache.lucene.codecs.lucene104.Lucene104Codec()))) {
      for (int segment = 0; segment < 2; segment++) {
        for (byte[] row : rows) {
          var document = new Document();
          document.add(new KnnByteVectorField("v", row, VectorSimilarityFunction.DOT_PRODUCT));
          writer.addDocument(document);
        }
        writer.commit();
      }
      var reader = DirectoryReader.open(directory);
      var leaf = reader.leaves().getFirst().reader();
      var entered = new java.util.concurrent.CountDownLatch(1);
      var retired = new java.util.concurrent.CountDownLatch(1);
      var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
      var nativeKernel = runtime.diagnosticKernel(Capability.VECTOR_I8);
      var fenced =
          new AccelerationProvider.Kernel() {
            @Override
            public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
              return nativeKernel.estimate(shape);
            }

            @Override
            public AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
                throws IOException {
              entered.countDown();
              try {
                if (!retired.await(30, java.util.concurrent.TimeUnit.SECONDS))
                  throw new IOException("close/merge fence timed out");
              } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IOException(error);
              }
              return nativeKernel.execute(batch);
            }
          };
      Thread closer =
          new Thread(
              () -> {
                try {
                  if (!entered.await(30, java.util.concurrent.TimeUnit.SECONDS))
                    throw new AssertionError("native owner did not enter");
                  reader.close();
                  assertTrue("native scorer must retain old reader", leaf.getRefCount() > 0);
                  writer
                      .getConfig()
                      .setMergePolicy(new org.apache.lucene.index.LogDocMergePolicy());
                  writer.forceMerge(1);
                  try (var reopened = DirectoryReader.open(writer)) {
                    assertEquals(count * 2, reopened.numDocs());
                  }
                } catch (Throwable error) {
                  failure.set(error);
                } finally {
                  retired.countDown();
                }
              });
      closer.start();
      try (var frame =
          new PreparedByteScoring(
              runtime,
              fenced,
              runtime.newCallScope(),
              query,
              count,
              VectorSimilarityFunction.DOT_PRODUCT)) {
        float[] output = new float[count];
        frame.scoreContiguous(leaf, leaf.getByteVectorValues("v"), 0, count, output, 0);
        for (int i = 0; i < count; i++)
          assertEquals(
              Float.floatToRawIntBits(VectorSimilarityFunction.DOT_PRODUCT.compare(query, rows[i])),
              Float.floatToRawIntBits(output[i]));
      } finally {
        retired.countDown();
        closer.join(30000);
        reader.close();
      }
      assertFalse("close/merge worker did not finish", closer.isAlive());
      if (failure.get() != null)
        throw new AssertionError("concurrent close/merge failed", failure.get());
      assertEquals(0, leaf.getRefCount());
      assertEquals(0L, runtime.inventory().execution().activeScopes());
    }
  }

  private static final class Allocator implements NativeAllocator {
    final AtomicInteger allocations = new AtomicInteger();
    final AtomicInteger releases = new AtomicInteger();
    boolean failClose;

    @Override
    public Allocation allocate(long bytes) {
      allocations.incrementAndGet();
      return new Allocation() {
        final ByteBuffer buffer = ByteBuffer.allocate(Math.toIntExact(bytes));
        boolean closed;

        @Override
        public ByteBuffer buffer() {
          return buffer;
        }

        @Override
        public long capacity() {
          return bytes;
        }

        @Override
        public boolean isAlive() {
          return !closed;
        }

        @Override
        public void close() {
          if (failClose) throw new IllegalStateException("injected close failure");
          if (!closed) {
            closed = true;
            releases.incrementAndGet();
          }
        }
      };
    }
  }

  @FunctionalInterface
  private interface Execute {
    AccelerationProvider.Result execute(AccelerationProvider.Batch batch) throws IOException;
  }

  private static AccelerationProvider.Kernel kernel(Execute execute) {
    return new AccelerationProvider.Kernel() {
      @Override
      public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
        return new AccelerationProvider.Estimate(true, 0, 0);
      }

      @Override
      public AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
          throws IOException {
        return execute.execute(batch);
      }
    };
  }

  private static RequestMemoryScope scope(AccelerationRuntime runtime, Allocator allocator) {
    MemoryLedger.Owner owner = runtime.currentOwner();
    return RequestMemoryScope.threadConfined(
        runtime.ledger(), owner == null ? OWNER : owner, allocator);
  }

  public void testQueryPreparedOnceAndBatchesReuseStorageWithExactReaderOrder() throws Exception {
    for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
      byte[] query = new byte[33];
      random().nextBytes(query);
      byte[] original = query.clone();
      byte[][] documents = new byte[7][query.length];
      for (byte[] row : documents) random().nextBytes(row);
      Allocator allocator = new Allocator();
      try (AccelerationRuntime runtime = runtime(1 << 20);
          PreparedByteScoring frame =
              new PreparedByteScoring(
                  runtime,
                  kernel(JavaKernelOracle::execute),
                  scope(runtime, allocator),
                  query,
                  documents.length,
                  similarity)) {
        Arrays.fill(query, (byte) 0); // Caller mutation cannot change the prepared query.
        long held = runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE);
        int allocations = allocator.allocations.get();
        float[] output = new float[9];
        for (int count = 0; count <= documents.length; count++) {
          Arrays.fill(output, 42);
          float maximum = Float.NEGATIVE_INFINITY;
          for (int i = 0; i < count; i++)
            maximum = Math.max(maximum, similarity.compare(original, documents[i]));
          assertEquals(
              Float.floatToRawIntBits(maximum),
              Float.floatToRawIntBits(frame.score(documents, 0, count, output, 1)));
          for (int i = 0; i < count; i++)
            assertEquals(
                Float.floatToRawIntBits(similarity.compare(original, documents[i])),
                Float.floatToRawIntBits(output[i + 1]));
          assertEquals(42f, output[0], 0);
          assertEquals(42f, output[count + 1], 0);
          assertEquals(allocations, allocator.allocations.get());
          assertEquals(
              held,
              runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
        }
        ByteVectorValues values = ByteVectorValues.fromBytes(List.of(documents), original.length);
        int[] ordinals = {6, 0, 6, 2};
        frame.score(values, ordinals, ordinals.length, output, 2);
        for (int i = 0; i < ordinals.length; i++)
          assertEquals(
              Float.floatToRawIntBits(similarity.compare(original, documents[ordinals[i]])),
              Float.floatToRawIntBits(output[i + 2]));
      }
      assertEquals(allocator.allocations.get(), allocator.releases.get());
    }
  }

  public void testFailureAfterPrivateWriteCannotPublishOrReplay() throws Exception {
    Allocator allocator = new Allocator();
    AtomicInteger calls = new AtomicInteger();
    try (AccelerationRuntime runtime = runtime(8192);
        PreparedByteScoring frame =
            new PreparedByteScoring(
                runtime,
                kernel(
                    batch -> {
                      calls.incrementAndGet();
                      batch.output().bytes().putFloat(0, 123);
                      return new AccelerationProvider.Result(
                          AccelerationProvider.Status.UNSUPPORTED, 0, 0);
                    }),
                scope(runtime, allocator),
                new byte[] {1},
                2,
                VectorSimilarityFunction.DOT_PRODUCT)) {
      float[] output = {42, 43};
      expectThrows(IOException.class, () -> frame.score(new byte[][] {{2}}, 0, 1, output, 0));
      assertArrayEquals(new float[] {42, 43}, output, 0);
      expectThrows(
          IllegalStateException.class, () -> frame.score(new byte[][] {{2}}, 0, 1, output, 0));
      assertEquals(1, calls.get());
    }
  }

  public void testCancellationAfterKernelSuccessDoesNotPublish() throws Exception {
    Allocator allocator = new Allocator();
    try (AccelerationRuntime runtime = runtime(8192)) {
      RequestMemoryScope scope = scope(runtime, allocator);
      try (PreparedByteScoring frame =
          new PreparedByteScoring(
              runtime,
              kernel(
                  batch -> {
                    AccelerationProvider.Result result = JavaKernelOracle.execute(batch);
                    scope.cancel();
                    return result;
                  }),
              scope,
              new byte[] {1},
              2,
              VectorSimilarityFunction.DOT_PRODUCT)) {
        float[] output = {42, 43};
        expectThrows(IOException.class, () -> frame.score(new byte[][] {{2}}, 0, 1, output, 0));
        assertArrayEquals(new float[] {42, 43}, output, 0);
      }
      assertEquals(
          0L, runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
    }
  }

  public void testThreadAndRequestOwnerChecksPrecedePublicationAndClose() throws Exception {
    Allocator allocator = new Allocator();
    try (AccelerationRuntime runtime = runtime(8192);
        PreparedByteScoring frame =
            new PreparedByteScoring(
                runtime,
                kernel(JavaKernelOracle::execute),
                scope(runtime, allocator),
                new byte[] {1},
                2,
                VectorSimilarityFunction.DOT_PRODUCT)) {
      float[] output = {42};
      Thread foreign =
          new Thread(
              () -> {
                expectThrows(
                    IllegalStateException.class,
                    () -> frame.score(new byte[][] {{2}}, 0, 1, output, 0));
                expectThrows(IllegalStateException.class, frame::close);
              });
      foreign.start();
      foreign.join(10000);
      assertFalse(foreign.isAlive());
      var context = new AccelerationRuntime.TaskContext(OWNER, Long.MAX_VALUE);
      try (var binding = runtime.enterTask(context)) {
        assertNotNull(binding);
        expectThrows(IOException.class, () -> frame.score(new byte[][] {{2}}, 0, 1, output, 0));
      }
      assertEquals(42f, output[0], 0);
      frame.score(new byte[][] {{2}}, 0, 1, output, 0);
    }
  }

  public void testAdmissionRollbackAndFailedCloseKeepChargesUntilExplicitRecovery()
      throws Exception {
    Allocator allocator = new Allocator();
    try (AccelerationRuntime runtime = runtime(128)) {
      expectThrows(
          RequestMemoryScope.AdmissionException.class,
          () ->
              new PreparedByteScoring(
                  runtime,
                  kernel(JavaKernelOracle::execute),
                  scope(runtime, allocator),
                  new byte[] {1},
                  2,
                  VectorSimilarityFunction.DOT_PRODUCT));
      assertEquals(
          0L, runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE).longValue());
    }
    try (AccelerationRuntime runtime = runtime(8192)) {
      PreparedByteScoring frame =
          new PreparedByteScoring(
              runtime,
              kernel(JavaKernelOracle::execute),
              scope(runtime, allocator),
              new byte[] {1},
              2,
              VectorSimilarityFunction.DOT_PRODUCT);
      allocator.failClose = true;
      expectThrows(IllegalStateException.class, frame::close);
      assertTrue(runtime.ledger().snapshot().failedReclamations() > 0);
      allocator.failClose = false;
      frame.close(); // Idempotent retirement does not silently retry failures.
      assertTrue(runtime.ledger().snapshot().failedReclamations() > 0);
      frame.retryFailedReclamations();
      assertEquals(0L, runtime.ledger().snapshot().failedReclamations());
      expectThrows(
          IllegalStateException.class,
          () -> frame.score(new byte[][] {{2}}, 0, 1, new float[1], 0));
    }
    assertEquals(allocator.allocations.get(), allocator.releases.get());
  }
}
