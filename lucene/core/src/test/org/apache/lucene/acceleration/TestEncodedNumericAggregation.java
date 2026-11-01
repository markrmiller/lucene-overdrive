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

import java.nio.file.Files;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FilterNumericDocValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.internal.acceleration.EncodedNumericValues;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.FixedBitSet;

public class TestEncodedNumericAggregation extends LuceneTestCase {
  private static IndexWriterConfig config() {
    return new IndexWriterConfig()
        .setCodec(TestUtil.alwaysDocValuesFormat(new Lucene90DocValuesFormat()))
        .setMergePolicy(NoMergePolicy.INSTANCE)
        .setMaxBufferedDocs(10000);
  }

  private static long[] values(int bits, int rows) {
    long[] result = new long[rows];
    Random random = new Random(911);
    for (int i = 0; i < rows; i++)
      result[i] =
          bits == 32 ? -1000 + 3 * Integer.toUnsignedLong(random.nextInt()) : random.nextLong();
    result[0] = bits == 32 ? -1000 : Long.MIN_VALUE;
    result[1] = bits == 32 ? -997 : Long.MAX_VALUE;
    result[2] = bits == 32 ? -1000 + 3 * 0xffffffffL : 0;
    return result;
  }

  private static void write(Directory directory, long[] values, boolean missing, boolean delete)
      throws Exception {
    try (var writer = new IndexWriter(directory, config())) {
      for (int i = 0; i < values.length; i++) {
        var doc = new Document();
        doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
        if (!missing || i != 3) doc.add(new NumericDocValuesField("n", values[i]));
        writer.addDocument(doc);
      }
      if (delete) writer.deleteDocuments(new Term("id", "17"));
    }
  }

  private static BlockOperations.Aggregate expected(
      long[] values,
      int from,
      int to,
      FixedBitSet selection,
      boolean delete,
      long lower,
      long upper) {
    long count = 0, sum = 0, minimum = 0, maximum = 0;
    boolean overflow = false;
    for (int doc = from; doc < to; doc++) {
      if (delete && doc == 17 || selection != null && !selection.get(doc)) continue;
      long value = values[doc];
      if (value < lower || value > upper) continue;
      if (count == 0 || value < minimum) minimum = value;
      if (count == 0 || value > maximum) maximum = value;
      long next = sum + value;
      overflow |= ((sum ^ next) & (value ^ next)) < 0;
      sum = next;
      count++;
    }
    return new BlockOperations.Aggregate(count, sum, minimum, maximum, count != 0, overflow);
  }

  private static AccelerationRuntime nativeRuntime() throws Exception {
    return AccelerationRuntime.forQualification(
        TestAccelerationRuntime.config(
            AccelerationConfig.Mode.AUTO,
            TestAccelerationRuntime.nativePack(),
            Set.of(),
            Set.of(),
            1));
  }

  public void testSelectionWordsMatchBitwiseOracleAcrossOffsetsAndTails() {
    for (int length : new int[] {1, 63, 64, 65, 131}) {
      for (int pattern = 0; pattern < 3; pattern++) {
        var selection = new FixedBitSet(length);
        var random = new Random(317);
        for (int doc = 0; doc < length; doc++)
          if (pattern == 1 || pattern == 2 && random.nextBoolean()) selection.set(doc);
        // Poison unused storage bits: the borrowed interval must still have a clean tail.
        if ((length & 63) != 0)
          selection.getBits()[selection.getBits().length - 1] |= -1L << length;
        for (int from = 0; from <= length; from++) {
          for (int count = 0; count <= Math.min(64, length - from); count++) {
            long expected = 0;
            for (int i = 0; i < count; i++) if (selection.get(from + i)) expected |= 1L << i;
            assertEquals(expected, EncodedNumericAggregation.selectionWord(selection, from, count));
          }
        }
      }
    }
  }

  public void testCodecSeamAdmitsOnlyDenseUnblockedAffine32And64() throws Exception {
    for (int bits : new int[] {32, 64}) {
      try (var directory = new ByteBuffersDirectory()) {
        long[] values = values(bits, 1025);
        write(directory, values, false, false);
        try (var reader = DirectoryReader.open(directory)) {
          var numeric = reader.leaves().getFirst().reader().getNumericDocValues("n");
          assertTrue(numeric instanceof EncodedNumericValues);
          var source = ((EncodedNumericValues) numeric).encodedSource();
          assertNotNull(source);
          assertEquals(bits, source.bits());
          assertEquals(values.length, source.rows());
          assertTrue(source.length() >= (long) source.rows() * (bits / 8));
          if (bits == 32) {
            assertEquals(3, source.multiplier());
            assertEquals(-1000, source.base());
          }
          assertEquals(-1, numeric.docID());
          for (int doc = 0; doc < values.length; doc++) {
            assertTrue(numeric.advanceExact(doc));
            assertEquals(values[doc], numeric.longValue());
          }
        }
      }
    }
    for (boolean missing : new boolean[] {false, true}) {
      try (var directory = new ByteBuffersDirectory()) {
        long[] values = values(64, 1025);
        if (!missing)
          for (int i = 0; i < values.length; i++) values[i] = i % 2 == 0 ? 0 : Long.MAX_VALUE;
        write(directory, values, missing, false);
        try (var reader = DirectoryReader.open(directory)) {
          var numeric = reader.leaves().getFirst().reader().getNumericDocValues("n");
          assertFalse(
              "table/sparse representations retain their original path",
              numeric instanceof EncodedNumericValues);
        }
      }
    }
  }

  public void testMappedAggregatePreservesAffineSelectionDeletedTailAndOverflow() throws Exception {
    try (var runtime = nativeRuntime()) {
      assertTrue(runtime.supports(Capability.DOCVALUES_AGGREGATE));
      for (int bits : new int[] {32, 64}) {
        for (boolean delete : new boolean[] {false, true}) {
          long[] values = values(bits, 1025);
          try (var directory = new MMapDirectory(createTempDir("encoded-aggregate"))) {
            write(directory, values, false, delete);
            try (var reader = DirectoryReader.open(directory)) {
              var leaf = reader.leaves().getFirst().reader();
              assertNotNull(((EncodedNumericValues) leaf.getNumericDocValues("n")).encodedSource());
              FixedBitSet selection = new FixedBitSet(values.length);
              for (int doc = 0; doc < values.length; doc++) if (doc % 3 != 0) selection.set(doc);
              long before =
                  runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls();
              long critical = runtime.criticalCalls().get(Capability.DOCVALUES_AGGREGATE).calls();
              assertEquals(
                  expected(values, 0, values.length, null, delete, Long.MIN_VALUE, Long.MAX_VALUE),
                  EncodedNumericAggregation.tryAggregate(
                      runtime, leaf, "n", 0, values.length, null, Long.MIN_VALUE, Long.MAX_VALUE));
              assertEquals(
                  expected(values, 3, 1001, selection, delete, Long.MIN_VALUE, Long.MAX_VALUE),
                  EncodedNumericAggregation.tryAggregate(
                      runtime, leaf, "n", 3, 1001, selection, Long.MIN_VALUE, Long.MAX_VALUE));
              assertEquals(
                  expected(values, 0, values.length, null, delete, 1, -1),
                  EncodedNumericAggregation.tryAggregate(
                      runtime, leaf, "n", 0, values.length, null, 1, -1));
              for (int from = 0; from < 64; from++) {
                int to = from + 129;
                assertEquals(
                    expected(values, from, to, selection, delete, Long.MIN_VALUE, Long.MAX_VALUE),
                    EncodedNumericAggregation.tryAggregate(
                        runtime, leaf, "n", from, to, selection, Long.MIN_VALUE, Long.MAX_VALUE));
              }
              assertEquals(
                  before + 67,
                  runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls());
              assertEquals(
                  "mapped inputs must always use ordinary downcalls",
                  critical,
                  runtime.criticalCalls().get(Capability.DOCVALUES_AGGREGATE).calls());
              assertEquals(
                  0, runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).errors());
            }
          }
        }
      }
    }
  }

  public void testDocValuesUpdatesKeepEachEncodedGenerationIndependent() throws Exception {
    try (var runtime = nativeRuntime();
        var directory = new MMapDirectory(createTempDir("encoded-update"));
        var writer = new IndexWriter(directory, config())) {
      long[] values = values(32, 1025);
      for (int i = 0; i < values.length; i++) {
        var doc = new Document();
        doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
        doc.add(new NumericDocValuesField("n", values[i]));
        writer.addDocument(doc);
      }
      writer.commit();
      try (var old = DirectoryReader.open(writer)) {
        writer.updateNumericDocValue(new Term("id", "1"), "n", Long.MAX_VALUE);
        writer.commit();
        try (var next = DirectoryReader.openIfChanged(old, writer)) {
          assertNotNull(next);
          var oldLeaf = old.leaves().getFirst().reader();
          var nextLeaf = next.leaves().getFirst().reader();
          assertEquals(
              expected(values, 0, values.length, null, false, Long.MIN_VALUE, Long.MAX_VALUE),
              EncodedNumericAggregation.tryAggregate(
                  runtime, oldLeaf, "n", 0, values.length, null, Long.MIN_VALUE, Long.MAX_VALUE));
          long[] updated = values.clone();
          updated[1] = Long.MAX_VALUE;
          assertEquals(
              expected(updated, 0, updated.length, null, false, Long.MIN_VALUE, Long.MAX_VALUE),
              EncodedNumericAggregation.tryAggregate(
                  runtime, nextLeaf, "n", 0, updated.length, null, Long.MIN_VALUE, Long.MAX_VALUE));
          assertEquals(
              32, ((EncodedNumericValues) oldLeaf.getNumericDocValues("n")).encodedSource().bits());
          assertEquals(
              64,
              ((EncodedNumericValues) nextLeaf.getNumericDocValues("n")).encodedSource().bits());
        }
      }
    }
  }

  public void testUnsupportedMappingOrLayoutDoesNotAllocateOrCallNative() throws Exception {
    try (var runtime = nativeRuntime()) {
      for (boolean missing : new boolean[] {false, true}) {
        try (var directory = new ByteBuffersDirectory()) {
          write(directory, values(64, 1025), missing, false);
          try (var reader = DirectoryReader.open(directory)) {
            var leaf = reader.leaves().getFirst().reader();
            var before = runtime.inventory().memory();
            long calls =
                runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls();
            assertNull(
                EncodedNumericAggregation.tryAggregate(
                    runtime, leaf, "n", 0, leaf.maxDoc(), null, Long.MIN_VALUE, Long.MAX_VALUE));
            assertEquals(before, runtime.inventory().memory());
            assertEquals(
                calls,
                runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls());
          }
        }
      }
    }
  }

  public void testNativeOnlyPolicyDoesNotAdmitMappedCodecBytes() throws Exception {
    var pack = TestAccelerationRuntime.nativePack();
    var policy = TrustedPathFixtures.file("encoded-native-only-policy");
    String row =
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.DOCVALUES_AGGREGATE.id
            + "\t32\t1025\t1025\t"
            + Long.MIN_VALUE
            + "\t"
            + Long.MAX_VALUE
            + "\tnative\t10000\tunit-fixture-no-performance-approval\n";
    Files.writeString(policy, row);
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    String key = "lucene.acceleration.native.qualificationPath";
    String previous = System.getProperty(key);
    System.setProperty(key, policy.toString());
    try (var runtime =
            new AccelerationRuntime(
                TestAccelerationRuntime.config(
                    AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1));
        var directory = new MMapDirectory(createTempDir("encoded-unqualified"))) {
      assertTrue(runtime.supports(Capability.DOCVALUES_AGGREGATE));
      write(directory, values(32, 1025), false, false);
      try (var reader = DirectoryReader.open(directory)) {
        var leaf = reader.leaves().getFirst().reader();
        var before = runtime.inventory();
        assertNull(
            EncodedNumericAggregation.tryAggregate(
                runtime, leaf, "n", 0, leaf.maxDoc(), null, Long.MIN_VALUE, Long.MAX_VALUE));
        assertEquals(before.memory(), runtime.inventory().memory());
        assertEquals(
            0, runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls());
      }
    } finally {
      if (previous == null) System.clearProperty(key);
      else System.setProperty(key, previous);
    }
  }

  public void testMappedSystemProfileAdmitsPhysicalNativeFramesAndExactParameters()
      throws Exception {
    var pack = TestAccelerationRuntime.nativePack();
    var policy = TrustedPathFixtures.file("encoded-mapped-policy");
    String row =
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.DOCVALUES_AGGREGATE.id
            + "\t32\t1025\t1025\t"
            + Long.MIN_VALUE
            + "\t"
            + Long.MAX_VALUE
            + "\tmapped\t10000\tunit-fixture-no-performance-approval\n";
    Files.writeString(policy, row);
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    String key = "lucene.acceleration.native.qualificationPath";
    String previous = System.getProperty(key);
    System.setProperty(key, policy.toString());
    try (var runtime =
            new AccelerationRuntime(
                TestAccelerationRuntime.config(
                    AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1));
        var directory = new MMapDirectory(createTempDir("encoded-system-mapped"))) {
      assertTrue(runtime.supports(Capability.DOCVALUES_AGGREGATE));
      long[] values = values(32, 1025);
      write(directory, values, false, false);
      try (var reader = DirectoryReader.open(directory)) {
        var leaf = reader.leaves().getFirst().reader();
        long critical = runtime.criticalCalls().get(Capability.DOCVALUES_AGGREGATE).calls();
        assertEquals(
            expected(values, 0, values.length, null, false, Long.MIN_VALUE, Long.MAX_VALUE),
            EncodedNumericAggregation.tryAggregate(
                runtime, leaf, "n", 0, values.length, null, Long.MIN_VALUE, Long.MAX_VALUE));
        assertEquals(
            1, runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls());
        assertEquals(
            0, runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).stagedBytes());
        assertEquals(critical, runtime.criticalCalls().get(Capability.DOCVALUES_AGGREGATE).calls());
        var before = runtime.inventory();
        assertNull(
            EncodedNumericAggregation.tryAggregate(
                runtime, leaf, "n", 0, values.length, null, Long.MIN_VALUE + 1, Long.MAX_VALUE));
        assertNull(
            EncodedNumericAggregation.tryAggregate(
                runtime, leaf, "n", 1, values.length, null, Long.MIN_VALUE, Long.MAX_VALUE));
        assertEquals(before.memory(), runtime.inventory().memory());
        assertEquals(before.counters(), runtime.inventory().counters());
      }
    } finally {
      if (previous == null) System.clearProperty(key);
      else System.setProperty(key, previous);
    }
  }

  private static final String JSON_RANGE_CONTEXT = "solr.json-range.full-live.v1";

  private static AccelerationRuntime scopedRuntime() throws Exception {
    var pack = TestAccelerationRuntime.nativePack();
    var policy = TrustedPathFixtures.file("encoded-json-range-context");
    Files.writeString(
        policy,
        AccelerationRuntime.BRIDGE_BUILD
            + "\t"
            + PackLoader.verify(pack).digest()
            + "\tx86-avx2-os-xsave-ymm\t"
            + Capability.DOCVALUES_AGGREGATE.id
            + "\t32\t1025\t1025\t"
            + Long.MIN_VALUE
            + "\t"
            + Long.MAX_VALUE
            + "\tmapped\t10000\tTEST-ONLY-json-range-context\t"
            + JSON_RANGE_CONTEXT
            + "\n");
    Files.setPosixFilePermissions(policy, PosixFilePermissions.fromString("rw-------"));
    String key = "lucene.acceleration.native.qualificationPath";
    String previous = System.getProperty(key);
    System.setProperty(key, policy.toString());
    try {
      return new AccelerationRuntime(
          TestAccelerationRuntime.config(
              AccelerationConfig.Mode.AUTO, pack, Set.of(), Set.of(), 1));
    } finally {
      if (previous == null) System.clearProperty(key);
      else System.setProperty(key, previous);
    }
  }

  public void testScopedMappedAggregatePublishesExactResultAndRefundsAtRuntimeClose()
      throws Exception {
    var runtime = scopedRuntime();
    try (runtime;
        var directory = new MMapDirectory(createTempDir("encoded-scoped-positive"))) {
      assertTrue(runtime.supports(Capability.DOCVALUES_AGGREGATE, JSON_RANGE_CONTEXT));
      assertFalse(runtime.supports(Capability.DOCVALUES_AGGREGATE, "generic"));
      long[] values = values(32, 1025);
      write(directory, values, false, false);
      try (var reader = DirectoryReader.open(directory)) {
        var leaf = reader.leaves().getFirst().reader();
        int refs = leaf.getRefCount();
        long critical = runtime.criticalCalls().get(Capability.DOCVALUES_AGGREGATE).calls();
        assertNotNull(((EncodedNumericValues) leaf.getNumericDocValues("n")).encodedSource());
        assertEquals(
            expected(values, 0, values.length, null, false, Long.MIN_VALUE, Long.MAX_VALUE),
            EncodedNumericAggregation.tryAggregate(
                runtime,
                leaf,
                "n",
                0,
                values.length,
                null,
                Long.MIN_VALUE,
                Long.MAX_VALUE,
                JSON_RANGE_CONTEXT));
        assertEquals(refs, leaf.getRefCount());
        var counters = runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE);
        assertEquals(1, counters.nativeCalls());
        assertEquals(0, counters.javaCalls());
        assertEquals(0, counters.errors());
        assertEquals(0, counters.stagedBytes());
        assertEquals(critical, runtime.criticalCalls().get(Capability.DOCVALUES_AGGREGATE).calls());
        var before = runtime.inventory();
        assertNull(
            EncodedNumericAggregation.tryAggregate(
                runtime,
                leaf,
                "n",
                0,
                values.length,
                null,
                Long.MIN_VALUE + 1,
                Long.MAX_VALUE,
                JSON_RANGE_CONTEXT));
        assertEquals(before.memory(), runtime.inventory().memory());
        assertEquals(before.execution(), runtime.inventory().execution());
        assertEquals(before.counters(), runtime.inventory().counters());
      }
    }
    assertEquals(0, runtime.inventory().execution().activeScopes());
    for (long bytes : runtime.inventory().memory().charged().values()) assertEquals(0, bytes);
  }

  public void testScopedOnlyPolicyRejectsOtherConsumersBeforeCodecAccessOrNativeAllocation()
      throws Exception {
    try (var runtime = scopedRuntime();
        var directory = new MMapDirectory(createTempDir("encoded-context-decline"))) {
      write(directory, values(32, 1025), false, false);
      try (var reader = DirectoryReader.open(directory)) {
        var original = reader.leaves().getFirst().reader();
        original.incRef();
        try (var leaf =
            new FilterLeafReader(original) {
              @Override
              public NumericDocValues getNumericDocValues(String field) {
                throw new AssertionError("unqualified consumer must decline before codec access");
              }

              @Override
              public CacheHelper getCoreCacheHelper() {
                return in.getCoreCacheHelper();
              }

              @Override
              public CacheHelper getReaderCacheHelper() {
                return in.getReaderCacheHelper();
              }
            }) {
          var before = runtime.inventory();
          int refs = leaf.getRefCount();
          assertNull(
              EncodedNumericAggregation.tryAggregate(
                  runtime, leaf, "n", 0, leaf.maxDoc(), null, Long.MIN_VALUE, Long.MAX_VALUE));
          for (String context :
              new String[] {"generic", "solr.frange.full-live.v1", "unknown-context"})
            assertNull(
                EncodedNumericAggregation.tryAggregate(
                    runtime,
                    leaf,
                    "n",
                    0,
                    leaf.maxDoc(),
                    null,
                    Long.MIN_VALUE,
                    Long.MAX_VALUE,
                    context));
          assertEquals(refs, leaf.getRefCount());
          assertEquals(before.memory(), runtime.inventory().memory());
          assertEquals(before.execution(), runtime.inventory().execution());
          assertEquals(before.counters(), runtime.inventory().counters());
        }
      }
    }
  }

  public void testScopedReaderPinSurvivesCloseThroughNativePublication() throws Exception {
    var runtime = scopedRuntime();
    try (runtime;
        var directory = new MMapDirectory(createTempDir("encoded-scoped-close"))) {
      long[] values = values(32, 1025);
      write(directory, values, false, false);
      var owner = DirectoryReader.open(directory);
      var original = owner.leaves().getFirst().reader();
      var source = ((EncodedNumericValues) original.getNumericDocValues("n")).encodedSource();
      assertNotNull(source);
      CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
      original.incRef();
      var wrapper =
          new FilterLeafReader(original) {
            @Override
            public NumericDocValues getNumericDocValues(String field) throws java.io.IOException {
              return new BlockingValues(in.getNumericDocValues(field), source, entered, release);
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
      FutureTask<BlockOperations.Aggregate> task =
          new FutureTask<>(
              () ->
                  EncodedNumericAggregation.tryAggregate(
                      runtime,
                      wrapper,
                      "n",
                      0,
                      values.length,
                      null,
                      Long.MIN_VALUE,
                      Long.MAX_VALUE,
                      JSON_RANGE_CONTEXT));
      Thread worker = new Thread(task, "encoded-scoped-aggregate-close");
      worker.start();
      try {
        assertTrue(entered.await(30, TimeUnit.SECONDS));
        assertEquals(2, wrapper.getRefCount());
        owner.close();
        wrapper.close();
        assertEquals(1, wrapper.getRefCount());
        release.countDown();
        assertEquals(
            expected(values, 0, values.length, null, false, Long.MIN_VALUE, Long.MAX_VALUE),
            task.get(30, TimeUnit.SECONDS));
        assertEquals(0, wrapper.getRefCount());
        assertEquals(
            1, runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE).nativeCalls());
      } finally {
        release.countDown();
        worker.join(30000);
        assertFalse("scoped aggregate worker must finish", worker.isAlive());
        if (owner.getRefCount() > 0) owner.close();
        if (wrapper.getRefCount() > 0) wrapper.close();
      }
    }
    assertEquals(0, runtime.inventory().execution().activeScopes());
    for (long bytes : runtime.inventory().memory().charged().values()) assertEquals(0, bytes);
  }

  public void testBorrowedWrapperPinKeepsRawOwnerAliveThroughMappedAggregate() throws Exception {
    try (var runtime = nativeRuntime();
        var directory = new MMapDirectory(createTempDir("encoded-borrowed-wrapper-close"))) {
      long[] data = values(32, 1025);
      write(directory, data, false, false);
      var owner = DirectoryReader.open(directory);
      var backing = owner.leaves().getFirst().reader();
      var source = ((EncodedNumericValues) backing.getNumericDocValues("n")).encodedSource();
      assertNotNull(source);
      CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
      // Borrowing view: the independent DirectoryReader owns the only backing reference.
      var view =
          new FilterLeafReader(backing) {
            @Override
            public NumericDocValues getNumericDocValues(String field) throws java.io.IOException {
              return new BlockingValues(in.getNumericDocValues(field), source, entered, release);
            }

            @Override
            public CacheHelper getCoreCacheHelper() {
              return in.getCoreCacheHelper();
            }

            @Override
            public CacheHelper getReaderCacheHelper() {
              return in.getReaderCacheHelper();
            }

            @Override
            protected void doClose() {} // This view must never release the independent owner's ref.
          };
      FutureTask<BlockOperations.Aggregate> task =
          new FutureTask<>(
              () ->
                  EncodedNumericAggregation.tryAggregate(
                      runtime, view, "n", 0, data.length, null, Long.MIN_VALUE, Long.MAX_VALUE));
      Thread worker = new Thread(task, "encoded-borrowed-view-close");
      worker.start();
      try {
        assertTrue(
            "mapped acquisition must reach the publication barrier",
            entered.await(30, TimeUnit.SECONDS));
        assertEquals(2, view.getRefCount());
        owner.close();
        assertEquals(
            "a wrapper pin alone must not let the independent backing owner close",
            1,
            backing.getRefCount());
        release.countDown();
        assertEquals(
            expected(data, 0, data.length, null, false, Long.MIN_VALUE, Long.MAX_VALUE),
            task.get(30, TimeUnit.SECONDS));
        assertEquals(0, backing.getRefCount());
        assertEquals(1, view.getRefCount());
        var counters = runtime.inventory().counters().get(Capability.DOCVALUES_AGGREGATE);
        assertEquals(1, counters.nativeCalls());
        assertEquals(0, counters.errors());
      } finally {
        release.countDown();
        worker.join(30_000);
        assertFalse("borrowed-view worker must finish", worker.isAlive());
        owner.close();
        view.close();
      }
    }
  }

  private static final class BlockingValues extends FilterNumericDocValues
      implements EncodedNumericValues {
    private final Source source;
    private final CountDownLatch entered, release;

    BlockingValues(
        NumericDocValues values, Source source, CountDownLatch entered, CountDownLatch release) {
      super(values);
      this.source = source;
      this.entered = entered;
      this.release = release;
    }

    @Override
    public Source encodedSource() {
      entered.countDown();
      try {
        if (!release.await(30, TimeUnit.SECONDS))
          throw new AssertionError("reader-close fixture did not release");
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new AssertionError(interrupted);
      }
      return source;
    }
  }

  public void testReaderCloseDefersMappingRetirementUntilPinnedAggregateCompletes()
      throws Exception {
    try (var runtime = nativeRuntime();
        var directory = new MMapDirectory(createTempDir("encoded-close"))) {
      long[] values = values(64, 1025);
      write(directory, values, false, false);
      var owner = DirectoryReader.open(directory);
      var original = owner.leaves().getFirst().reader();
      var source = ((EncodedNumericValues) original.getNumericDocValues("n")).encodedSource();
      assertNotNull(source);
      CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
      original.incRef();
      var wrapper =
          new FilterLeafReader(original) {
            @Override
            public NumericDocValues getNumericDocValues(String field) throws java.io.IOException {
              return new BlockingValues(in.getNumericDocValues(field), source, entered, release);
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
      FutureTask<BlockOperations.Aggregate> task =
          new FutureTask<>(
              () ->
                  EncodedNumericAggregation.tryAggregate(
                      runtime,
                      wrapper,
                      "n",
                      0,
                      values.length,
                      null,
                      Long.MIN_VALUE,
                      Long.MAX_VALUE));
      Thread worker = new Thread(task, "encoded-aggregate-close");
      worker.start();
      try {
        assertTrue(entered.await(30, TimeUnit.SECONDS));
        assertEquals(2, wrapper.getRefCount());
        owner.close();
        wrapper.close();
        assertEquals(1, wrapper.getRefCount());
        release.countDown();
        assertEquals(
            expected(values, 0, values.length, null, false, Long.MIN_VALUE, Long.MAX_VALUE),
            task.get(30, TimeUnit.SECONDS));
        assertEquals(0, wrapper.getRefCount());
      } finally {
        release.countDown();
        worker.join(30000);
        if (owner.getRefCount() > 0) owner.close();
        if (wrapper.getRefCount() > 0) wrapper.close();
      }
    }
  }
}
