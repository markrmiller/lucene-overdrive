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
package org.apache.lucene.index;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.RamUsageTester;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.packed.PackedInts;
import org.apache.lucene.util.packed.PackedLongValues;

public class TestOrdinalMap extends LuceneTestCase {
  private static final Field ORDINAL_MAP_OWNER_FIELD;

  static {
    try {
      ORDINAL_MAP_OWNER_FIELD = OrdinalMap.class.getDeclaredField("owner");
    } catch (Exception e) {
      throw new Error(e);
    }
  }

  private static final RamUsageTester.Accumulator ORDINAL_MAP_ACCUMULATOR =
      new RamUsageTester.Accumulator() {

        @Override
        public long accumulateObject(
            Object o,
            long shallowSize,
            java.util.Map<Field, Object> fieldValues,
            java.util.Collection<Object> queue) {
          if (o == LongValues.ZEROES
              || o == LongValues.IDENTITY
              || o == PackedInts.NullReader.forCount(PackedLongValues.DEFAULT_PAGE_SIZE)) {
            return 0L;
          }
          if (o instanceof OrdinalMap) {
            fieldValues = new HashMap<>(fieldValues);
            fieldValues.remove(ORDINAL_MAP_OWNER_FIELD);
          }
          return super.accumulateObject(o, shallowSize, fieldValues, queue);
        }
      };

  @SuppressForbidden(reason = "Inspect only Lucene-owned static fields for no-pack allocations")
  public void testStockBuildDoesNotAllocateSharedRegistry() throws Exception {
    OrdinalMap map = OrdinalMap.build(null, new TermsEnum[0], new long[0], PackedInts.DEFAULT);
    assertEquals(0, map.getValueCount());
    for (Field field : OrdinalMap.class.getDeclaredFields()) {
      if (Modifier.isStatic(field.getModifiers()) && Map.class.isAssignableFrom(field.getType())) {
        field.setAccessible(true);
        assertNull("stock initialization retained a shared registry", field.get(null));
      }
    }
  }

  public void testRamBytesUsed() throws IOException {
    Directory dir = newDirectory();
    IndexWriterConfig cfg =
        new IndexWriterConfig(new MockAnalyzer(random()))
            .setCodec(TestUtil.alwaysDocValuesFormat(TestUtil.getDefaultDocValuesFormat()));
    RandomIndexWriter iw = new RandomIndexWriter(random(), dir, cfg);
    final int maxDoc = TestUtil.nextInt(random(), 10, 1000);
    final int maxTermLength = TestUtil.nextInt(random(), 1, 4);
    for (int i = 0; i < maxDoc; ++i) {
      Document d = new Document();
      if (random().nextBoolean()) {
        d.add(
            new SortedDocValuesField(
                "sdv", new BytesRef(TestUtil.randomSimpleString(random(), maxTermLength))));
      }
      final int numSortedSet = random().nextInt(3);
      for (int j = 0; j < numSortedSet; ++j) {
        d.add(
            new SortedSetDocValuesField(
                "ssdv", new BytesRef(TestUtil.randomSimpleString(random(), maxTermLength))));
      }
      iw.addDocument(d);
      if (rarely()) {
        iw.getReader().close();
      }
    }
    iw.commit();
    DirectoryReader r = iw.getReader();
    SortedDocValues sdv = MultiDocValues.getSortedValues(r, "sdv");
    if (sdv instanceof MultiDocValues.MultiSortedDocValues) {
      OrdinalMap map = ((MultiDocValues.MultiSortedDocValues) sdv).mapping;
      assertEquals(RamUsageTester.ramUsed(map, ORDINAL_MAP_ACCUMULATOR), map.ramBytesUsed());
    }
    SortedSetDocValues ssdv = MultiDocValues.getSortedSetValues(r, "ssdv");
    if (ssdv instanceof MultiDocValues.MultiSortedSetDocValues) {
      OrdinalMap map = ((MultiDocValues.MultiSortedSetDocValues) ssdv).mapping;
      assertEquals(RamUsageTester.ramUsed(map, ORDINAL_MAP_ACCUMULATOR), map.ramBytesUsed());
    }
    iw.close();
    r.close();
    dir.close();
  }

  /**
   * Tests the case where one segment contains all of the global ords. In this case, we apply a
   * small optimization and hardcode the first segment indices and global ord deltas as all zeroes.
   */
  public void testOneSegmentWithAllValues() throws IOException {
    Directory dir = newDirectory();
    IndexWriterConfig cfg =
        new IndexWriterConfig(new MockAnalyzer(random()))
            .setCodec(TestUtil.alwaysDocValuesFormat(TestUtil.getDefaultDocValuesFormat()))
            .setMergePolicy(NoMergePolicy.INSTANCE);
    IndexWriter iw = new IndexWriter(dir, cfg);

    int numTerms = 1000;
    for (int i = 0; i < numTerms; ++i) {
      Document d = new Document();
      String term = String.valueOf(i);
      d.add(new SortedDocValuesField("sdv", new BytesRef(term)));
      iw.addDocument(d);
    }
    iw.forceMerge(1);

    for (int i = 0; i < 10; ++i) {
      Document d = new Document();
      String term = String.valueOf(random().nextInt(numTerms));
      d.add(new SortedDocValuesField("sdv", new BytesRef(term)));
      iw.addDocument(d);
    }
    iw.commit();

    DirectoryReader r = DirectoryReader.open(iw);
    SortedDocValues sdv = MultiDocValues.getSortedValues(r, "sdv");
    assertNotNull(sdv);
    assertTrue(sdv instanceof MultiDocValues.MultiSortedDocValues);

    // Check that the optimization kicks in.
    OrdinalMap map = ((MultiDocValues.MultiSortedDocValues) sdv).mapping;
    assertEquals(LongValues.ZEROES, map.firstSegments);
    assertEquals(LongValues.ZEROES, map.globalOrdDeltas);

    // Check the map's basic behavior.
    assertEquals(numTerms, (int) map.getValueCount());
    for (int i = 0; i < numTerms; i++) {
      assertEquals(0, map.getFirstSegmentNumber(i));
      assertEquals(i, map.getFirstSegmentOrd(i));
    }

    iw.close();
    r.close();
    dir.close();
  }

  /** The builder reserves and checks cancellation once per chunk, overshooting by under a chunk. */
  public void testBuildBudgetGrowsInBoundedChunks() throws IOException {
    int numTerms = TestUtil.nextInt(random(), 2000, 6000);
    try (Directory dir = newDirectory()) {
      try (IndexWriter iw = new IndexWriter(dir, newIndexWriterConfig())) {
        for (int i = 0; i < numTerms; ++i) {
          Document d = new Document();
          d.add(new SortedDocValuesField("sdv", new BytesRef(Integer.toString(i))));
          iw.addDocument(d);
        }
        iw.forceMerge(1);
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        SortedDocValues values = r.leaves().getFirst().reader().getSortedDocValues("sdv");
        MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(64L << 20, 0, 0), 64L << 20);
        AtomicInteger checks = new AtomicInteger();
        AtomicLong lastReserved = new AtomicLong();
        OrdinalMap map =
            OrdinalMap.buildManaged(
                r.getReaderCacheHelper().getKey(),
                new TermsEnum[] {values.termsEnum()},
                new long[] {values.getValueCount()},
                PackedInts.DEFAULT,
                () -> {
                  checks.incrementAndGet();
                  lastReserved.set(ledger.snapshot().reserved().get(MemoryLedger.Kind.HEAP));
                  return false;
                },
                ledger);
        assertEquals(numTerms, map.getValueCount());

        // One segment: one entry per segment ord plus two per global ord.
        long entries = 3L * numTerms;
        long chunks =
            (entries + OrdinalMap.BUDGET_CHUNK_ENTRIES - 1) / OrdinalMap.BUDGET_CHUNK_ENTRIES;
        // One check per chunk, plus the one before publication.
        assertEquals(chunks + 1, checks.get());
        long overshoot =
            lastReserved.get() - (131072 + 65536) - entries * OrdinalMap.BUDGET_ENTRY_BYTES;
        assertTrue(overshoot >= 0);
        assertTrue(overshoot < OrdinalMap.BUDGET_CHUNK_ENTRIES * OrdinalMap.BUDGET_ENTRY_BYTES);
        assertEquals(0L, (long) ledger.snapshot().reserved().get(MemoryLedger.Kind.HEAP));
        assertEquals(
            map.ramBytesUsed(), (long) ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP));
        map.close();

        AtomicInteger cancelAfter = new AtomicInteger(2);
        expectThrows(
            IOException.class,
            () ->
                OrdinalMap.buildManaged(
                    r.getReaderCacheHelper().getKey(),
                    new TermsEnum[] {values.termsEnum()},
                    new long[] {values.getValueCount()},
                    PackedInts.DEFAULT,
                    () -> cancelAfter.decrementAndGet() < 0,
                    ledger));
        assertEquals(0L, (long) ledger.snapshot().reserved().get(MemoryLedger.Kind.HEAP));
        assertEquals(0L, (long) ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP));
      }
    }
  }

  /** The default build is stock: no checkpoint or cancellation, and raw mappings. */
  public void testDefaultBuildIgnoresInterruptsAndBoundTasks() throws IOException {
    // An interrupt would close an NIOFS channel; this test is about the ordinal build only.
    try (Directory dir = new ByteBuffersDirectory()) {
      try (IndexWriter iw =
          new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
        for (int segment = 0; segment < 2; ++segment) {
          for (int i = 0; i < 10; ++i) {
            Document d = new Document();
            d.add(new SortedDocValuesField("sdv", new BytesRef(Integer.toString(segment * 5 + i))));
            iw.addDocument(d);
          }
          iw.commit();
        }
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        assertEquals(2, r.leaves().size());
        Thread.currentThread().interrupt();
        OrdinalMap interrupted;
        try {
          interrupted = buildDefault(r);
          assertTrue(Thread.currentThread().isInterrupted());
        } finally {
          Thread.interrupted();
        }

        AccelerationRuntime.TaskContext expired =
            new AccelerationRuntime.TaskContext(
                new MemoryLedger.Owner("expired-ordinal", "core", 1, "request"),
                System.nanoTime() - 1);
        expired.cancel();
        OrdinalMap bound;
        AccelerationRuntime.TaskBinding binding = AccelerationRuntime.system().enterTask(expired);
        try {
          bound = buildDefault(r);
        } finally {
          binding.close();
        }

        for (OrdinalMap map : new OrdinalMap[] {interrupted, bound}) {
          assertEquals(15, map.getValueCount());
          map.close();
          // Each segment gets the stored mapping itself, not a wrapper around it.
          assertNotSame(map.getGlobalOrds(0), map.getGlobalOrds(1));
          for (int i = 0; i < 2; ++i) {
            LongValues ords = map.getGlobalOrds(i);
            assertTrue(Arrays.stream(map.segmentToGlobalOrds).anyMatch(stored -> stored == ords));
          }
          assertEquals(0, map.getFirstSegmentNumber(0));
          assertEquals(14, map.getGlobalOrds(1).get(9));
        }
      }
    }
  }

  private static OrdinalMap buildDefault(DirectoryReader r) throws IOException {
    TermsEnum[] subs = new TermsEnum[r.leaves().size()];
    long[] weights = new long[subs.length];
    for (int i = 0; i < subs.length; ++i) {
      SortedDocValues values = r.leaves().get(i).reader().getSortedDocValues("sdv");
      subs[i] = values.termsEnum();
      weights[i] = values.getValueCount();
    }
    return OrdinalMap.build(r.getReaderCacheHelper().getKey(), subs, weights, PackedInts.DEFAULT);
  }
}
