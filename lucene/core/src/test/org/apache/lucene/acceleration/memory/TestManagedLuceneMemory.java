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
package org.apache.lucene.acceleration.memory;

import java.io.IOException;
import java.lang.ref.Reference;
import java.time.Duration;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.acceleration.AccelerationConfig;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.OrdinalMap;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.CollectionTerminatedException;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LRUQueryCache;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryCachingPolicy;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.SimpleCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.LongValues;

public class TestManagedLuceneMemory extends LuceneTestCase {
  private static class Counting extends SimpleCollector {
    int count;

    @Override
    public void collect(int doc) {
      count++;
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }
  }

  private static int collect(IndexSearcher searcher) throws IOException {
    return searcher.search(
        MatchAllDocsQuery.INSTANCE,
        new CollectorManager<Counting, Integer>() {
          @Override
          public Counting newCollector() {
            return new Counting();
          }

          @Override
          public Integer reduce(Collection<Counting> collectors) {
            int count = 0;
            for (Counting collector : collectors) count += collector.count;
            return count;
          }
        });
  }

  private static AccelerationRuntime runtime() throws Exception {
    return runtime(AccelerationConfig.StoragePolicy.CURRENT);
  }

  private static AccelerationRuntime runtime(AccelerationConfig.StoragePolicy storage)
      throws Exception {
    return new AccelerationRuntime(
        new AccelerationConfig(
            AccelerationConfig.Mode.OFF,
            null,
            Set.of(),
            Set.of(),
            16L << 20,
            32L << 20,
            1,
            false,
            storage,
            new AccelerationConfig.MemoryPolicy(16L << 20, 0, 0, 0, 0),
            new AccelerationConfig.ShadowPolicy(false, 1, 0, 0),
            new AccelerationConfig.Tuning(Map.of(), Map.of(), 0, 0)));
  }

  public void testQueryCacheEnrolledAndFinalReaderCloseDrains() throws Exception {
    AccelerationRuntime runtime = runtime();
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        for (int i = 0; i < 20; i++) writer.addDocument(new Document());
      }
      LRUQueryCache cache = new LRUQueryCache(10, 1L << 20, leaf -> true, 250, runtime);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);
        searcher.setQueryCache(cache);
        searcher.setQueryCachingPolicy(
            new QueryCachingPolicy() {
              @Override
              public void onUse(Query query) {}

              @Override
              public boolean shouldCache(Query query) {
                return true;
              }
            });
        assertEquals(20, collect(searcher));
        assertTrue(runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.HEAP) > 0);
        assertEquals(20, collect(searcher));
        DocIdSetIterator pinned =
            searcher
                .createWeight(MatchAllDocsQuery.INSTANCE, ScoreMode.COMPLETE_NO_SCORES, 1)
                .scorer(reader.leaves().getFirst())
                .iterator();
        assertEquals(0, pinned.nextDoc());
        cache.clear();
        assertTrue(runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.HEAP) > 0);
        assertEquals(1, pinned.nextDoc());
      }
      assertTrue(runtime.ledger().awaitDrained(Duration.ofSeconds(1)));
      assertEquals(
          0, runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
    }
    runtime.shutdown(System.nanoTime() + Duration.ofSeconds(1).toNanos());
  }

  public void testSelectiveNativeCacheOwnerRetirementPinsReader() throws Exception {
    assumeTrue("native storage requires JDK25", Runtime.version().feature() >= 25);
    AccelerationRuntime runtime = runtime(AccelerationConfig.StoragePolicy.SELECTIVE_OFF_HEAP);
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        for (int i = 0; i < 20; i++) writer.addDocument(new Document());
      }
      LRUQueryCache cache = new LRUQueryCache(10, 1L << 20, leaf -> true, 250, runtime);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);
        searcher.setQueryCache(cache);
        searcher.setQueryCachingPolicy(
            new QueryCachingPolicy() {
              @Override
              public void onUse(Query query) {}

              @Override
              public boolean shouldCache(Query query) {
                return true;
              }
            });
        assertEquals(20, collect(searcher));
        assertTrue(runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE) > 0);
        DocIdSetIterator pinned =
            searcher
                .createWeight(MatchAllDocsQuery.INSTANCE, ScoreMode.COMPLETE_NO_SCORES, 1)
                .scorer(reader.leaves().getFirst())
                .iterator();
        assertEquals(0, pinned.nextDoc());
        cache.clear();
        assertTrue(runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.NATIVE) > 0);
        assertEquals(1, pinned.nextDoc());
      }
      assertTrue(runtime.ledger().awaitDrained(Duration.ofSeconds(1)));
    }
    runtime.shutdown(System.nanoTime() + Duration.ofSeconds(1).toNanos());
  }

  public void testEvictedCacheEntriesDoNotAddCoreListeners() throws Exception {
    AccelerationRuntime runtime = runtime();
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        writer.addDocument(new Document());
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        AtomicInteger listeners = new AtomicInteger();
        var original = reader.leaves().getFirst().reader();
        var helper = original.getCoreCacheHelper();
        var counted =
            new IndexReader.CacheHelper() {
              @Override
              public IndexReader.CacheKey getKey() {
                return helper.getKey();
              }

              @Override
              public void addClosedListener(IndexReader.ClosedListener listener) {
                listeners.incrementAndGet();
                helper.addClosedListener(listener);
              }
            };
        var leaf =
            new FilterLeafReader(original) {
              @Override
              public CacheHelper getCoreCacheHelper() {
                return counted;
              }

              @Override
              public CacheHelper getReaderCacheHelper() {
                return in.getReaderCacheHelper();
              }
            };
        IndexSearcher searcher = new IndexSearcher(leaf);
        LRUQueryCache cache = new LRUQueryCache(1, 1L << 20, context -> true, 250, runtime);
        searcher.setQueryCache(cache);
        searcher.setQueryCachingPolicy(alwaysCache());
        for (int i = 0; i < 20; i++) {
          assertEquals(1, collect(searcher));
          cache.clearQuery(MatchAllDocsQuery.INSTANCE);
        }
        assertTrue(
            "core listeners must not grow per population: " + listeners, listeners.get() <= 2);
      }
    }
    runtime.close();
  }

  public void testAbandonedHeapCacheIteratorsReleaseOnOpenReader() throws Exception {
    checkAbandonedCacheIterators(AccelerationConfig.StoragePolicy.CURRENT);
  }

  public void testAbandonedNativeCacheIteratorsReleaseOnOpenReader() throws Exception {
    assumeTrue("native storage requires JDK25", Runtime.version().feature() >= 25);
    checkAbandonedCacheIterators(AccelerationConfig.StoragePolicy.SELECTIVE_OFF_HEAP);
  }

  private void checkAbandonedCacheIterators(AccelerationConfig.StoragePolicy storage)
      throws Exception {
    AccelerationRuntime runtime = runtime(storage);
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        for (int i = 0; i < 20; i++) writer.addDocument(new Document());
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        LRUQueryCache cache = new LRUQueryCache(1, 1L << 20, context -> true, 250, runtime);
        searcher.setQueryCache(cache);
        searcher.setQueryCachingPolicy(alwaysCache());
        assertEquals(20, collect(searcher));
        DocIdSetIterator active =
            searcher
                .createWeight(MatchAllDocsQuery.INSTANCE, ScoreMode.COMPLETE_NO_SCORES, 1)
                .scorer(reader.leaves().getFirst())
                .iterator();
        assertEquals(0, active.nextDoc());
        cache.clearQuery(MatchAllDocsQuery.INSTANCE);
        long pinnedBytes = chargedBytes(runtime);
        assertTrue(pinnedBytes > 0);
        for (int i = 0; i < 20; i++) {
          searcher.search(
              MatchAllDocsQuery.INSTANCE,
              new CollectorManager<Counting, Void>() {
                @Override
                public Counting newCollector() {
                  return new Counting() {
                    @Override
                    public void collect(int doc) {
                      throw new CollectionTerminatedException();
                    }

                    @Override
                    public DocIdSetIterator competitiveIterator() {
                      return DocIdSetIterator.all(20);
                    }
                  };
                }

                @Override
                public Void reduce(Collection<Counting> collectors) {
                  return null;
                }
              });
          cache.clearQuery(MatchAllDocsQuery.INSTANCE);
        }
        for (int attempt = 0; attempt < 20 && chargedBytes(runtime) != pinnedBytes; attempt++) {
          System.gc();
          runtime.ledger().awaitDrained(Duration.ofMillis(50));
        }
        assertEquals(
            "abandoned search iterators must release while the reader stays open",
            pinnedBytes,
            chargedBytes(runtime));
        assertEquals(1, active.nextDoc());
        while (active.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {}
        assertTrue(runtime.ledger().awaitDrained(Duration.ofSeconds(1)));
        Reference.reachabilityFence(active);
        Reference.reachabilityFence(reader);
      }
    }
    runtime.close();
  }

  private static long chargedBytes(AccelerationRuntime runtime) {
    return runtime.ledger().snapshot().charged().values().stream().mapToLong(Long::longValue).sum();
  }

  private static QueryCachingPolicy alwaysCache() {
    return new QueryCachingPolicy() {
      @Override
      public void onUse(Query query) {}

      @Override
      public boolean shouldCache(Query query) {
        return true;
      }
    };
  }

  public void testOrdinalBuildAndRetainedPackedMapChargedOnce() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(4L << 20, 0, 0), 4L << 20);
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        for (int i = 0; i < 10; i++) {
          Document document = new Document();
          document.add(new SortedDocValuesField("id", new BytesRef(Integer.toString(i))));
          writer.addDocument(document);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        var values = reader.leaves().getFirst().reader().getSortedDocValues("id");
        ledger.setOrdinalLimit(131072 + 65536 - 1);
        expectThrows(
            RequestMemoryScope.AdmissionException.class,
            () ->
                OrdinalMap.build(
                    reader.getReaderCacheHelper().getKey(),
                    new TermsEnum[] {values.termsEnum()},
                    new long[] {values.getValueCount()},
                    0.5f,
                    ledger));
        assertEquals(0L, ledger.ordinalBytes());
        ledger.setOrdinalLimit(1L << 20);
        OrdinalMap map =
            OrdinalMap.build(
                reader.getReaderCacheHelper().getKey(),
                new TermsEnum[] {values.termsEnum()},
                new long[] {values.getValueCount()},
                0.5f,
                ledger);
        assertEquals(
            map.ramBytesUsed(),
            ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
        assertEquals(0, ledger.snapshot().reserved().get(MemoryLedger.Kind.HEAP).longValue());
        assertEquals(map.ramBytesUsed(), ledger.ordinalBytes());
        ledger.setOrdinalLimit(0);
        LongValues borrowed = map.getGlobalOrds(0);
        assertEquals(9, borrowed.get(9));
        map.close();
        map.close();
        expectThrows(IllegalStateException.class, () -> borrowed.get(0));
        assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
        assertEquals(0L, ledger.ordinalBytes());
      }
    }
  }

  public void testSharedPhysicalMappingRefundsOnlyAfterLastInputClose() throws Exception {
    MemoryLedger ledger = new MemoryLedger(new MemoryLedger.Limits(0, 0, 1L << 20), 1L << 20);
    try (MMapDirectory directory = new MMapDirectory(createTempDir())) {
      directory.setMemoryLedger(ledger);
      directory.setGroupingFunction(name -> Optional.of("shared"));
      for (String name : new String[] {"first", "second"}) {
        try (var output = directory.createOutput(name, IOContext.DEFAULT)) {
          output.writeLong(73);
        }
      }
      IndexInput first = directory.openInput("first", IOContext.DEFAULT);
      IndexInput second = directory.openInput("second", IOContext.DEFAULT);
      long peak = ledger.snapshot().charged().get(MemoryLedger.Kind.MAPPED);
      assertTrue(peak >= 16);
      first.close();
      assertEquals(peak, ledger.snapshot().charged().get(MemoryLedger.Kind.MAPPED).longValue());
      assertEquals(73, second.readLong());
      second.close();
      assertEquals(0, ledger.snapshot().charged().get(MemoryLedger.Kind.MAPPED).longValue());
    }
  }

  public void testCancelledHeapCacheBuildCannotPublish() throws Exception {
    AccelerationRuntime runtime = runtime();
    MemoryLedger.Owner owner = new MemoryLedger.Owner("cancel-build", "core", 1, "request");
    AccelerationRuntime.TaskContext context = new AccelerationRuntime.TaskContext(owner, 0);
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        for (int i = 0; i < 20; i++) writer.addDocument(new Document());
      }
      LRUQueryCache cache =
          new LRUQueryCache(10, 1L << 20, leaf -> true, 250, runtime) {
            @Override
            protected CacheAndCount cacheImpl(
                org.apache.lucene.search.BulkScorer scorer, int maxDoc) throws IOException {
              CacheAndCount value = super.cacheImpl(scorer, maxDoc);
              context.cancel();
              return value;
            }
          };
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(cache);
        searcher.setQueryCachingPolicy(
            new QueryCachingPolicy() {
              @Override
              public void onUse(Query query) {}

              @Override
              public boolean shouldCache(Query query) {
                return true;
              }
            });
        AccelerationRuntime.TaskBinding binding = runtime.enterTask(context);
        try {
          expectThrows(java.io.InterruptedIOException.class, () -> collect(searcher));
        } finally {
          binding.close();
        }
        assertEquals(0, cache.getCacheSize());
        assertEquals(
            0, runtime.ledger().snapshot().charged().get(MemoryLedger.Kind.HEAP).longValue());
        assertEquals(
            0, runtime.ledger().snapshot().reserved().get(MemoryLedger.Kind.HEAP).longValue());
        assertEquals(20, collect(searcher));
        assertTrue(cache.getCacheSize() > 0);
        cache.clear();
      }
    }
    runtime.shutdown(System.nanoTime() + Duration.ofSeconds(1).toNanos());
  }

  public void testCancelledBoundTaskStopsOnlyManagedOrdinalBuild() throws Exception {
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        Document document = new Document();
        document.add(new SortedDocValuesField("id", new BytesRef("value")));
        writer.addDocument(document);
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        var values = reader.leaves().getFirst().reader().getSortedDocValues("id");
        AccelerationRuntime system = AccelerationRuntime.system();
        AccelerationRuntime.TaskContext context =
            new AccelerationRuntime.TaskContext(
                new MemoryLedger.Owner("cancel-ordinal", "core", 1, "request"), 0);
        context.cancel();
        AccelerationRuntime.TaskBinding binding = system.enterTask(context);
        try {
          expectThrows(
              java.io.InterruptedIOException.class,
              () ->
                  OrdinalMap.build(
                      reader.getReaderCacheHelper().getKey(),
                      new TermsEnum[] {values.termsEnum()},
                      new long[] {values.getValueCount()},
                      0.5f,
                      system.ledger()));
          // The default build stays stock and ignores the bound task.
          var values2 = reader.leaves().getFirst().reader().getSortedDocValues("id");
          assertEquals(
              1,
              OrdinalMap.build(
                      reader.getReaderCacheHelper().getKey(),
                      new TermsEnum[] {values2.termsEnum()},
                      new long[] {values2.getValueCount()},
                      0.5f)
                  .getValueCount());
        } finally {
          binding.close();
        }
      }
    }
  }
}
