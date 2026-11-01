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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import org.apache.lucene.acceleration.AccelerationConfig;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.SuppressForbidden;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/** Full query rewrite, natural filtered exact search, ranking and query-owned frame cleanup. */
@SuppressForbidden(reason = "Private benchmark index, startup configuration and diagnostic output")
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 1,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class ExactByteQueryBenchmark {
  @Param({"JAVA64", "JAVA512", "NATIVE512"})
  public String implementation;

  private static final int DIMENSIONS = 384;
  private static final int DOCUMENTS = 4096;
  private static final int SELECTED = 512;
  private Path path;
  private MMapDirectory directory;
  private DirectoryReader reader;
  private IndexSearcher searcher;
  private KnnByteVectorQuery query;
  private AccelerationRuntime runtime;
  private ScoreDoc[] oracle;
  private volatile TopDocs last;
  private final LongAdder calls = new LongAdder();
  private long iterationCalls;
  private long nativeBefore;
  private int references;
  private java.util.Map<MemoryLedger.Kind, Long> readerCharges;
  private java.util.Map<MemoryLedger.Kind, Long> initialCharges;

  @State(Scope.Thread)
  public static class QueryOwner {
    final MemoryLedger.Owner owner =
        new MemoryLedger.Owner("jmh", "exact-byte-query", 0, UUID.randomUUID().toString());
  }

  @Setup(Level.Trial)
  public void setup() throws IOException {
    if (!List.of("JAVA64", "JAVA512", "NATIVE512").contains(implementation))
      throw new IllegalArgumentException("unknown exact-query implementation");
    runtime = AccelerationRuntime.system();
    boolean nativeMode = implementation.equals("NATIVE512");
    if (runtime.isQualificationOnly()
        || runtime.config().heapBytes() <= 0
        || runtime.config().requestBytes() < 1 << 20
        || (nativeMode
            ? runtime.config().mode() != AccelerationConfig.Mode.AUTO
            : runtime.config().mode() != AccelerationConfig.Mode.OFF))
      throw new IllegalStateException(
          "explicit SYSTEM AUTO/native or OFF/Java, with common enrolled budgets, required");
    if (nativeMode != runtime.supports(Capability.VECTOR_I8))
      throw new IllegalStateException("wrong native selection: " + runtime.inventory());
    initialCharges = runtime.inventory().memory().charged();
    path = Files.createTempDirectory("exact-byte-query-jmh-");
    try {
      directory = new MMapDirectory(path);
      Random random = new Random(10502);
      byte[] target = new byte[DIMENSIONS];
      random.nextBytes(target);
      byte[][] selected = new byte[SELECTED][];
      try (IndexWriter writer =
          new IndexWriter(
              directory,
              new IndexWriterConfig()
                  .setCodec(new Lucene104Codec())
                  .setMergePolicy(NoMergePolicy.INSTANCE)
                  .setMaxBufferedDocs(10000))) {
        for (int doc = 0; doc < DOCUMENTS; doc++) {
          byte[] vector = new byte[DIMENSIONS];
          if (doc % 4 == 0) java.util.Arrays.fill(vector, Byte.MIN_VALUE);
          else if (doc % 4 == 1) java.util.Arrays.fill(vector, Byte.MAX_VALUE);
          else random.nextBytes(vector);
          if (doc < SELECTED) selected[doc] = vector;
          Document value = new Document();
          value.add(new KnnByteVectorField("v", vector, VectorSimilarityFunction.DOT_PRODUCT));
          value.add(new StringField("eligible", doc < SELECTED ? "yes" : "no", Field.Store.NO));
          writer.addDocument(value);
        }
      }
      reader = DirectoryReader.open(directory);
      if (reader.leaves().size() != 1
          || reader.hasDeletions()
          || reader.maxDoc() != DOCUMENTS
          || reader.leaves().getFirst().reader().getByteVectorValues("v").size() != DOCUMENTS)
        throw new IllegalStateException(
            "exact native comparison requires one dense undeleted leaf");
      if (implementation.equals("JAVA512")) reader = new Java512Reader(reader);
      searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      query =
          new KnnByteVectorQuery("v", target, SELECTED, new TermQuery(new Term("eligible", "yes")));
      oracle = oracle(target, selected);
      references = reader.leaves().getFirst().reader().getRefCount();
      long firstNative = nativeCalls();
      long started = System.nanoTime();
      last = execute(new QueryOwner());
      long elapsed = System.nanoTime() - started;
      verify(last, oracle);
      if (nativeCalls() - firstNative != (nativeMode ? 1 : 0))
        throw new IllegalStateException("first whole query did not use exactly its intended route");
      if (runtime.inventory().execution().activeScopes() != 0)
        throw new IllegalStateException("first query scope retained");
      readerCharges = runtime.inventory().memory().charged();
      System.out.println(
          "Exact query first request: implementation="
              + implementation
              + "; ns="
              + elapsed
              + "; docs="
              + DOCUMENTS
              + "; selected="
              + SELECTED
              + "; dims="
              + DIMENSIONS
              + "; nativeCalls="
              + (nativeCalls() - firstNative)
              + "; query="
              + query
              + "; vectorProvider="
              + org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil.javaProviderDescription()
              + "; selectedProvider="
              + runtime.inventory().selected());
    } catch (IOException | RuntimeException | Error failure) {
      try {
        close();
      } catch (IOException closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  static ScoreDoc[] oracle(byte[] target, byte[][] vectors) {
    ArrayList<ScoreDoc> expected = new ArrayList<>();
    for (int doc = 0; doc < vectors.length; doc++) {
      int dot = 0;
      for (int dim = 0; dim < target.length; dim++) dot += target[dim] * vectors[doc][dim];
      expected.add(new ScoreDoc(doc, 0.5f + dot / (float) (target.length * (1 << 15))));
    }
    expected.sort(
        Comparator.<ScoreDoc>comparingDouble(hit -> -hit.score).thenComparingInt(hit -> hit.doc));
    return expected.toArray(ScoreDoc[]::new);
  }

  static void verify(TopDocs actual, ScoreDoc[] expected) {
    if (actual.totalHits.relation() != TotalHits.Relation.EQUAL_TO
        || actual.totalHits.value() != expected.length
        || actual.scoreDocs.length != expected.length)
      throw new IllegalStateException("full TopDocs cardinality/totalHits mismatch");
    for (int i = 0; i < expected.length; i++)
      if (actual.scoreDocs[i].doc != expected[i].doc
          || Float.floatToIntBits(actual.scoreDocs[i].score)
              != Float.floatToIntBits(expected[i].score))
        throw new IllegalStateException("full TopDocs signed-byte oracle/tie mismatch at " + i);
  }

  private long nativeCalls() {
    return runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls();
  }

  @Setup(Level.Iteration)
  public void beforeIteration() {
    iterationCalls = calls.sum();
    nativeBefore = nativeCalls();
  }

  private TopDocs execute(QueryOwner worker) throws IOException {
    if (runtime.currentOwner() != null)
      throw new IllegalStateException("borrowed an unrelated query owner");
    var task = new AccelerationRuntime.TaskContext(worker.owner, 0);
    try (var binding = runtime.enterTask(task)) {
      if (binding == null) throw new IllegalStateException("missing query owner binding");
      // Only this control changes Java's 64-row codec bulk size. Its arrays are charged before
      // the query allocates them and are released after the complete TopDocs has been returned.
      if (implementation.equals("JAVA512")) {
        long arrays =
            2L
                * RamUsageEstimator.alignObjectSize(
                    RamUsageEstimator.NUM_BYTES_ARRAY_HEADER + (long) SELECTED * Integer.BYTES);
        try (RequestMemoryScope scope = runtime.newCallScope();
            var charge = scope.reserveHeap(arrays + 512)) {
          if (charge == null) throw new IllegalStateException("Java512 buffer admission missing");
          return searcher.search(query, SELECTED);
        }
      }
      return searcher.search(query, SELECTED);
    } finally {
      task.cancel();
      if (runtime.currentOwner() != null)
        throw new IllegalStateException("query owner was not restored");
    }
  }

  @Benchmark
  public TopDocs query(QueryOwner worker) throws IOException {
    TopDocs answer = execute(worker);
    last = answer;
    calls.increment();
    return answer;
  }

  @TearDown(Level.Iteration)
  public void afterIteration() {
    verify(last, oracle);
    long count = calls.sum() - iterationCalls;
    if (nativeCalls() - nativeBefore != (implementation.equals("NATIVE512") ? count : 0))
      throw new IllegalStateException(
          "iteration native entries differ from whole-query invocations");
    var inventory = runtime.inventory();
    var counter = inventory.counters().get(Capability.VECTOR_I8);
    if (implementation.equals("NATIVE512")
        && (counter.javaCalls() != 0
            || counter.fallbacks().values().stream().mapToLong(Long::longValue).sum() != 0))
      throw new IllegalStateException("native query recorded a Java fallback: " + counter);
    if (counter.stagedBytes() != 0 || counter.errors() != 0 || counter.quarantined())
      throw new IllegalStateException("unexpected staged input, native failure or quarantine");
    for (var entry : inventory.counters().entrySet())
      if (entry.getKey() != Capability.VECTOR_I8 && entry.getValue().nativeCalls() != 0)
        throw new IllegalStateException("another native consumer entered during query comparison");
    if (inventory.execution().activeScopes() != 0
        || !inventory.memory().charged().equals(readerCharges)
        || reader.leaves().getFirst().reader().getRefCount() != references)
      throw new IllegalStateException("query resources retained across iteration: " + inventory);
  }

  @TearDown(Level.Trial)
  public void close() throws IOException {
    try {
      IOUtils.close(reader, directory);
    } finally {
      if (path != null) IOUtils.rm(path);
    }
    if (runtime != null
        && initialCharges != null
        && !runtime.inventory().memory().charged().equals(initialCharges))
      throw new IllegalStateException("reader close did not refund benchmark allocations");
  }

  /** Benchmark-only codec wrapper; the query planner and exact HitQueue remain unchanged. */
  private static final class Java512Reader extends FilterDirectoryReader {
    Java512Reader(DirectoryReader input) throws IOException {
      super(
          input,
          new SubReaderWrapper() {
            @Override
            public LeafReader wrap(LeafReader leaf) {
              return new FilterLeafReader(leaf) {
                @Override
                public ByteVectorValues getByteVectorValues(String field) throws IOException {
                  ByteVectorValues values = in.getByteVectorValues(field);
                  return values == null ? null : new Java512Values(values);
                }

                @Override
                public IndexReader.CacheHelper getCoreCacheHelper() {
                  return in.getCoreCacheHelper();
                }

                @Override
                public IndexReader.CacheHelper getReaderCacheHelper() {
                  return in.getReaderCacheHelper();
                }
              };
            }
          });
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader input) throws IOException {
      return new Java512Reader(input);
    }

    @Override
    public IndexReader.CacheHelper getReaderCacheHelper() {
      return in.getReaderCacheHelper();
    }
  }

  private static final class Java512Values extends ByteVectorValues {
    private final ByteVectorValues values;

    Java512Values(ByteVectorValues values) {
      this.values = values;
    }

    @Override
    public int dimension() {
      return values.dimension();
    }

    @Override
    public int size() {
      return values.size();
    }

    @Override
    public byte[] vectorValue(int ordinal) throws IOException {
      return values.vectorValue(ordinal);
    }

    @Override
    public int ordToDoc(int ordinal) {
      return values.ordToDoc(ordinal);
    }

    @Override
    public DocIndexIterator iterator() {
      return values.iterator();
    }

    @Override
    public ByteVectorValues copy() throws IOException {
      return new Java512Values(values.copy());
    }

    @Override
    public VectorScorer scorer(byte[] target) throws IOException {
      return java512(values.scorer(target));
    }
  }

  static VectorScorer java512(VectorScorer source) {
    return new VectorScorer() {
      @Override
      public float score() throws IOException {
        return source.score();
      }

      @Override
      public DocIdSetIterator iterator() {
        return source.iterator();
      }

      @Override
      public Bulk bulk(DocIdSetIterator matching) throws IOException {
        DocIdSetIterator docs =
            matching == null
                ? iterator()
                : ConjunctionUtils.createConjunction(List.of(matching, iterator()), List.of());
        if (docs.docID() == -1) docs.nextDoc();
        return (upTo, live, buffer) -> {
          if (buffer.docs.length != SELECTED) {
            buffer.docs = new int[SELECTED];
            buffer.features = new float[SELECTED];
          }
          int size = 0;
          float maximum = Float.NEGATIVE_INFINITY;
          for (int doc = docs.docID(); doc < upTo && size < SELECTED; doc = docs.nextDoc())
            if (live == null || live.get(doc)) {
              buffer.docs[size] = doc;
              buffer.features[size] = score();
              maximum = Math.max(maximum, buffer.features[size++]);
            }
          buffer.size = size;
          return maximum;
        };
      }
    };
  }
}
