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
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
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

/** Whole-query HNSW comparison for mapped Lucene104 scalar-quantized vectors. */
@SuppressForbidden(reason = "Benchmark temporary index and runtime diagnostics")
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1)
public class OsqHnswSearchBenchmark {
  @Param({"JAVA", "NATIVE1", "NATIVE8", "NATIVE16"})
  public String implementation;

  private static final int DIMENSIONS = 384;
  private static final int DOCUMENTS = 50_000;
  private static final int QUERY_COUNT = 256;
  private static final int TOP_K = 10;

  private Path path;
  private MMapDirectory directory;
  private DirectoryReader reader;
  private IndexSearcher searcher;
  private float[][] queries;
  private long[] expected;
  private AccelerationRuntime runtime;
  private Map<MemoryLedger.Kind, Long> initialCharges;
  private Map<MemoryLedger.Kind, Long> readerCharges;
  private long iterationCalls;
  private long iterationNative;
  private final LongAdder calls = new LongAdder();

  @State(Scope.Thread)
  public static class Cursor {
    private int next;
  }

  @Setup(Level.Trial)
  public void setup() throws IOException {
    boolean nativeMode = implementation.startsWith("NATIVE");
    runtime = AccelerationRuntime.system();
    if (nativeMode != runtime.supports(Capability.OSQ_GATHER))
      throw new IllegalStateException("wrong OSQ selection: " + runtime.inventory());
    initialCharges = runtime.inventory().memory().charged();
    path = Files.createTempDirectory("lucene-osq-hnsw-jmh-");
    try {
      directory = new MMapDirectory(path);
      var format =
          new Lucene104HnswScalarQuantizedVectorsFormat(ScalarEncoding.UNSIGNED_BYTE, 16, 100);
      var config =
          new IndexWriterConfig()
              .setCodec(
                  new Lucene104Codec() {
                    @Override
                    public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                      return format;
                    }
                  });
      Random random = new Random(0x10530);
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int doc = 0; doc < DOCUMENTS; doc++) {
          Document value = new Document();
          value.add(
              new KnnFloatVectorField(
                  "v", unitVector(random), VectorSimilarityFunction.DOT_PRODUCT));
          writer.addDocument(value);
        }
        writer.forceMerge(1);
      }
      reader = DirectoryReader.open(directory);
      if (reader.leaves().size() != 1 || reader.hasDeletions() || reader.maxDoc() != DOCUMENTS)
        throw new IllegalStateException("HNSW benchmark requires one dense undeleted leaf");
      searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      queries = new float[QUERY_COUNT][];
      expected = new long[QUERY_COUNT];
      for (int i = 0; i < queries.length; i++) {
        queries[i] = unitVector(random);
        expected[i] = hash(execute(i));
      }
      long nativeCalls = nativeCalls();
      if (nativeMode && nativeCalls == 0 || !nativeMode && nativeCalls != 0)
        throw new IllegalStateException(
            "setup did not use the selected route: " + runtime.inventory());
      readerCharges = runtime.inventory().memory().charged();
      long oracle = 0;
      for (long value : expected) oracle = Long.rotateLeft(oracle, 7) ^ value;
      System.out.println(
          "HNSW benchmark setup: implementation="
              + implementation
              + "; docs="
              + DOCUMENTS
              + "; dimensions="
              + DIMENSIONS
              + "; topK="
              + TOP_K
              + "; queries="
              + QUERY_COUNT
              + "; oracle="
              + Long.toUnsignedString(oracle, 16)
              + "; nativeCalls="
              + nativeCalls
              + "; vectorProvider="
              + org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil.javaProviderDescription());
    } catch (IOException | RuntimeException | Error failure) {
      closeResources();
      throw failure;
    }
  }

  private static float[] unitVector(Random random) {
    float[] vector = new float[DIMENSIONS];
    float norm = 0;
    for (int i = 0; i < vector.length; i++) {
      vector[i] = random.nextFloat() * 2 - 1;
      norm += vector[i] * vector[i];
    }
    float scale = (float) (1 / Math.sqrt(norm));
    for (int i = 0; i < vector.length; i++) vector[i] *= scale;
    return vector;
  }

  private TopDocs execute(int query) throws IOException {
    return searcher.search(new KnnFloatVectorQuery("v", queries[query], TOP_K), TOP_K);
  }

  private static long hash(TopDocs docs) {
    long hash = docs.totalHits.value() ^ docs.totalHits.relation().ordinal();
    for (ScoreDoc hit : docs.scoreDocs) {
      hash = Long.rotateLeft(hash, 11) ^ hit.doc;
      hash = Long.rotateLeft(hash, 11) ^ Float.floatToRawIntBits(hit.score);
    }
    return hash;
  }

  private long nativeCalls() {
    return runtime.inventory().counters().get(Capability.OSQ_GATHER).nativeCalls();
  }

  @Setup(Level.Iteration)
  public void beforeIteration() {
    iterationCalls = calls.sum();
    iterationNative = nativeCalls();
  }

  @Benchmark
  public TopDocs search(Cursor cursor) throws IOException {
    int query = cursor.next++ & (QUERY_COUNT - 1);
    TopDocs answer = execute(query);
    if ((cursor.next & 1023) == 1 && hash(answer) != expected[query])
      throw new IllegalStateException("HNSW result changed at query " + query);
    calls.increment();
    return answer;
  }

  @TearDown(Level.Iteration)
  public void afterIteration() {
    long completed = calls.sum() - iterationCalls;
    long nativeEntries = nativeCalls() - iterationNative;
    boolean nativeMode = implementation.startsWith("NATIVE");
    if (completed <= 0
        || nativeMode && nativeEntries < completed
        || !nativeMode && nativeEntries != 0)
      throw new IllegalStateException(
          "whole-query route mismatch: completed=" + completed + "; native=" + nativeEntries);
    var inventory = runtime.inventory();
    var counter = inventory.counters().get(Capability.OSQ_GATHER);
    if (counter.errors() != 0 || counter.quarantined())
      throw new IllegalStateException("native error or quarantine: " + counter);
    for (var fallback : counter.fallbacks().entrySet())
      if (fallback.getValue() != 0
          && !fallback.getKey().equals(AccelerationRuntime.Reason.SMALL_BATCH))
        throw new IllegalStateException("unexpected HNSW fallback: " + counter);
    if (retainedResources(
        nativeMode, inventory.execution().activeScopes(), inventory.memory(), readerCharges))
      throw new IllegalStateException("HNSW request retained resources: " + inventory);
  }

  static boolean retainedResources(
      boolean nativeMode,
      long frames,
      MemoryLedger.Snapshot memory,
      Map<MemoryLedger.Kind, Long> readerCharges) {
    long frameBytes = memory.charged().get(MemoryLedger.Kind.NATIVE);
    return nativeMode
            && (frames < 1
                || frames > 4
                || frameBytes != frames * 20_800
                || memory.reserved().values().stream().anyMatch(bytes -> bytes != 0)
                || memory.charged().get(MemoryLedger.Kind.HEAP).longValue()
                    != readerCharges.get(MemoryLedger.Kind.HEAP).longValue()
                || memory.charged().get(MemoryLedger.Kind.MAPPED).longValue()
                    != readerCharges.get(MemoryLedger.Kind.MAPPED).longValue())
        || !nativeMode && !memory.charged().equals(readerCharges);
  }

  @TearDown(Level.Trial)
  public void close() throws IOException {
    if (runtime != null)
      System.out.println(
          "HNSW benchmark final: implementation="
              + implementation
              + "; counter="
              + runtime.inventory().counters().get(Capability.OSQ_GATHER)
              + "; memory="
              + runtime.inventory().memory()
              + "; execution="
              + runtime.inventory().execution());
    closeResources();
    if (runtime != null) {
      runtime.close();
      BenchmarkAccelerationSupport.verifyDrained(runtime);
      if (initialCharges != null && !runtime.inventory().memory().charged().equals(initialCharges))
        throw new IllegalStateException("runtime close did not refund HNSW benchmark allocations");
    }
  }

  private void closeResources() throws IOException {
    try {
      IOUtils.close(reader, directory);
    } finally {
      if (path != null) IOUtils.rm(path);
    }
  }
}
