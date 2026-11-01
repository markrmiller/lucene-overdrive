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
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;
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

/**
 * Prototype reader boundary for Lucene104 quantized-vector bulk scoring on a real mapped index.
 * CURRENT_JAVA is the stock {@code Lucene104ScalarQuantizedVectorScorer} float-query scorer, which
 * reads each row and its corrections through the IndexInput. NATIVE_OPTIMIZED is the same scorer
 * with an explicit runtime, whose bulk calls score rows in place from the mapped field region by
 * ordinal (kernel 30). Each operation scores one random batch, as one HNSW expansion (32) or an
 * exact-search block (64) would.
 */
@SuppressForbidden(reason = "Benchmark temporary index and identity diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class OsqGatherBenchmark {
  @Param({"CURRENT_JAVA", "NATIVE_OPTIMIZED"})
  public String implementation;

  @Param({
    "UNSIGNED_BYTE",
    "PACKED_NIBBLE",
    "SEVEN_BIT",
    "SINGLE_BIT_QUERY_NIBBLE",
    "DIBIT_QUERY_NIBBLE"
  })
  public String encoding;

  @Param({"384"})
  public int dimensions;

  @Param({"32", "64"})
  public int batch;

  private static final int DOCS = 50_000, BATCHES = 1024;

  private Path indexPath;
  private Directory directory;
  private DirectoryReader reader;
  private AccelerationRuntime runtime;
  private RandomVectorScorer scorer;
  private int[][] ordinals;
  private float[] scores;
  private int next;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    indexPath = Files.createTempDirectory("lucene-osq-gather-jmh");
    try {
      directory = new MMapDirectory(indexPath);
      Random random = new Random(0x10530);
      var format = new Lucene104ScalarQuantizedVectorsFormat(ScalarEncoding.valueOf(encoding));
      var config =
          new IndexWriterConfig()
              .setCodec(
                  new Lucene104Codec() {
                    @Override
                    public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                      return format;
                    }
                  });
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int d = 0; d < DOCS; d++) {
          Document doc = new Document();
          doc.add(
              new KnnFloatVectorField("v", vector(random), VectorSimilarityFunction.DOT_PRODUCT));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      reader = DirectoryReader.open(directory);
      var segment = (SegmentReader) reader.leaves().get(0).reader();
      KnnVectorsReader vectors = segment.getVectorReader();
      if (vectors instanceof PerFieldKnnVectorsFormat.FieldsReader fields)
        vectors = fields.getFieldReader("v");
      var values =
          (QuantizedByteVectorValues)
              ((QuantizedVectorsReader) vectors).getQuantizedVectorValues("v");
      float[] query = vector(random);
      var java =
          new Lucene104ScalarQuantizedVectorScorer(DefaultFlatVectorScorer.INSTANCE, null)
              .getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, values, query);
      ordinals = new int[BATCHES][batch];
      for (int[] set : ordinals) for (int i = 0; i < batch; i++) set[i] = random.nextInt(DOCS);
      scores = new float[batch];
      if (implementation.equals("CURRENT_JAVA")) {
        scorer = java;
      } else {
        runtime = BenchmarkAccelerationSupport.open(implementation, Capability.OSQ_GATHER);
        scorer =
            new Lucene104ScalarQuantizedVectorScorer(DefaultFlatVectorScorer.INSTANCE, runtime)
                .getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, values.copy(), query);
        float[] want = new float[batch];
        for (int[] set : ordinals) {
          java.bulkScore(set, want, batch);
          scorer.bulkScore(set, scores, batch);
          for (int i = 0; i < batch; i++)
            if (Float.floatToRawIntBits(want[i]) != Float.floatToRawIntBits(scores[i]))
              throw new IllegalStateException("OSQ gather parity mismatch");
        }
        var coverage = runtime.inventory().counters().get(Capability.OSQ_GATHER);
        if (coverage.nativeCalls() != BATCHES
            || coverage.fallbacks().values().stream().anyMatch(count -> count != 0))
          throw new IllegalStateException(
              "OSQ gather did not use native for every batch: " + coverage);
      }
    } catch (IOException | RuntimeException | Error failure) {
      closeResources();
      throw failure;
    }
  }

  private float[] vector(Random random) {
    float[] v = new float[dimensions];
    float norm = 0;
    for (int i = 0; i < dimensions; i++) {
      v[i] = random.nextFloat() * 2 - 1;
      norm += v[i] * v[i];
    }
    float scale = (float) (1 / Math.sqrt(norm));
    for (int i = 0; i < dimensions; i++) v[i] *= scale;
    return v;
  }

  @Benchmark
  public float score() throws IOException {
    int[] set = ordinals[next++ & (BATCHES - 1)];
    return scorer.bulkScore(set, scores, batch);
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    closeResources();
    if (runtime != null) BenchmarkAccelerationSupport.verifyDrained(runtime);
  }

  private void closeResources() throws IOException {
    try {
      IOUtils.close(reader, directory);
    } finally {
      try {
        if (runtime != null) runtime.close();
      } finally {
        if (indexPath != null) IOUtils.rm(indexPath);
      }
    }
  }
}
