/* Licensed to the Apache Software Foundation (ASF) under one or more
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
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.OwnedByteBatchScorer;
import org.apache.lucene.acceleration.PreparedByteScoring;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
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
import org.openjdk.jmh.infra.Blackhole;

/** Reader lookup, row staging, kernel execution and publication with one reusable owner frame. */
@SuppressForbidden(reason = "Benchmark temporary index, identity and drain diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class ReaderByteScoringBenchmark {
  @Param({"CURRENT_JAVA", "NATIVE_REFERENCE", "NATIVE_OPTIMIZED"})
  public String implementation;

  @Param({"64", "384"})
  public int dimensions;

  @Param({"64", "512"})
  public int batch;

  @Param({"CONTIGUOUS", "SCATTERED"})
  public String accessPattern;

  @Param({"STAGED", "MAPPED_CONTIGUOUS"})
  public String delivery;

  private Path indexPath;
  private Directory directory;
  private DirectoryReader reader;
  private ByteVectorValues values;
  private RandomVectorScorer javaScorer;
  private AccelerationRuntime runtime;
  private PreparedByteScoring prepared;
  private OwnedByteBatchScorer owned;
  private int[] ordinals;
  private float[] scores;
  private float[] expected;
  private long invocations;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    if (delivery.equals("MAPPED_CONTIGUOUS") && !accessPattern.equals("CONTIGUOUS"))
      throw new IllegalArgumentException("direct mapping requires contiguous ordinal range");
    Random random = new Random(0x10502);
    indexPath = Files.createTempDirectory("lucene-prepared-vector-jmh");
    try {
      directory = FSDirectory.open(indexPath);
      try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
        for (int i = 0; i < 4096; i++) {
          byte[] vector = new byte[dimensions];
          random.nextBytes(vector);
          Document document = new Document();
          document.add(new KnnByteVectorField("v", vector, VectorSimilarityFunction.DOT_PRODUCT));
          writer.addDocument(document);
        }
        writer.forceMerge(1);
      }
      reader = DirectoryReader.open(directory);
      if (reader.leaves().size() != 1)
        throw new IllegalStateException("expected one vector segment");
      values = reader.leaves().get(0).reader().getByteVectorValues("v").copy();
      byte[] query = new byte[dimensions];
      random.nextBytes(query);
      ordinals = new int[batch];
      for (int i = 0; i < batch; i++) {
        ordinals[i] =
            switch (accessPattern) {
              case "CONTIGUOUS" -> i;
              case "SCATTERED" -> random.nextInt(values.size());
              default -> throw new IllegalArgumentException("unknown reader access pattern");
            };
      }
      scores = new float[batch];
      expected = new float[batch];
      javaScorer =
          FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
              .getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, values.copy(), query)
              .forGraphTraversal();
      System.out.println(
          "Reader Java provider: "
              + FlatVectorScorerUtil.javaProviderDescription()
              + "; scorer="
              + javaScorer.getClass().getName()
              + "; values="
              + values.getClass().getName()
              + "; access="
              + accessPattern);
      javaScorer.bulkScore(ordinals, expected, batch);
      if (!implementation.equals("CURRENT_JAVA")) {
        runtime = BenchmarkAccelerationSupport.open(implementation, Capability.VECTOR_I8);
        if (implementation.equals("MEASURED_DISPATCH")) {
          owned = OwnedByteBatchScorer.create(runtime, reader.leaves().get(0).reader(), "v", query);
        } else
          prepared =
              PreparedByteScoring.prepare(
                  runtime, query, batch, VectorSimilarityFunction.DOT_PRODUCT);
        System.out.println(
            "Reader scoring identity: "
                + runtime.inventory().selected()
                + "; values="
                + values.getClass().getName()
                + "; access="
                + accessPattern
                + "; delivery="
                + delivery);
      }
      verifyParity();
    } catch (IOException | RuntimeException | Error failure) {
      closeResources();
      throw failure;
    }
  }

  private void verifyParity() throws IOException {
    float expectedMax = javaScorer.bulkScore(ordinals, expected, batch);
    float actualMax = compute();
    if (Float.floatToRawIntBits(expectedMax) != Float.floatToRawIntBits(actualMax))
      throw new IllegalStateException("reader scoring maximum parity");
    for (int i = 0; i < batch; i++)
      if (Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(scores[i]))
        throw new IllegalStateException("reader scoring parity at ordinal position " + i);
  }

  private float compute() throws IOException {
    if (owned != null) return owned.bulkScore(ordinals, scores, batch);
    return prepared == null
        ? javaScorer.bulkScore(ordinals, scores, batch)
        : delivery.equals("MAPPED_CONTIGUOUS")
            ? prepared.scoreContiguous(
                reader.leaves().get(0).reader(), values, ordinals[0], batch, scores, 0)
            : prepared.score(values, ordinals, batch, scores, 0);
  }

  /** Query storage, frame and result arrays persist for the owner thread's trial. */
  @Benchmark
  public float score(Blackhole blackhole) throws IOException {
    invocations++;
    float maximum = compute();
    blackhole.consume(scores);
    return maximum;
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    try {
      verifyParity();
      System.out.println("Reader scoring calls: " + invocations);
      if (owned != null
          && runtime.inventory().counters().get(Capability.VECTOR_I8).nativeCalls() == 0)
        throw new IllegalStateException("measured dispatch did not use native");
    } finally {
      closeResources();
    }
    if (runtime != null) BenchmarkAccelerationSupport.verifyDrained(runtime);
  }

  private void closeResources() throws IOException {
    try {
      if (owned != null) owned.close();
      if (prepared != null) prepared.close();
    } finally {
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
}
