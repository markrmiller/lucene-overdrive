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
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.BlockOperations;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.index.VectorSimilarityFunction;
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

/** Query batches including validation, input staging, scratch, downcalls and publication. */
@SuppressForbidden(reason = "Benchmark provenance and actual native coverage diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(2)
public class ByteVectorBoundaryBenchmark {
  @Param({"CURRENT_JAVA", "IMPROVED_JAVA", "NATIVE"})
  public String implementation;

  @Param({"64", "384"})
  public int dimensions;

  @Param({"64", "512"})
  public int batch;

  private byte[] query;
  private byte[][] documents;
  private byte[] flat;
  private AccelerationRuntime runtime;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    Random random = new Random(0x10502);
    query = new byte[dimensions];
    random.nextBytes(query);
    documents = new byte[batch][dimensions];
    flat = new byte[batch * dimensions];
    for (int i = 0; i < batch; i++) {
      random.nextBytes(documents[i]);
      System.arraycopy(documents[i], 0, flat, i * dimensions, dimensions);
    }
    if (implementation.equals("NATIVE")) {
      runtime =
          AccelerationRuntime.forQualification(
              org.apache.lucene.acceleration.AccelerationConfig.fromSystemProperties());
      if (!runtime.supports(Capability.VECTOR_I8))
        throw new IllegalStateException(
            "byte-vector native benchmark unqualified: " + runtime.inventory());
      System.out.println("Vector benchmark frozen identity: " + runtime.inventory().selected());
    }
    if (!Arrays.equals(currentJava(), improvedJava()) || !Arrays.equals(currentJava(), score()))
      throw new IllegalStateException("benchmark raw score parity");
  }

  private float[] currentJava() {
    float[] output = new float[batch];
    for (int i = 0; i < batch; i++)
      output[i] = VectorSimilarityFunction.DOT_PRODUCT.compare(query, documents[i]);
    return output;
  }

  /** Prepared contiguous rows with fused exact integer reduction and score publication. */
  private float[] improvedJava() {
    float[] output = new float[batch];
    float denominator = (float) (dimensions * (1 << 15));
    for (int row = 0; row < batch; row++) {
      int offset = row * dimensions, sum = 0, i = 0;
      for (; i + 3 < dimensions; i += 4) {
        sum += query[i] * flat[offset + i];
        sum += query[i + 1] * flat[offset + i + 1];
        sum += query[i + 2] * flat[offset + i + 2];
        sum += query[i + 3] * flat[offset + i + 3];
      }
      for (; i < dimensions; i++) sum += query[i] * flat[offset + i];
      output[row] = 0.5f + sum / denominator;
    }
    return output;
  }

  @Benchmark
  public float[] score() throws IOException {
    return switch (implementation) {
      case "CURRENT_JAVA" -> currentJava();
      case "IMPROVED_JAVA" -> improvedJava();
      case "NATIVE" ->
          BlockOperations.byteScores(
              runtime, query, documents, VectorSimilarityFunction.DOT_PRODUCT);
      default -> throw new IllegalArgumentException("unknown implementation");
    };
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    if (runtime != null) {
      var counter = runtime.inventory().counters().get(Capability.VECTOR_I8);
      if (counter.nativeCalls() == 0 || counter.javaCalls() != 0)
        throw new IllegalStateException("incomplete native vector coverage: " + counter);
      System.out.println("Vector benchmark coverage: " + counter);
      runtime.close();
    }
  }
}
