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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.AccelerationProvider;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
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

/** Prepared byte scoring: setup owns all input, output and native workspace allocation. */
@SuppressForbidden(reason = "Benchmark identity, actual ISA and memory-drain diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class PreparedAccelerationBenchmark {
  @Param({"CURRENT_JAVA", "NATIVE_REFERENCE", "NATIVE_OPTIMIZED"})
  public String implementation;

  @Param({"64", "384"})
  public int dimensions;

  @Param({"64", "512"})
  public int batch;

  @Param({"DOT_PRODUCT"})
  public VectorSimilarityFunction similarity;

  private byte[] query;
  private byte[][] documents;
  private float[] javaOutput;
  private AccelerationRuntime runtime;
  private RequestMemoryScope scope;
  private AccelerationProvider.Kernel kernel;
  private AccelerationProvider.Batch prepared;
  private ByteBuffer nativeOutput;
  private long invocations;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var random = new BenchmarkAccelerationSupport.PreparedInputs();
    query = new byte[dimensions];
    random.fill(query);
    documents = new byte[batch][dimensions];
    javaOutput = new float[batch];
    for (byte[] document : documents) random.fill(document);
    query[0] = 1;
    for (byte[] document : documents) document[0] = 2;
    currentJava();
    System.out.println("Prepared Java provider: " + FlatVectorScorerUtil.javaProviderDescription());
    if (implementation.equals("CURRENT_JAVA")) return;
    runtime = BenchmarkAccelerationSupport.open(implementation, Capability.VECTOR_I8);
    try {
      kernel = runtime.diagnosticKernel(Capability.VECTOR_I8);
      long bytes = (long) dimensions * (batch + 1) + (long) batch * Float.BYTES;
      var estimate =
          kernel.estimate(
              new AccelerationProvider.Shape(
                  batch,
                  dimensions,
                  bytes,
                  Set.of(AccelerationProvider.Encoding.BYTES),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  1,
                  true));
      if (!estimate.supported() || estimate.stagingBytes() != 0)
        throw new IllegalStateException("unsupported prepared shape or implicit staging");
      scope = runtime.newQualificationScope();
      ByteBuffer nativeQuery = scope.allocate(dimensions);
      nativeQuery.put(query).flip();
      ByteBuffer rows = scope.allocate((long) batch * dimensions);
      for (byte[] document : documents) rows.put(document);
      rows.flip();
      nativeOutput = scope.allocate((long) batch * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 8; i++) parameters.putLong(i * Long.BYTES, 0);
      parameters
          .putLong(0, batch)
          .putLong(8, dimensions)
          .putLong(16, dimensions)
          .putLong(24, similarity.ordinal());
      prepared =
          new AccelerationProvider.Batch(
              Capability.VECTOR_I8,
              List.of(
                  new AccelerationProvider.Input(
                      nativeQuery, dimensions, AccelerationProvider.Encoding.BYTES),
                  new AccelerationProvider.Input(
                      rows, (long) batch * dimensions, AccelerationProvider.Encoding.BYTES)),
              new AccelerationProvider.Output(nativeOutput, batch),
              parameters,
              scope.allocate(estimate.scratchBytes()),
              scope.allocate(256));
      nativeScores();
      verifyParity();
      System.out.println("Prepared scoring identity: " + runtime.inventory().selected());
    } catch (IOException | RuntimeException | Error failure) {
      try {
        if (scope != null) scope.close();
      } finally {
        runtime.close();
      }
      throw failure;
    }
  }

  private void currentJava() {
    for (int i = 0; i < batch; i++) javaOutput[i] = similarity.compare(query, documents[i]);
  }

  private void nativeScores() throws IOException {
    var result = kernel.execute(prepared);
    if (result == null
        || result.status() != AccelerationProvider.Status.OK
        || result.detail() != 0
        || result.outputElements() != batch)
      throw new IllegalStateException("prepared kernel failed: " + result);
  }

  private void verifyParity() {
    for (int i = 0; i < batch; i++)
      if (Float.floatToRawIntBits(javaOutput[i])
          != Float.floatToRawIntBits(nativeOutput.getFloat(i * Float.BYTES)))
        throw new IllegalStateException("prepared score parity at row " + i);
  }

  /**
   * Reuses inputs, outputs, descriptors and scope. JMH consumes the complete output reference,
   * keeping all stores observable without adding a per-element checksum to the timed operation.
   */
  @Benchmark
  public Object score() throws IOException {
    invocations++;
    if (kernel == null) {
      currentJava();
      return javaOutput;
    }
    nativeScores();
    return nativeOutput;
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    if (runtime == null) return;
    try {
      currentJava();
      nativeScores();
      verifyParity();
      System.out.println("Prepared scoring calls: " + invocations);
    } finally {
      try {
        scope.close();
      } finally {
        runtime.close();
      }
    }
    BenchmarkAccelerationSupport.verifyDrained(runtime);
  }
}
