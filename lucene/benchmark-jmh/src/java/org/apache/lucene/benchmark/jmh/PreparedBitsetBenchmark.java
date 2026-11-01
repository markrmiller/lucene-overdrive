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

/** Immutable prepared bitsets, with reusable output and identical final-word masking. */
@SuppressForbidden(reason = "Benchmark identity, actual ISA and memory-drain diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class PreparedBitsetBenchmark {
  @Param({"CURRENT_JAVA", "NATIVE_REFERENCE", "NATIVE_OPTIMIZED"})
  public String implementation;

  @Param({"256", "4096", "65536"})
  public int words;

  private long[] left;
  private long[] right;
  private long[] javaOutput;
  private long tailMask;
  private AccelerationRuntime runtime;
  private RequestMemoryScope scope;
  private AccelerationProvider.Kernel kernel;
  private AccelerationProvider.Batch prepared;
  private ByteBuffer nativeOutput;
  private long invocations;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var random = new BenchmarkAccelerationSupport.PreparedInputs();
    left = new long[words];
    right = new long[words];
    javaOutput = new long[words];
    for (int i = 0; i < words; i++) left[i] = random.nextLong();
    for (int i = 0; i < words; i++) right[i] = random.nextLong();
    long bitLength = (long) words * Long.SIZE - 7;
    tailMask = (1L << (bitLength & 63)) - 1;
    // Poison all unused bits so omitting the final-word mask fails qualification.
    left[words - 1] |= ~tailMask;
    right[words - 1] |= ~tailMask;
    currentJava();
    if (implementation.equals("CURRENT_JAVA")) return;
    runtime = BenchmarkAccelerationSupport.open(implementation, Capability.DOCSET_INTERSECT);
    try {
      kernel = runtime.diagnosticKernel(Capability.DOCSET_INTERSECT);
      var estimate =
          kernel.estimate(
              new AccelerationProvider.Shape(
                  words,
                  0,
                  (long) words * Long.BYTES * 3,
                  Set.of(AccelerationProvider.Encoding.I64),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  Long.BYTES,
                  true));
      if (!estimate.supported() || estimate.stagingBytes() != 0)
        throw new IllegalStateException("unsupported prepared bitset shape or implicit staging");
      scope = runtime.newQualificationScope();
      ByteBuffer nativeLeft =
          scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer nativeRight =
          scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < words; i++) {
        nativeLeft.putLong(left[i]);
        nativeRight.putLong(right[i]);
      }
      nativeLeft.flip();
      nativeRight.flip();
      nativeOutput = scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 8; i++) parameters.putLong(i * Long.BYTES, 0);
      parameters.putLong(0, words).putLong(16, bitLength);
      prepared =
          new AccelerationProvider.Batch(
              Capability.DOCSET_INTERSECT,
              List.of(
                  new AccelerationProvider.Input(
                      nativeLeft, words, AccelerationProvider.Encoding.I64),
                  new AccelerationProvider.Input(
                      nativeRight, words, AccelerationProvider.Encoding.I64)),
              new AccelerationProvider.Output(nativeOutput, words),
              parameters,
              scope.allocate(estimate.scratchBytes()),
              scope.allocate(256));
      nativeIntersection();
      verifyParity();
      System.out.println("Prepared bitset identity: " + runtime.inventory().selected());
    } catch (IOException | RuntimeException | Error failure) {
      try {
        if (scope != null) scope.close();
      } finally {
        runtime.close();
      }
      throw failure;
    }
  }

  /** Current block-operation Java fallback; HotSpot retains its normal loop optimization. */
  private void currentJava() {
    for (int i = 0; i < words; i++) javaOutput[i] = left[i] & right[i];
    javaOutput[words - 1] &= tailMask;
  }

  private void nativeIntersection() throws IOException {
    var result = kernel.execute(prepared);
    if (result == null
        || result.status() != AccelerationProvider.Status.OK
        || result.detail() != 0
        || result.outputElements() != words)
      throw new IllegalStateException("prepared bitset kernel failed: " + result);
  }

  private void verifyParity() {
    for (int i = 0; i < words; i++)
      if (javaOutput[i] != nativeOutput.getLong(i * Long.BYTES))
        throw new IllegalStateException("prepared intersection parity at word " + i);
  }

  /** Returns complete output to JMH without allocation, input copying or a timed checksum. */
  @Benchmark
  public Object intersect() throws IOException {
    invocations++;
    if (kernel == null) {
      currentJava();
      return javaOutput;
    }
    nativeIntersection();
    return nativeOutput;
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    if (runtime == null) return;
    try {
      currentJava();
      nativeIntersection();
      verifyParity();
      System.out.println("Prepared bitset calls: " + invocations);
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
