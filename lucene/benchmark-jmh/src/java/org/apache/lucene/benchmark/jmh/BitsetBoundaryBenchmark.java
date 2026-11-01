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
import java.nio.LongBuffer;
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

/**
 * Repeated query-filter intersections including candidate staging and complete heap publication.
 */
@SuppressForbidden(reason = "Qualification identity and memory drain diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class BitsetBoundaryBenchmark {
  @Param({"CURRENT_JAVA", "NATIVE_OPTIMIZED"})
  public String implementation;

  @Param({"65536"})
  public int words;

  private long[] queryFilter;
  private long[] candidate;
  private long[] output;
  private long tailMask;
  private AccelerationRuntime runtime;
  private RequestMemoryScope scope;
  private RequestMemoryScope.HeapReservation heap;
  private AccelerationProvider.Kernel kernel;
  private AccelerationProvider.Batch prepared;
  private LongBuffer nativeCandidate;
  private LongBuffer nativeOutput;
  private long calls;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    if (words < 1 || words > 1 << 20) throw new IllegalArgumentException("bounded word capacity");
    var random = new BenchmarkAccelerationSupport.PreparedInputs();
    queryFilter = new long[words];
    candidate = new long[words];
    output = new long[words];
    for (int i = 0; i < words; i++) queryFilter[i] = random.nextLong();
    for (int i = 0; i < words; i++) candidate[i] = random.nextLong();
    long bitLength = (long) words * Long.SIZE - 7;
    tailMask = (1L << (bitLength & 63)) - 1;
    queryFilter[words - 1] |= ~tailMask;
    candidate[words - 1] |= ~tailMask;
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
        throw new IOException("unsupported intersection workspace");
      scope = runtime.newQualificationScope();
      heap = scope.reserveHeap(2048);
      ByteBuffer query = scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      query.asLongBuffer().put(queryFilter);
      ByteBuffer rows = scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      nativeCandidate = rows.asLongBuffer();
      ByteBuffer result = scope.allocate((long) words * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      nativeOutput = result.asLongBuffer();
      ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 8; i++) parameters.putLong(i * Long.BYTES, 0);
      parameters.putLong(0, words).putLong(16, bitLength);
      prepared =
          new AccelerationProvider.Batch(
              Capability.DOCSET_INTERSECT,
              List.of(
                  new AccelerationProvider.Input(query, words, AccelerationProvider.Encoding.I64),
                  new AccelerationProvider.Input(rows, words, AccelerationProvider.Encoding.I64)),
              new AccelerationProvider.Output(result, words),
              parameters,
              scope.allocate(estimate.scratchBytes()),
              scope.allocate(256));
      intersect();
      verify();
      System.out.println("Full intersection identity: " + runtime.inventory().selected());
    } catch (IOException | RuntimeException | Error failure) {
      closeResources();
      throw failure;
    }
  }

  /** The fixed query filter is copied once; each call reads, stages and publishes one candidate. */
  @Benchmark
  public long[] intersect() throws IOException {
    calls++;
    if (kernel == null) {
      for (int i = 0; i < words; i++) output[i] = queryFilter[i] & candidate[i];
      output[words - 1] &= tailMask;
    } else {
      runtime.checkpoint();
      if (scope.isCancelled()) throw new IOException("intersection cancelled");
      nativeCandidate.clear();
      nativeCandidate.put(candidate);
      var result = kernel.execute(prepared);
      if (result == null
          || result.status() != AccelerationProvider.Status.OK
          || result.detail() != 0
          || result.outputElements() != words)
        throw new IOException("intersection failed: " + result);
      runtime.checkpoint();
      nativeOutput.clear();
      nativeOutput.get(output);
    }
    return output;
  }

  private void verify() {
    for (int i = 0; i < words; i++) {
      long expected = queryFilter[i] & candidate[i];
      if (i == words - 1) expected &= tailMask;
      if (output[i] != expected)
        throw new IllegalStateException("intersection parity at word " + i);
    }
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    try {
      intersect();
      verify();
      System.out.println("Full intersection calls: " + calls);
    } finally {
      closeResources();
    }
    if (runtime != null) BenchmarkAccelerationSupport.verifyDrained(runtime);
  }

  private void closeResources() throws IOException {
    try {
      if (heap != null) heap.close();
    } finally {
      try {
        if (scope != null) scope.close();
      } finally {
        if (runtime != null) runtime.close();
      }
    }
  }
}
