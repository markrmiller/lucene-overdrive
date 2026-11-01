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
import org.apache.lucene.util.FixedBitSet;
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

/** Four baselines measured through the full allocation, dispatch and publication boundary. */
@SuppressForbidden(reason = "Benchmark provenance and actual native coverage diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(2)
public class AccelerationBoundaryBenchmark {
  /** Native reference/optimized are separate frozen JVM configurations in boundary-benchmark.sh. */
  @Param({"CURRENT_JAVA", "IMPROVED_JAVA", "NATIVE"})
  public String implementation;

  @Param({"256", "4096", "65536"})
  public int words;

  private long[] left;
  private long[] right;
  private FixedBitSet leftBits;
  private FixedBitSet rightBits;
  private AccelerationRuntime runtime;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    Random random = new Random(0x10502);
    left = new long[words];
    right = new long[words];
    for (int i = 0; i < words; i++) {
      left[i] = random.nextLong();
      right[i] = random.nextLong();
    }
    leftBits = new FixedBitSet(left, words * Long.SIZE);
    rightBits = new FixedBitSet(right, words * Long.SIZE);
    if (implementation.equals("NATIVE")) {
      runtime =
          AccelerationRuntime.forQualification(
              org.apache.lucene.acceleration.AccelerationConfig.fromSystemProperties());
      if (!runtime.supports(Capability.DOCSET_INTERSECT))
        throw new IllegalStateException(
            "native benchmark requires qualified intersection: " + runtime.inventory());
      System.out.println(
          "Acceleration benchmark frozen identity: " + runtime.inventory().selected());
    }
    long[] expected = improvedJava();
    if (!Arrays.equals(expected, intersect())) throw new IllegalStateException("benchmark parity");
  }

  /** Equivalent layout change: allocate one output and fuse copying with intersection. */
  private long[] improvedJava() {
    long[] output = new long[words];
    int i = 0;
    for (; i + 3 < words; i += 4) {
      output[i] = left[i] & right[i];
      output[i + 1] = left[i + 1] & right[i + 1];
      output[i + 2] = left[i + 2] & right[i + 2];
      output[i + 3] = left[i + 3] & right[i + 3];
    }
    for (; i < words; i++) output[i] = left[i] & right[i];
    return output;
  }

  @Benchmark
  public long[] intersect() throws IOException {
    return switch (implementation) {
      case "CURRENT_JAVA" -> {
        FixedBitSet copy = leftBits.clone();
        copy.and(rightBits);
        yield copy.getBits();
      }
      case "IMPROVED_JAVA" -> improvedJava();
      case "NATIVE" ->
          BlockOperations.bitset(
              runtime, Capability.DOCSET_INTERSECT, left, right, words * Long.SIZE);
      default -> throw new IllegalArgumentException("unknown implementation");
    };
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    if (runtime != null) {
      var counter = runtime.inventory().counters().get(Capability.DOCSET_INTERSECT);
      if (counter.nativeCalls() == 0 || counter.javaCalls() != 0)
        throw new IllegalStateException("native coverage was not complete: " + counter);
      System.out.println("Acceleration benchmark coverage: " + counter);
      runtime.close();
    }
  }
}
