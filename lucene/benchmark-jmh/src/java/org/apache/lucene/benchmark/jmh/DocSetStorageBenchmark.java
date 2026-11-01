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
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.acceleration.AccelerationConfig;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.memory.DocSetLease;
import org.apache.lucene.acceleration.memory.DocSetView;
import org.apache.lucene.acceleration.memory.DocSetViews;
import org.apache.lucene.acceleration.memory.MemoryLedger;
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

/** Warm reusable representations with Java computation and separately charged storage. */
@SuppressForbidden(reason = "Benchmark provenance and actual native coverage diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(2)
public class DocSetStorageBenchmark {
  @Param({"SORTED", "COMPRESSED", "DENSE", "NATIVE_DENSE"})
  public String representation;

  @Param({"1024", "65536"})
  public int maxDoc;

  @Param({"2", "64"})
  public int stride;

  private AccelerationRuntime runtime;
  private DocSetViews.Owned owned;
  private DocSetLease lease;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    runtime =
        new AccelerationRuntime(
            new AccelerationConfig(
                AccelerationConfig.Mode.OFF,
                null,
                Set.of(),
                Set.of(),
                64L << 20,
                8L << 20,
                64,
                false,
                AccelerationConfig.StoragePolicy.CURRENT,
                new AccelerationConfig.MemoryPolicy(64L << 20, 0, 0, 0, 0),
                new AccelerationConfig.ShadowPolicy(false, 1, 0, 0),
                new AccelerationConfig.Tuning(java.util.Map.of(), java.util.Map.of(), 0, 0)));
    int[] docs = new int[(maxDoc + stride - 1) / stride];
    for (int i = 0; i < docs.length; i++) docs[i] = i * stride;
    var kind = DocSetView.Representation.valueOf(representation);
    owned =
        DocSetViews.copyOf(
            docs,
            maxDoc,
            kind,
            runtime.ledger(),
            new MemoryLedger.Owner("benchmark", "core", 1, "cache"),
            kind == DocSetView.Representation.NATIVE_DENSE ? runtime.nativeAllocator() : null);
    lease = owned.tryAcquire();
    System.out.println(
        "Storage representation=" + representation + " capacity=" + owned.capacityBytes());
    if (lease.view().cardinality() != docs.length || membership() != 128 / stride)
      throw new IllegalStateException("storage benchmark parity");
  }

  @Benchmark
  public int membership() {
    DocSetView view = lease.view();
    int count = 0;
    for (int i = 0; i < 128; i++) if (view.contains(i)) count++;
    return count;
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    owned.close();
    lease.close();
    runtime.close();
    if (runtime.inventory().memory().charged().values().stream().anyMatch(bytes -> bytes != 0))
      throw new IllegalStateException("storage benchmark retained memory");
  }
}
