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
import java.nio.file.Path;
import java.util.Set;
import org.apache.lucene.acceleration.AccelerationConfig;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.util.SuppressForbidden;

/** Explicit pack selection and drain checks shared by qualification benchmarks. */
@SuppressForbidden(reason = "Benchmark configuration and teardown diagnostics")
final class BenchmarkAccelerationSupport {
  private BenchmarkAccelerationSupport() {}

  /** Same prepared byte stream as native/benchmark.c with seed 10502. Setup only. */
  static final class PreparedInputs {
    private long state = 10502;

    byte nextByte() {
      state ^= state << 13;
      state ^= state >>> 7;
      state ^= state << 17;
      return (byte) state;
    }

    void fill(byte[] bytes) {
      for (int i = 0; i < bytes.length; i++) bytes[i] = nextByte();
    }

    long nextLong() {
      long value = 0;
      for (int i = 0; i < Long.BYTES; i++) value |= (nextByte() & 255L) << (i * 8);
      return value;
    }
  }

  static AccelerationRuntime open(String implementation, Capability capability) throws IOException {
    String property =
        switch (implementation) {
          case "NATIVE_REFERENCE" -> "lucene.acceleration.benchmark.referencePackPath";
          case "NATIVE_OPTIMIZED", "MEASURED_DISPATCH" ->
              "lucene.acceleration.benchmark.optimizedPackPath";
          default ->
              throw new IllegalArgumentException(
                  "unknown native implementation: " + implementation);
        };
    String value = System.getProperty(property);
    if (value == null || value.isBlank())
      throw new IllegalArgumentException("explicit benchmark pack is required: " + property);
    var configured = AccelerationConfig.fromSystemProperties();
    var config =
        new AccelerationConfig(
            AccelerationConfig.Mode.REQUIRED,
            Path.of(value),
            Set.of(capability),
            Set.of(),
            configured.nativeBytes(),
            configured.requestBytes(),
            1,
            true,
            AccelerationConfig.StoragePolicy.CURRENT,
            configured.memoryPolicy(),
            new AccelerationConfig.ShadowPolicy(false, 1, 0, 0),
            configured.tuning());
    var runtime =
        implementation.equals("MEASURED_DISPATCH")
            ? new AccelerationRuntime(config)
            : AccelerationRuntime.forQualification(config);
    String identity = runtime.inventory().selected().get(capability);
    if (identity == null
        || implementation.equals("NATIVE_REFERENCE") && !identity.contains(":baseline-scalar:")
        || implementation.equals("NATIVE_OPTIMIZED") && identity.contains(":baseline-scalar:")) {
      runtime.close();
      throw new IllegalStateException(
          "benchmark implementation does not match actual ISA: " + identity);
    }
    return runtime;
  }

  static void verifyDrained(AccelerationRuntime runtime) {
    var inventory = runtime.inventory();
    var memory = inventory.memory();
    System.out.println(
        "Benchmark post-close memory: " + memory + "; execution=" + inventory.execution());
    if (memory.charged().values().stream().anyMatch(bytes -> bytes != 0)
        || memory.reserved().values().stream().anyMatch(bytes -> bytes != 0)
        || memory.failedReclamations() != 0
        || inventory.execution().activeScopes() != 0)
      throw new IllegalStateException("benchmark owner did not drain: " + inventory);
  }
}
