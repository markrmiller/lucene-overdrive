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
package org.apache.lucene.util.quantization;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Set;
import org.apache.lucene.acceleration.AccelerationConfig;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Packing must reach its own capability without another packing kernel enrolled. */
public class TestScalarPackingCapability extends LuceneTestCase {
  public void testBinaryDispatchWithOnlyBinaryCapability() throws Exception {
    assertMatchingPacking(Capability.QUANTIZE_BINARY, 1);
  }

  public void testDibitDispatchWithOnlyDibitCapability() throws Exception {
    assertMatchingPacking(Capability.QUANTIZE_DIBIT, 2);
  }

  private void assertMatchingPacking(Capability capability, int bits) throws Exception {
    String configured = System.getProperty("tests.acceleration.pack");
    assumeTrue("native packing regression requires an explicit pack", configured != null);
    assumeTrue("native FFM regression requires JDK25", Runtime.version().feature() >= 25);
    var disabled = EnumSet.allOf(Capability.class);
    disabled.remove(capability);
    var config =
        new AccelerationConfig(
            AccelerationConfig.Mode.REQUIRED,
            Path.of(configured),
            Set.of(capability),
            disabled,
            16L << 20,
            8L << 20,
            1,
            true,
            AccelerationConfig.StoragePolicy.CURRENT);
    try (var runtime = AccelerationRuntime.forQualification(config)) {
      assertEquals(Set.of(capability), runtime.inventory().selected().keySet());
      assertFalse(runtime.supports(Capability.QUANTIZE_TRANSPOSE));
      for (int length : new int[] {256, 257, 4096}) {
        byte[] vector = new byte[length];
        for (int i = 0; i < length; i++) vector[i] = (byte) random().nextInt(1 << bits);
        int blocks = (length + 7) >>> 3;
        byte[] expected = new byte[blocks * bits];
        for (int plane = 0; plane < bits; plane++)
          for (int i = 0; i < vector.length; i++)
            expected[plane * blocks + (i >>> 3)] |=
                (byte) (((vector[i] >>> plane) & 1) << (7 - (i & 7)));
        byte[] actual = new byte[expected.length];
        Arrays.fill(actual, (byte) 0xff);
        byte[] original = vector.clone();
        long before = runtime.inventory().counters().get(capability).nativeCalls();
        if (bits == 1) OptimizedScalarQuantizer.packAsBinary(runtime, vector, actual);
        else OptimizedScalarQuantizer.transposeDibit(runtime, vector, actual);
        assertArrayEquals(expected, actual);
        assertArrayEquals(original, vector);
        var counter = runtime.inventory().counters().get(capability);
        assertEquals("matching capability must execute C", before + 1, counter.nativeCalls());
        assertEquals(0, counter.javaCalls());
        assertEquals(0, counter.errors());
        assertFalse(counter.quarantined());
        assertTrue(counter.fallbacks().values().stream().allMatch(n -> n == 0));
        for (var entry : runtime.inventory().counters().entrySet())
          if (entry.getKey() != capability) assertEquals(0, entry.getValue().nativeCalls());
      }
    }
  }
}
