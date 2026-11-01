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
package org.apache.lucene.internal.vectorization;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import org.apache.lucene.acceleration.AccelerationConfig;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.BlockOperations;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.tests.util.LuceneTestCase;

/** Strided numeric decode parity and private publication through the native kernel. */
public class TestAcceleratedDecoders extends LuceneTestCase {
  private static AccelerationRuntime nativeRuntime(long requestBytes) throws IOException {
    String pack = System.getProperty("tests.acceleration.pack");
    assumeTrue("optional native reference pack was not supplied", pack != null);
    assumeTrue("native FFM tests require JDK25", Runtime.version().feature() >= 25);
    var runtime =
        AccelerationRuntime.forQualification(
            new AccelerationConfig(
                AccelerationConfig.Mode.AUTO,
                Path.of(pack).toAbsolutePath(),
                Set.of(),
                Set.of(),
                16L << 20,
                requestBytes,
                1,
                true,
                AccelerationConfig.StoragePolicy.CURRENT));
    for (Capability capability : new Capability[] {Capability.DOCVALUES_DECODE}) {
      assertTrue(runtime.inventory().rejections().toString(), runtime.supports(capability));
    }
    return runtime;
  }

  public void testStridedDecodeTransformsWrapExactlyLikeJavaLong() throws Exception {
    var runtime = nativeRuntime(8L << 20);
    for (int width = 8; width <= 64; width += 8) {
      int count = 65, stride = width / 8 + 3, offset = 5;
      byte[] source = new byte[offset + count * stride];
      random().nextBytes(source);
      long multiplier = Long.MAX_VALUE, base = Long.MIN_VALUE + 7;
      long[] expected = new long[count + 7];
      long[] actual = new long[count + 7];
      Arrays.fill(expected, 12345);
      Arrays.fill(actual, 12345);
      for (int i = 0; i < count; i++)
        expected[4 + i] = raw(source, offset + i * stride, width / 8) * multiplier + base;
      BlockOperations.decodeInto(
          runtime, source, offset, count, width, stride, multiplier, base, actual, 4);
      assertArrayEquals(expected, actual);
    }
    assertEquals(8L, runtime.inventory().counters().get(Capability.DOCVALUES_DECODE).nativeCalls());
    long[] unpublished = {99, 88};
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> BlockOperations.decodeInto(runtime, new byte[7], 0, 1, 64, 8, 1, 0, unpublished, 0));
    expectThrows(
        ArithmeticException.class,
        () ->
            BlockOperations.decodeInto(
                runtime, new byte[8], 0, Integer.MAX_VALUE, 64, 8, 1, 0, unpublished, 0));
    assertArrayEquals(new long[] {99, 88}, unpublished);
    assertReclaimed(runtime);
  }

  private static long raw(byte[] bytes, int offset, int width) {
    long value = 0;
    for (int i = 0; i < width; i++) value |= (bytes[offset + i] & 255L) << (i * 8);
    return value;
  }

  private static void assertReclaimed(AccelerationRuntime runtime) {
    assertEquals(
        Long.valueOf(0), runtime.inventory().memory().charged().get(MemoryLedger.Kind.NATIVE));
  }
}
