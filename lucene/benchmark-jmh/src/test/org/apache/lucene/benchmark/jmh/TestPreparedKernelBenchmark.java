/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
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

import org.apache.lucene.tests.util.LuceneTestCase;

public class TestPreparedKernelBenchmark extends LuceneTestCase {
  public void testSmallPreparedAggregateMatchesDecodedPrefix() throws Exception {
    long[] expected = aggregate("CURRENT_JAVA");
    assertArrayEquals(expected, aggregate("PREPARED_JAVA"));
  }

  public void testDecodedValuesCanRunRepeatedly() throws Exception {
    var benchmark = new PreparedKernelBenchmark();
    benchmark.implementation = "CURRENT_JAVA";
    benchmark.op = "DECODE";
    benchmark.size = 256;
    benchmark.setup();
    try {
      benchmark.beforeInvocation();
      long[] first = ((long[]) benchmark.run()).clone();
      benchmark.beforeInvocation();
      assertArrayEquals(first, (long[]) benchmark.run());
    } finally {
      benchmark.finish();
    }
  }

  private static long[] aggregate(String implementation) throws Exception {
    var benchmark = new PreparedKernelBenchmark();
    benchmark.implementation = implementation;
    benchmark.op = "AGGREGATE";
    benchmark.size = 256;
    benchmark.setup();
    try {
      benchmark.beforeInvocation();
      long[] result = ((long[]) benchmark.run()).clone();
      benchmark.beforeInvocation();
      assertArrayEquals(result, (long[]) benchmark.run());
      return result;
    } finally {
      benchmark.finish();
    }
  }
}
