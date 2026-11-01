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

import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestOsqHnswSearchBenchmark extends LuceneTestCase {
  public void testNativeReaderChargesCompareByteValues() {
    var ledger = new MemoryLedger(new MemoryLedger.Limits(8192, 20_800, 16_384), 65_536);
    List<MemoryLedger.Charge> charges = new ArrayList<>();
    try {
      charges.add(charge(ledger, MemoryLedger.Kind.HEAP, 4096));
      charges.add(charge(ledger, MemoryLedger.Kind.MAPPED, 8192));
      var readerCharges = ledger.snapshot().charged();
      charges.add(charge(ledger, MemoryLedger.Kind.NATIVE, 20_800));
      assertFalse(
          OsqHnswSearchBenchmark.retainedResources(true, 1, ledger.snapshot(), readerCharges));
      for (var kind : new MemoryLedger.Kind[] {MemoryLedger.Kind.HEAP, MemoryLedger.Kind.MAPPED}) {
        MemoryLedger.Charge leaked = charge(ledger, kind, 256);
        try {
          assertTrue(
              OsqHnswSearchBenchmark.retainedResources(true, 1, ledger.snapshot(), readerCharges));
        } finally {
          leaked.releaseAfterDeallocation();
        }
      }
      assertFalse(
          OsqHnswSearchBenchmark.retainedResources(true, 1, ledger.snapshot(), readerCharges));
    } finally {
      for (MemoryLedger.Charge charge : charges) charge.releaseAfterDeallocation();
    }
  }

  private static MemoryLedger.Charge charge(
      MemoryLedger ledger, MemoryLedger.Kind kind, long bytes) {
    try (var reservation =
        ledger
            .tryReserve(
                new MemoryLedger.Owner("reader", "core", 0, "benchmark"),
                kind,
                bytes,
                MemoryLedger.Purpose.CACHE_VALUE)
            .reservation()) {
      assertNotNull(reservation);
      return reservation.commit(bytes, MemoryLedger.State.PUBLISHED);
    }
  }
}
