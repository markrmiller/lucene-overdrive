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
package org.apache.lucene.acceleration;

import java.io.IOException;
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestLeafReaderPin extends LuceneTestCase {
  private static final class Fixture implements AutoCloseable {
    final ByteBuffersDirectory directory = new ByteBuffersDirectory();
    final DirectoryReader owner;
    final LeafReader backing;

    Fixture() throws IOException {
      try (var writer = new IndexWriter(directory, new IndexWriterConfig())) {
        var doc = new Document();
        doc.add(new NumericDocValuesField("n", 7));
        writer.addDocument(doc);
      }
      owner = DirectoryReader.open(directory);
      backing = owner.leaves().get(0).reader();
    }

    @Override
    public void close() throws IOException {
      owner.close();
      directory.close();
    }
  }

  // The backing owner is independent, as with Solr's deadline-aware reader views.
  private static class BorrowedView extends FilterLeafReader {
    int closes;
    IOException failure;

    BorrowedView(LeafReader delegate) {
      super(delegate);
    }

    @Override
    protected void doClose() throws IOException {
      closes++;
      if (failure != null) throw failure;
    }

    @Override
    public CacheHelper getCoreCacheHelper() {
      return in.getCoreCacheHelper();
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
      return in.getReaderCacheHelper();
    }
  }

  public void testIdentityPinsExactlyOnceAndCloseIsIdempotent() throws Exception {
    try (var fixture = new Fixture()) {
      var pin = LeafReaderPin.tryAcquire(fixture.backing);
      assertNotNull(pin);
      assertEquals(2, fixture.backing.getRefCount());
      fixture.owner.close();
      assertEquals(1, fixture.backing.getRefCount());
      pin.close();
      assertEquals(0, fixture.backing.getRefCount());
      pin.close();
      assertEquals(0, fixture.backing.getRefCount());
    }
  }

  public void testEveryIntermediateViewRetainsItsOwnResources() throws Exception {
    try (var fixture = new Fixture()) {
      var middle = new BorrowedView(fixture.backing);
      var outer = new BorrowedView(middle);
      var pin = LeafReaderPin.tryAcquire(outer);
      assertNotNull(pin);
      assertEquals(2, fixture.backing.getRefCount());
      assertEquals(2, middle.getRefCount());
      assertEquals(2, outer.getRefCount());
      fixture.owner.close();
      middle.close();
      outer.close();
      assertEquals(0, middle.closes);
      assertEquals(0, outer.closes);
      var values = outer.getNumericDocValues("n");
      assertTrue(values.advanceExact(0));
      assertEquals(7, values.longValue());
      pin.close();
      assertEquals(1, middle.closes);
      assertEquals(1, outer.closes);
      assertEquals(0, fixture.backing.getRefCount());
    }
  }

  public void testClosedBackingBehindOpenViewIsRejected() throws Exception {
    try (var fixture = new Fixture()) {
      var outer = new BorrowedView(fixture.backing);
      fixture.owner.close();
      assertEquals(1, outer.getRefCount());
      assertNull(LeafReaderPin.tryAcquire(outer));
      assertEquals(1, outer.getRefCount());
      outer.close();
    }
  }

  public void testClosedMiddleRollsBackBackingPin() throws Exception {
    try (var fixture = new Fixture()) {
      var middle = new BorrowedView(fixture.backing);
      var outer = new BorrowedView(middle);
      middle.close();
      assertNull(LeafReaderPin.tryAcquire(outer));
      assertEquals(1, fixture.backing.getRefCount());
      assertEquals(1, outer.getRefCount());
      assertEquals(1, middle.closes);
      outer.close();
    }
  }

  public void testCleanupFailureStillReleasesEveryReferenceExactlyOnce() throws Exception {
    try (var fixture = new Fixture()) {
      var middle = new BorrowedView(fixture.backing);
      var outer = new BorrowedView(middle);
      middle.failure = new IOException("middle close");
      outer.failure = new IOException("outer close");
      var pin = LeafReaderPin.tryAcquire(outer);
      assertNotNull(pin);
      fixture.owner.close();
      middle.close();
      outer.close();
      assertSame(outer.failure, expectThrows(IOException.class, pin::close));
      assertArrayEquals(new Throwable[] {middle.failure}, outer.failure.getSuppressed());
      assertEquals(0, fixture.backing.getRefCount());
      assertEquals(0, middle.getRefCount());
      assertEquals(0, outer.getRefCount());
      pin.close();
      assertEquals(1, middle.closes);
      assertEquals(1, outer.closes);
    }
  }

  public void testPreparationRejectsClosedBackingBeforeCodecAccess() throws Exception {
    try (var fixture = new Fixture();
        var runtime =
            new AccelerationRuntime(
                new AccelerationConfig(
                    AccelerationConfig.Mode.OFF,
                    null,
                    Set.of(),
                    Set.of(),
                    0,
                    0,
                    1,
                    false,
                    AccelerationConfig.StoragePolicy.CURRENT))) {
      var outer =
          new BorrowedView(fixture.backing) {
            @Override
            public org.apache.lucene.index.FieldInfos getFieldInfos() {
              throw new AssertionError("closed backing reached field metadata");
            }

            @Override
            public NumericDocValues getNormValues(String field) {
              throw new AssertionError("closed backing reached norms");
            }
          };
      fixture.owner.close();
      expectThrows(
          IOException.class, () -> OwnedByteBatchScorer.create(runtime, outer, "v", new byte[1]));
      expectThrows(
          IOException.class,
          () -> Bm25Block.prepare(runtime, outer, "f", new BM25Similarity(), 1, null, null));
      assertEquals(1, outer.getRefCount());
      outer.close();
      assertEquals(0, runtime.inventory().execution().activeScopes());
    }
  }

  public void testOriginalViewExceptionsArePreserved() throws Exception {
    try (var fixture = new Fixture()) {
      var cancelled = new IOException("view deadline reached");
      var outer =
          new BorrowedView(fixture.backing) {
            @Override
            public NumericDocValues getNumericDocValues(String field) throws IOException {
              throw cancelled;
            }
          };
      try (var pin = LeafReaderPin.tryAcquire(outer)) {
        assertNotNull(pin);
        assertSame(
            cancelled, expectThrows(IOException.class, () -> outer.getNumericDocValues("n")));
      }
      assertEquals(1, outer.getRefCount());
      assertEquals(1, fixture.backing.getRefCount());
      outer.close();
    }
  }
}
