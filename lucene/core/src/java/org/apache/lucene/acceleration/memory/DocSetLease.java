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
package org.apache.lucene.acceleration.memory;

import java.io.IOException;
import java.util.Objects;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * Operation-wide lease on a cache-owned immutable document set.
 *
 * @lucene.experimental
 */
public final class DocSetLease implements AutoCloseable {
  private final SharedValueLease<DocSetView> lease;

  DocSetLease(SharedValueLease<DocSetView> lease) {
    this.lease = Objects.requireNonNull(lease);
  }

  /** Borrowed view valid until this lease closes. */
  public DocSetView view() {
    return lease.value();
  }

  /** Creates an iterator that owns this lease, closing it automatically at exhaustion. */
  public LeasedIterator iterator() {
    return new LeasedIterator(this);
  }

  @Override
  public void close() {
    lease.close();
  }

  /** Close explicitly when abandoning iteration before exhaustion. */
  public static final class LeasedIterator extends DocIdSetIterator implements AutoCloseable {
    private DocSetLease owner;
    private int doc = -1;
    private final long cost;
    private final DocIdSetIterator delegate;

    private LeasedIterator(DocSetLease owner) {
      this.owner = owner;
      cost = owner.view().cardinality();
      delegate = owner.view().iterator();
    }

    @Override
    public int docID() {
      return doc;
    }

    @Override
    public long cost() {
      return cost;
    }

    @Override
    public int nextDoc() throws IOException {
      return advance(doc == NO_MORE_DOCS ? NO_MORE_DOCS : doc + 1);
    }

    @Override
    public int advance(int target) throws IOException {
      if (owner == null) return doc = NO_MORE_DOCS;
      if (target <= doc) target = doc + 1;
      owner.view(); // Verify the lease before touching borrowed cursor storage.
      doc = delegate.advance(target);
      if (doc == NO_MORE_DOCS) close();
      return doc;
    }

    @Override
    public void close() {
      if (owner != null) {
        owner.close();
        owner = null;
      }
      doc = NO_MORE_DOCS;
    }
  }
}
