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
import java.lang.ref.Reference;
import org.apache.lucene.search.DocIdSet;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * Lucene adapter whose iterators own leases until exhaustion, explicit close or abandonment. The
 * reader owner must invoke {@link #closeReaders()} at final reader closure to reclaim abandoned
 * iterators that Lucene's iterator API cannot explicitly close.
 *
 * @lucene.experimental
 */
public final class MemoryDocIdSet extends DocIdSet implements AutoCloseable {
  private final DocSetViews.Owned owner;
  private final ManagedMemory.Readers readers;

  /** Takes ownership of an initialized immutable document set. */
  public MemoryDocIdSet(DocSetViews.Owned owner) {
    this(owner, new ManagedMemory.Readers());
  }

  /** Takes ownership of a set whose iterators share the supplied final-reader lifetime. */
  public MemoryDocIdSet(DocSetViews.Owned owner, ManagedMemory.Readers readers) {
    this.owner = owner;
    this.readers = readers;
  }

  @Override
  public DocIdSetIterator iterator() {
    DocSetLease lease = owner.tryAcquire();
    if (lease == null) return null;
    try {
      Iterator iterator = new Iterator(lease.iterator());
      return iterator.cleanup.isClosed() ? null : iterator;
    } catch (RuntimeException | Error failure) {
      lease.close();
      throw failure;
    }
  }

  @Override
  public long ramBytesUsed() {
    return 128;
  }

  /** Actual native or heap backing capacity, separate from Lucene's heap cache overhead. */
  public long capacityBytes() {
    return owner.capacityBytes();
  }

  /** Transfers initialized backing storage to a durable cache owner. */
  public boolean transferOwner(MemoryLedger.Owner next) {
    return owner.transferOwner(next);
  }

  @Override
  public void close() {
    owner.close();
  }

  /**
   * Explicit retry after a failed physical close. Failed owners also remain available through
   * {@link ManagedMemory#retryFailedReclamations(MemoryLedger)} after this adapter disappears.
   */
  public void retryFailedReclamation() {
    owner.retryFailedReclamation();
  }

  /** Final reader shutdown, after all reader users have released their reader references. */
  public void closeReaders() {
    try {
      readers.close();
    } finally {
      close();
    }
  }

  private final class Iterator extends DocIdSetIterator implements AutoCloseable {
    private final DocSetLease.LeasedIterator delegate;
    private final ManagedMemory.ReaderLease cleanup;

    Iterator(DocSetLease.LeasedIterator delegate) {
      this.delegate = delegate;
      cleanup = readers.register(this, delegate::close);
    }

    @Override
    public int docID() {
      return delegate.docID();
    }

    @Override
    public long cost() {
      return delegate.cost();
    }

    @Override
    public int nextDoc() throws IOException {
      try {
        int doc = delegate.nextDoc();
        if (doc == NO_MORE_DOCS) close();
        return doc;
      } finally {
        Reference.reachabilityFence(this);
      }
    }

    @Override
    public int advance(int target) throws IOException {
      try {
        int doc = delegate.advance(target);
        if (doc == NO_MORE_DOCS) close();
        return doc;
      } finally {
        Reference.reachabilityFence(this);
      }
    }

    @Override
    public void close() {
      cleanup.close();
    }
  }
}
