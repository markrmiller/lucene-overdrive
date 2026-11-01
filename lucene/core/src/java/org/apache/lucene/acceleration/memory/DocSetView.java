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

import org.apache.lucene.search.DocIdSetIterator;

/**
 * Immutable, lifetime-bounded read operations shared by every document-set representation.
 *
 * @lucene.experimental
 */
public interface DocSetView extends AutoCloseable {
  /** Representation selected from measured cardinality and admitted capacities. */
  enum Representation {
    SORTED,
    COMPRESSED,
    DENSE,
    NATIVE_DENSE
  }

  /** Exclusive document ID bound. */
  int maxDoc();

  /** Number of set documents. */
  int cardinality();

  /** Random membership lookup. */
  boolean contains(int doc);

  /** First set document at or after target, or {@link DocIdSetIterator#NO_MORE_DOCS}. */
  int nextDoc(int target);

  /** Borrowed forward iterator; the owning allocation or lease must stay alive throughout use. */
  default DocIdSetIterator iterator() {
    DocSetView view = this;
    return new DocIdSetIterator() {
      private int doc = -1;

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public long cost() {
        return view.cardinality();
      }

      @Override
      public int nextDoc() {
        return advance(doc == NO_MORE_DOCS ? NO_MORE_DOCS : doc + 1);
      }

      @Override
      public int advance(int target) {
        if (doc == NO_MORE_DOCS) return doc;
        return doc = view.nextDoc(Math.max(target, doc + 1));
      }
    };
  }

  /** Owned charged backing capacity. */
  long capacityBytes();

  /** Concrete representation. */
  Representation representation();

  /** Owner retirement; callers holding a lease must close that lease instead. */
  @Override
  void close();
}
