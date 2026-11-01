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
import java.util.ArrayList;
import java.util.Objects;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.util.IOUtils;

/**
 * Holds reader ownership through a leaf view and every underlying filter delegate.
 *
 * <p>This guard protects ownership only. It does not make doc-values iterators thread safe, nor
 * retain a scorer's resources outside the guarded call. Continue using the original view for IO so
 * its deadline checks and other semantics remain active.
 *
 * @lucene.experimental
 */
public final class LeafReaderPin implements AutoCloseable {
  private final LeafReader[] readers;
  private int firstPinned;

  private LeafReaderPin(LeafReader[] readers) {
    this.readers = readers;
    firstPinned = readers.length;
  }

  /**
   * Pins the backing reader, each intermediate filter, and finally {@code view}. Returns null if
   * any reader is already closed, rolling back all references acquired by this call. A rollback
   * failure is propagated after every acquired reference has been released.
   */
  public static LeafReaderPin tryAcquire(LeafReader view) throws IOException {
    Objects.requireNonNull(view);
    var chain = new ArrayList<LeafReader>();
    LeafReader current = view;
    while (true) {
      chain.add(current);
      if (current instanceof FilterLeafReader filter) current = filter.getDelegate();
      else break;
    }
    // Allocate the complete guard before acquiring references.
    var pin = new LeafReaderPin(chain.toArray(LeafReader[]::new));
    try {
      for (int i = pin.readers.length - 1; i >= 0; i--) {
        if (!pin.readers[i].tryIncRef()) {
          pin.close();
          return null;
        }
        pin.firstPinned = i;
      }
      return pin;
    } catch (Throwable failure) {
      try {
        pin.close();
      } catch (Throwable cleanup) {
        failure = IOUtils.useOrSuppress(failure, cleanup);
      }
      throw IOUtils.rethrowAlways(failure);
    }
  }

  /** Releases every acquired reference exactly once, including when a reader's close throws. */
  @Override
  public synchronized void close() throws IOException {
    Throwable failure = null;
    while (firstPinned < readers.length) {
      LeafReader reader = readers[firstPinned];
      readers[firstPinned++] = null;
      try {
        reader.decRef();
      } catch (Throwable cleanup) {
        failure = IOUtils.useOrSuppress(failure, cleanup);
      }
    }
    if (failure != null) throw IOUtils.rethrowAlways(failure);
  }
}
