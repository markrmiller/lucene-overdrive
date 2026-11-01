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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.BooleanSupplier;

/**
 * Singleflight immutable builds with generation eligibility checked again at publication.
 * Cancellation removes only that waiter; the builder observes cancellation when no waiters remain.
 *
 * @lucene.experimental
 */
public final class GenerationCache<K, V> implements AutoCloseable {
  /** Builder must poll cancellation and unwind every partial allocation on failure. */
  @FunctionalInterface
  public interface Builder<V> {
    /** Returns a completely initialized owner, with capacity already enrolled. */
    SharedValueLease.Entry<V> build(BooleanSupplier cancelled) throws IOException;
  }

  private static final class Flight<V> {
    final CompletableFuture<SharedValueLease.Entry<V>> ready = new CompletableFuture<>();
    int waiters;
    volatile boolean invalidated;
  }

  private final Object lock = new Object();
  private final Map<K, Flight<V>> flights = new HashMap<>();
  private long generation;
  private boolean closed;
  private long builds;
  private long sharedWaits;
  private long cancelledWaiters;
  private long rejectedPublications;

  /** Initial eligible generation. */
  public GenerationCache(long generation) {
    this.generation = generation;
  }

  /** Aggregate singleflight diagnostics. */
  public record Statistics(
      long builds,
      long sharedWaits,
      long cancelledWaiters,
      long rejectedPublications,
      int entries) {}

  /** Snapshot without calling builders or destructors. */
  public Statistics statistics() {
    synchronized (lock) {
      return new Statistics(
          builds, sharedWaits, cancelledWaiters, rejectedPublications, flights.size());
    }
  }

  /** Acquires the shared value, or builds exactly once for an eligible key and generation. */
  public SharedValueLease<V> acquire(
      K key, long expectedGeneration, Builder<V> builder, BooleanSupplier cancelled)
      throws IOException, InterruptedException {
    Objects.requireNonNull(key);
    Objects.requireNonNull(builder);
    Objects.requireNonNull(cancelled);
    if (cancelled.getAsBoolean()) throw new IOException("cancelled cache waiter");
    Flight<V> flight;
    boolean lead;
    synchronized (lock) {
      if (closed || generation != expectedGeneration)
        throw new IOException("ineligible generation");
      flight = flights.get(key);
      lead = flight == null;
      if (lead) {
        flight = new Flight<>();
        flights.put(key, flight);
        builds++;
      } else sharedWaits++;
      flight.waiters = Math.incrementExact(flight.waiters);
    }
    final Flight<V> work = flight;
    try {
      if (lead) {
        SharedValueLease.Entry<V> value = null;
        try {
          value =
              Objects.requireNonNull(
                  builder.build(
                      () -> {
                        boolean leaderCancelled = cancelled.getAsBoolean();
                        synchronized (lock) {
                          return work.invalidated
                              || work.waiters == 0
                              || work.waiters == 1 && leaderCancelled;
                        }
                      }));
          boolean publish;
          synchronized (lock) {
            publish =
                closed == false
                    && generation == expectedGeneration
                    && work.invalidated == false
                    && flights.get(key) == work;
            if (publish) work.ready.complete(value);
            else rejectedPublications++;
          }
          if (publish == false) {
            value.retire();
            value = null;
            work.ready.completeExceptionally(
                new IOException("generation invalidated during build"));
          }
        } catch (IOException | RuntimeException | Error failure) {
          if (value != null && work.ready.isDone() == false) value.retire();
          synchronized (lock) {
            flights.remove(key, work);
          }
          work.ready.completeExceptionally(failure);
          throw failure;
        }
      }
      while (true) {
        if (work.invalidated) throw new IOException("generation invalidated while waiting");
        if (cancelled.getAsBoolean()) {
          synchronized (lock) {
            cancelledWaiters++;
          }
          throw new IOException("cancelled cache waiter");
        }
        try {
          SharedValueLease<V> lease = work.ready.get(10, TimeUnit.MILLISECONDS).tryAcquire();
          if (lease == null) throw new IOException("cache value retired before acquisition");
          return lease;
        } catch (
            @SuppressWarnings("unused")
            TimeoutException pending) {
          // A finite wakeup keeps cancellation independent of another waiter's builder.
        } catch (ExecutionException failure) {
          Throwable cause = failure.getCause();
          if (cause instanceof IOException io) throw io;
          if (cause instanceof RuntimeException runtime) throw runtime;
          if (cause instanceof Error error) throw error;
          throw new IOException(cause);
        }
      }
    } finally {
      synchronized (lock) {
        work.waiters--;
        if (work.waiters == 0 && work.ready.isDone() == false) {
          work.invalidated = true;
          flights.remove(key, work);
        }
      }
    }
  }

  /** Invalidates old builders and owners; active leases keep their backing memory alive. */
  public void advanceGeneration(long nextGeneration) {
    ArrayList<SharedValueLease.Entry<V>> retire = new ArrayList<>();
    synchronized (lock) {
      if (closed) throw new IllegalStateException("closed generation cache");
      if (nextGeneration <= generation)
        throw new IllegalArgumentException("generation must advance");
      generation = nextGeneration;
      invalidate(retire);
    }
    for (SharedValueLease.Entry<V> value : retire) value.retire();
  }

  private void invalidate(ArrayList<SharedValueLease.Entry<V>> retire) {
    for (Flight<V> flight : flights.values()) {
      flight.invalidated = true;
      if (flight.ready.isDone() && flight.ready.isCompletedExceptionally() == false)
        retire.add(flight.ready.getNow(null));
    }
    flights.clear();
  }

  @Override
  public void close() {
    ArrayList<SharedValueLease.Entry<V>> retire = new ArrayList<>();
    synchronized (lock) {
      if (closed) return;
      closed = true;
      invalidate(retire);
    }
    for (SharedValueLease.Entry<V> value : retire) value.retire();
  }
}
