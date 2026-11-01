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
package org.apache.lucene.search;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.UUID;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.LeafReaderPin;
import org.apache.lucene.acceleration.MappedSlices;
import org.apache.lucene.acceleration.OwnedByteBatchScorer;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.RamUsageEstimator;

/** Exact-search ownership only; never returned to a graph seed or cached on a query. */
final class ExactByteVectorScorer implements VectorScorer, Closeable {
  private static final int ROWS = 512;
  private final AccelerationRuntime runtime;
  private final String consumerContext;
  private final LeafReader reader;
  private final ByteVectorValues values;
  private final byte[] target;
  private final String field;
  private final VectorScorer fallback;
  private final DocIdSetIterator vectors;
  private final RequestMemoryScope scope;
  private final RequestMemoryScope.HeapReservation heap;
  private final AccelerationRuntime.TaskBinding binding;
  private final AccelerationRuntime.TaskContext ownedTask;
  private final int[] docs = new int[ROWS];
  private final float[] scores = new float[ROWS];
  private OwnedByteBatchScorer owned;
  private boolean closed;

  static ExactByteVectorScorer tryCreate(
      LeafReaderContext context,
      FieldInfo info,
      ByteVectorValues values,
      byte[] target,
      long cost,
      VectorScorer fallback)
      throws IOException {
    AccelerationRuntime runtime = AccelerationRuntime.configured();
    if (runtime == null) return null;
    LeafReader reader = context.reader();
    if (fallback == null
        || cost < ROWS
        || values.dimension() != 384
        || target.length != 384
        || reader.hasDeletions()
        || values.size() != reader.maxDoc()
        || context.parent != null && context.parent.reader().leaves().size() != 1
        || info.getVectorSimilarityFunction() != VectorSimilarityFunction.DOT_PRODUCT) return null;
    AccelerationRuntime.TaskContext task = runtime.currentTaskContext();
    String consumerContext =
        task == null
            ? OwnedByteBatchScorer.EXACT_STANDALONE_PACKED_CONTEXT
            : OwnedByteBatchScorer.EXACT_OWNED_CONTEXT;
    if (!OwnedByteBatchScorer.supportsExactMappedBatch(
            runtime, values, info.getVectorSimilarityFunction(), consumerContext)
        || !mapped(reader, values, 0)) return null;
    AccelerationRuntime.TaskBinding binding = null;
    AccelerationRuntime.TaskContext ownedTask = null;
    RequestMemoryScope scope = null;
    RequestMemoryScope.HeapReservation heap = null;
    try {
      if (task == null) {
        ownedTask =
            new AccelerationRuntime.TaskContext(
                new MemoryLedger.Owner("lucene", "knn-byte-exact", 0, UUID.randomUUID().toString()),
                0);
        binding = runtime.enterTask(ownedTask);
      }
      scope = runtime.newCallScope();
      long arrays =
          2L
              * RamUsageEstimator.alignObjectSize(
                  RamUsageEstimator.NUM_BYTES_ARRAY_HEADER + (long) ROWS * Integer.BYTES);
      heap =
          scope.reserveHeap(
              arrays + RamUsageEstimator.shallowSizeOfInstance(ExactByteVectorScorer.class) + 512);
      return new ExactByteVectorScorer(
          runtime,
          reader,
          values,
          target,
          fallback,
          scope,
          heap,
          binding,
          ownedTask,
          info.name,
          consumerContext);
    } catch (RequestMemoryScope.AdmissionException denied) {
      closeFailedConstruction(scope, heap, binding, ownedTask);
      if (denied.reason() == RequestMemoryScope.Failure.NODE_CAPACITY
          || denied.reason() == RequestMemoryScope.Failure.REQUEST_LIMIT
          || denied.reason() == RequestMemoryScope.Failure.CORE_LIMIT
          || denied.reason() == RequestMemoryScope.Failure.FAIR_SHARE
          || denied.reason() == RequestMemoryScope.Failure.PRESSURE) return null;
      throw denied;
    } catch (IOException | RuntimeException | Error failure) {
      try {
        closeFailedConstruction(scope, heap, binding, ownedTask);
      } catch (RuntimeException | Error closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  private static void closeFailedConstruction(
      RequestMemoryScope scope,
      RequestMemoryScope.HeapReservation heap,
      AccelerationRuntime.TaskBinding binding,
      AccelerationRuntime.TaskContext task) {
    try {
      if (heap != null) heap.close();
    } finally {
      try {
        if (scope != null) scope.close();
      } finally {
        try {
          if (task != null) task.cancel();
        } finally {
          if (binding != null) binding.close();
        }
      }
    }
  }

  private ExactByteVectorScorer(
      AccelerationRuntime runtime,
      LeafReader reader,
      ByteVectorValues values,
      byte[] target,
      VectorScorer fallback,
      RequestMemoryScope scope,
      RequestMemoryScope.HeapReservation heap,
      AccelerationRuntime.TaskBinding binding,
      AccelerationRuntime.TaskContext ownedTask,
      String field,
      String consumerContext) {
    this.runtime = runtime;
    this.consumerContext = consumerContext;
    this.reader = reader;
    this.values = values;
    this.target = target;
    this.fallback = fallback;
    this.scope = scope;
    this.heap = heap;
    this.field = field;
    this.binding = binding;
    this.ownedTask = ownedTask;
    this.vectors = values.iterator();
  }

  private static boolean mapped(LeafReader reader, ByteVectorValues values, int first)
      throws IOException {
    try (var pin = LeafReaderPin.tryAcquire(reader)) {
      if (pin == null) return false;
      if (!(values instanceof HasIndexSlice sliced) || sliced.getSlice() == null) return false;
      var view = MappedSlices.slice(sliced.getSlice(), (long) first * 384, ROWS * 384L);
      return view != null && view.isDirect();
    }
  }

  private float javaScore(int doc) throws IOException {
    var iterator = fallback.iterator();
    if (iterator.docID() < doc) iterator.advance(doc);
    if (iterator.docID() != doc) throw new IOException("dense byte-vector iterator mismatch");
    return fallback.score();
  }

  @Override
  public float score() throws IOException {
    return javaScore(vectors.docID());
  }

  @Override
  public DocIdSetIterator iterator() {
    return vectors;
  }

  @Override
  public Bulk bulk(DocIdSetIterator matchingDocs) throws IOException {
    DocIdSetIterator matches =
        matchingDocs == null
            ? vectors
            : ConjunctionUtils.createConjunction(List.of(matchingDocs, vectors), List.of());
    if (matches.docID() == -1) matches.nextDoc();
    return (upTo, liveDocs, buffer) -> {
      if (closed) throw new IllegalStateException("exact byte scorer is closed");
      runtime.checkpoint();
      int size = 0;
      for (int doc = matches.docID(); doc < upTo && size < ROWS; doc = matches.nextDoc()) {
        if (liveDocs == null || liveDocs.get(doc)) docs[size++] = doc;
      }
      boolean contiguous = size == ROWS;
      for (int i = 1; contiguous && i < size; i++) contiguous = docs[i] == docs[0] + i;
      float maximum;
      if (size == 0) {
        maximum = Float.NEGATIVE_INFINITY;
      } else if (owned != null || contiguous && mapped(reader, values, docs[0])) {
        if (owned == null)
          owned = OwnedByteBatchScorer.create(runtime, reader, field, target, consumerContext);
        maximum = owned.bulkScore(docs, scores, size);
      } else {
        maximum = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
          if ((i & 63) == 0) runtime.checkpoint();
          scores[i] = javaScore(docs[i]);
          maximum = Math.max(maximum, scores[i]);
        }
      }
      runtime.checkpoint();
      buffer.docs = docs;
      buffer.features = scores;
      buffer.size = size;
      return maximum;
    };
  }

  @Override
  public void close() {
    if (closed) return;
    closed = true;
    try {
      if (owned != null) owned.close();
    } finally {
      closeFailedConstruction(scope, heap, binding, ownedTask);
    }
  }
}
