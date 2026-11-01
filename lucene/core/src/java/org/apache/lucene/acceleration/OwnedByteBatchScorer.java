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
package org.apache.lucene.acceleration;

import java.io.IOException;
import java.util.Objects;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Explicit query/leaf lifetime for measured byte batches. Close on the creating thread before
 * releasing the reader or request binding. Scalar and graph scoring use the active Java scorer.
 * Native scratch is bounded and reused until close; there is no worker-thread cache. Only the
 * measured contiguous mapped DOT_PRODUCT shapes (64 or 384 dimensions, 64 or 512 rows) can dispatch
 * natively.
 *
 * @lucene.experimental
 */
public final class OwnedByteBatchScorer implements RandomVectorScorer, AutoCloseable {
  /** Exact-query contract when the caller already supplies its task owner. */
  public static final String EXACT_OWNED_CONTEXT = "lucene.knn-byte.exact.owned.v1";

  /** Exact-query contract when the scorer creates its own task owner. */
  public static final String EXACT_STANDALONE_CONTEXT = "lucene.knn-byte.exact.standalone.v1";

  /** Exact-query contract with packed mapped preparation and a scorer-owned task. */
  public static final String EXACT_STANDALONE_PACKED_CONTEXT =
      "lucene.knn-byte.exact.standalone.packed.v2";

  private final AccelerationRuntime runtime;
  private final String consumerContext;
  private final LeafReader reader;
  private final ByteVectorValues values;
  private final RandomVectorScorer javaScorer;
  private final PreparedByteScoring prepared;
  private final MemoryLedger.Owner taskOwner;
  private final Thread thread = Thread.currentThread();
  private boolean closed;

  /** Creates a scorer for this reader's field, snapshotting its query exactly once. */
  public static OwnedByteBatchScorer create(
      AccelerationRuntime runtime, LeafReader reader, String field, byte[] query)
      throws IOException {
    return create(runtime, reader, field, query, "generic");
  }

  /** Creates a scorer whose optional mapped frame must match the supplied consumer contract. */
  public static OwnedByteBatchScorer create(
      AccelerationRuntime runtime,
      LeafReader reader,
      String field,
      byte[] query,
      String consumerContext)
      throws IOException {
    Objects.requireNonNull(runtime);
    Objects.requireNonNull(reader);
    OwnedByteBatchScorer result = null;
    try (var pin = LeafReaderPin.tryAcquire(reader)) {
      if (pin == null) throw new IOException("vector reader closed");
      var info = reader.getFieldInfos().fieldInfo(field);
      if (info == null || info.getVectorDimension() == 0)
        throw new IllegalArgumentException("byte-vector field required");
      var values = reader.getByteVectorValues(field);
      if (values == null || values.dimension() != query.length)
        throw new IllegalArgumentException("byte-vector query dimension mismatch");
      result =
          new OwnedByteBatchScorer(
              runtime,
              reader,
              values.copy(),
              query.clone(),
              info.getVectorSimilarityFunction(),
              consumerContext);
    } catch (Throwable failure) {
      if (result != null) {
        try {
          result.close();
        } catch (Throwable cleanup) {
          failure = IOUtils.useOrSuppress(failure, cleanup);
        }
      }
      throw IOUtils.rethrowAlways(failure);
    }
    return result;
  }

  /**
   * Exact mapped preflight for a proposed 512-row plain-byte batch, without allocating its frame.
   * Reader density, live documents and a real mapped slice remain the caller's responsibility.
   */
  public static boolean supportsExactMappedBatch(
      AccelerationRuntime runtime, ByteVectorValues values, VectorSimilarityFunction similarity) {
    return supportsExactMappedBatch(runtime, values, similarity, "generic");
  }

  /** Pure exact preflight including the consumer contract, before query buffers or frames exist. */
  public static boolean supportsExactMappedBatch(
      AccelerationRuntime runtime,
      ByteVectorValues values,
      VectorSimilarityFunction similarity,
      String consumerContext) {
    if (runtime.isQualificationOnly()
        || values.dimension() != 384
        || values.getVectorByteLength() != 384
        || similarity != VectorSimilarityFunction.DOT_PRODUCT) return false;
    var kernel = runtime.measuredByteKernel(384, 512, similarity, consumerContext);
    if (kernel == null) return false;
    long payload = 512L * 384 + 384 + 512L * Float.BYTES;
    var physical =
        new AccelerationProvider.Shape(
            512,
            384,
            payload,
            java.util.Set.of(AccelerationProvider.Encoding.BYTES),
            AccelerationProvider.LayoutKind.NATIVE_BUFFER,
            1,
            true);
    var estimate = kernel.estimate(physical);
    if (!estimate.supported() || estimate.stagingBytes() != 0) return false;
    var mapped =
        new AccelerationProvider.Shape(
            512,
            384,
            Math.addExact(payload + 320, estimate.scratchBytes()),
            physical.encodings(),
            AccelerationProvider.LayoutKind.MAPPED_BUFFER,
            1,
            true);
    return runtime.supports(Capability.VECTOR_I8, mapped, similarity.ordinal(), 0, consumerContext);
  }

  private OwnedByteBatchScorer(
      AccelerationRuntime runtime,
      LeafReader reader,
      ByteVectorValues values,
      byte[] query,
      VectorSimilarityFunction similarity,
      String consumerContext)
      throws IOException {
    this.runtime = runtime;
    this.consumerContext = consumerContext;
    taskOwner = runtime.currentOwner();
    this.reader = reader;
    this.values = values;
    javaScorer =
        FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
            .getRandomVectorScorer(similarity, values.copy(), query)
            .forGraphTraversal();
    PreparedByteScoring frame;
    try {
      frame = PreparedByteScoring.prepareMeasured(runtime, query, similarity, consumerContext);
    } catch (
        @SuppressWarnings("unused")
        RequestMemoryScope.AdmissionException denied) {
      runtime.recordJava(Capability.VECTOR_I8, AccelerationRuntime.Reason.BUDGET_DENIED);
      frame = null;
    }
    prepared = frame;
  }

  private void check() throws IOException {
    if (Thread.currentThread() != thread || closed)
      throw new IllegalStateException("byte scorer is closed or belongs to another thread");
    runtime.checkpoint();
    if (!Objects.equals(taskOwner, runtime.currentOwner()))
      throw new IOException("byte scorer request owner changed");
  }

  @Override
  public float score(int node) throws IOException {
    check();
    return javaScorer.score(node);
  }

  @Override
  public float bulkScore(int[] nodes, float[] scores, int count) throws IOException {
    check();
    Objects.checkFromIndexSize(0, count, nodes.length);
    Objects.checkFromIndexSize(0, count, scores.length);
    for (int i = 0; i < count; i++) Objects.checkIndex(nodes[i], values.size());
    boolean contiguous =
        prepared != null
            && MeasuredByteProfile.shape(
                values.dimension(), count, VectorSimilarityFunction.DOT_PRODUCT)
            && runtime.measuredByteKernel(
                    values.dimension(),
                    count,
                    VectorSimilarityFunction.DOT_PRODUCT,
                    consumerContext)
                != null;
    if (contiguous) {
      for (int i = 1; i < count; i++) {
        if (nodes[i] != nodes[0] + i) {
          contiguous = false;
          break;
        }
      }
    }
    if (contiguous) {
      try {
        return prepared.scoreContiguous(reader, values, nodes[0], count, scores, 0);
      } catch (
          @SuppressWarnings("unused")
          PreparedByteScoring.UnavailableMappingException unavailable) {
        // The typed failure precedes kernel execution. Unknown native failures are never replayed.
      }
    }
    runtime.recordJava(Capability.VECTOR_I8, AccelerationRuntime.Reason.NO_MEASURED_WIN);
    return javaScorer.bulkScore(nodes, scores, count);
  }

  @Override
  public RandomVectorScorer forGraphTraversal() {
    return javaScorer;
  }

  @Override
  public int maxOrd() {
    return javaScorer.maxOrd();
  }

  @Override
  public int ordToDoc(int ord) {
    return javaScorer.ordToDoc(ord);
  }

  @Override
  public Bits getAcceptOrds(Bits acceptDocs) {
    return javaScorer.getAcceptOrds(acceptDocs);
  }

  @Override
  public void close() {
    if (Thread.currentThread() != thread) throw new IllegalStateException("foreign scorer thread");
    if (closed) return;
    closed = true;
    if (prepared != null) prepared.close();
  }
}
