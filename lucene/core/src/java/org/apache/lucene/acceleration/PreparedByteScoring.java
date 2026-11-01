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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * Diagnostic, owner-thread I8 batch scoring with reusable admitted staging. The query is copied
 * once; candidate bytes, output, parameters and metadata are reused until explicit close. Its
 * public factory is diagnostic; the measured owner-scoped scorer uses the same frame internally. It
 * never performs Java replay after a native failure. Caller-owned vector values and output arrays
 * retain their existing owners. A caller must keep its reader open throughout each call and keep
 * the creating request binding installed throughout use.
 *
 * <p>Each batch creates a few bounded Java descriptor views, but allocates no per-vector storage or
 * native arenas. Native storage is admitted for the maximum batch before preparation succeeds.
 *
 * @lucene.experimental
 */
public final class PreparedByteScoring implements AutoCloseable {
  private final AccelerationRuntime runtime;
  private final String consumerContext;
  private final AccelerationProvider.Kernel kernel;
  private final RequestMemoryScope scope;
  private final RequestMemoryScope.HeapReservation heap;
  private final MemoryLedger.Owner taskOwner;
  private final Thread thread = Thread.currentThread();
  private final int dimensions;
  private final int maximumRows;
  private final AccelerationProvider.Input query;
  private final ByteBuffer rows;
  private final ByteBuffer output;
  private final ByteBuffer parameters;
  private final ByteBuffer scratch;
  private final ByteBuffer metadata;
  private boolean busy;
  private boolean closed;
  private boolean failed;

  /**
   * Prepares only a diagnostic runtime's verified I8 kernel; unavailable kernels fail explicitly.
   */
  public static PreparedByteScoring prepare(
      AccelerationRuntime runtime,
      byte[] query,
      int maximumRows,
      VectorSimilarityFunction similarity)
      throws IOException {
    Objects.requireNonNull(runtime);
    AccelerationProvider.Kernel kernel = runtime.diagnosticKernel(Capability.VECTOR_I8);
    return new PreparedByteScoring(
        runtime, kernel, runtime.newQualificationScope(), query, maximumRows, similarity);
  }

  static PreparedByteScoring prepareMeasured(
      AccelerationRuntime runtime, byte[] query, VectorSimilarityFunction similarity)
      throws IOException {
    return prepareMeasured(runtime, query, similarity, "generic");
  }

  static PreparedByteScoring prepareMeasured(
      AccelerationRuntime runtime,
      byte[] query,
      VectorSimilarityFunction similarity,
      String consumerContext)
      throws IOException {
    int rows = runtime.measuredByteCapacity(query.length, similarity, consumerContext);
    var kernel = runtime.measuredByteKernel(query.length, rows, similarity, consumerContext);
    if (kernel == null) return null;
    long payload = (long) rows * query.length + query.length + (long) rows * Float.BYTES;
    var physical =
        new AccelerationProvider.Shape(
            rows,
            query.length,
            payload,
            Set.of(AccelerationProvider.Encoding.BYTES),
            AccelerationProvider.LayoutKind.NATIVE_BUFFER,
            1,
            true);
    var estimate = kernel.estimate(physical);
    if (!estimate.supported() || estimate.stagingBytes() != 0) return null;
    var mapped =
        new AccelerationProvider.Shape(
            rows,
            query.length,
            Math.addExact(payload + 320, estimate.scratchBytes()),
            physical.encodings(),
            AccelerationProvider.LayoutKind.MAPPED_BUFFER,
            1,
            true);
    if (!runtime.supports(Capability.VECTOR_I8, mapped, similarity.ordinal(), 0, consumerContext))
      return null;
    return new PreparedByteScoring(
        runtime, kernel, runtime.newCallScope(), query, rows, similarity, consumerContext);
  }

  // Package-private injection keeps lifecycle/failure fixtures independent of a native pack.
  PreparedByteScoring(
      AccelerationRuntime runtime,
      AccelerationProvider.Kernel kernel,
      RequestMemoryScope scope,
      byte[] query,
      int maximumRows,
      VectorSimilarityFunction similarity)
      throws IOException {
    this(runtime, kernel, scope, query, maximumRows, similarity, "generic");
  }

  private PreparedByteScoring(
      AccelerationRuntime runtime,
      AccelerationProvider.Kernel kernel,
      RequestMemoryScope scope,
      byte[] query,
      int maximumRows,
      VectorSimilarityFunction similarity,
      String consumerContext)
      throws IOException {
    this.runtime = Objects.requireNonNull(runtime);
    this.consumerContext = Objects.requireNonNull(consumerContext);
    this.kernel = Objects.requireNonNull(kernel);
    this.scope = Objects.requireNonNull(scope);
    RequestMemoryScope.HeapReservation credits = null;
    try {
      if (scope.ledger() != runtime.ledger())
        throw new IllegalArgumentException("diagnostic runtime and matching ledger required");
      Objects.requireNonNull(query);
      Objects.requireNonNull(similarity);
      dimensions = query.length;
      this.maximumRows = maximumRows;
      long rowBytes = Math.multiplyExact((long) maximumRows, dimensions);
      if (dimensions < 1 || dimensions > 32768 || maximumRows < 1 || rowBytes > 16L << 20)
        throw new IllegalArgumentException("unsupported prepared I8 shape");
      var estimate =
          kernel.estimate(
              new AccelerationProvider.Shape(
                  maximumRows,
                  dimensions,
                  rowBytes + dimensions + maximumRows * 4L,
                  Set.of(AccelerationProvider.Encoding.BYTES),
                  AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                  1,
                  true));
      if (!estimate.supported() || estimate.stagingBytes() != 0 || estimate.scratchBytes() < 0)
        throw new IOException("unsupported prepared I8 workspace");
      long outputBytes = Math.multiplyExact((long) maximumRows, Float.BYTES);
      long logicalBytes =
          Math.addExact(
              Math.addExact(Math.addExact(rowBytes, dimensions), Math.addExact(320L, outputBytes)),
              estimate.scratchBytes());
      if (logicalBytes > 128L << 20) throw new IOException("unsupported prepared I8 workspace");
      // One owner and one ledger admission cover the five disjoint, eight-aligned ABI spans.
      // Diagnostic row staging stays independent; SYSTEM consumes the reader's mapped rows.
      long outputOffset = align8(dimensions);
      long parametersOffset = align8(Math.addExact(outputOffset, outputBytes));
      long scratchOffset = Math.addExact(parametersOffset, 64);
      long metadataOffset = align8(Math.addExact(scratchOffset, estimate.scratchBytes()));
      long workspaceBytes = Math.addExact(metadataOffset, 256);
      long stagedBytes = runtime.isQualificationOnly() ? align8(rowBytes) : 0;
      if (Math.addExact(workspaceBytes, stagedBytes) > 128L << 20)
        throw new IOException("unsupported prepared I8 workspace");
      runtime.checkpoint();
      taskOwner = runtime.currentOwner();
      credits = scope.reserveHeap(2048); // Frame metadata and bounded per-batch descriptor overlap.
      ByteBuffer workspace = scope.allocate(workspaceBytes);
      ByteBuffer preparedQuery = workspace.slice(0, dimensions).order(ByteOrder.LITTLE_ENDIAN);
      preparedQuery.put(query).flip();
      this.query =
          new AccelerationProvider.Input(
              preparedQuery, dimensions, AccelerationProvider.Encoding.BYTES);
      rows =
          runtime.isQualificationOnly()
              ? scope.allocate(rowBytes).order(ByteOrder.LITTLE_ENDIAN)
              : null;
      output =
          workspace
              .slice(Math.toIntExact(outputOffset), Math.toIntExact(outputBytes))
              .order(ByteOrder.LITTLE_ENDIAN);
      parameters =
          workspace.slice(Math.toIntExact(parametersOffset), 64).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 8; i++) parameters.putLong(i * 8, 0);
      parameters.putLong(8, dimensions);
      parameters.putLong(16, dimensions);
      parameters.putLong(24, similarity.ordinal());
      scratch =
          workspace
              .slice(Math.toIntExact(scratchOffset), Math.toIntExact(estimate.scratchBytes()))
              .order(ByteOrder.LITTLE_ENDIAN);
      metadata =
          workspace.slice(Math.toIntExact(metadataOffset), 256).order(ByteOrder.LITTLE_ENDIAN);
      checkpoint();
      heap = credits;
    } catch (IOException | RuntimeException | Error failure) {
      if (credits != null) {
        try {
          credits.close();
        } catch (RuntimeException | Error closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      try {
        scope.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  /** Scores array rows in input order, publishing only after complete success. */
  public float score(byte[][] documents, int offset, int count, float[] scores, int scoreOffset)
      throws IOException {
    if (!runtime.isQualificationOnly()) throw new IOException("staged scoring is unmeasured");
    begin(count, scores, scoreOffset);
    try {
      Objects.checkFromIndexSize(offset, count, documents.length);
      for (int i = 0; i < count; i++)
        if (Objects.requireNonNull(documents[offset + i]).length != dimensions)
          throw new IllegalArgumentException("vector dimension mismatch");
      rows.clear().limit(Math.multiplyExact(count, dimensions));
      for (int i = 0; i < count; i++) {
        if ((i & 63) == 0) checkpoint();
        rows.put(documents[offset + i]);
      }
      return execute(count, scores, scoreOffset);
    } finally {
      busy = false;
    }
  }

  /**
   * Stages existing reader vectors directly, supporting unordered and duplicate ordinals. Shared
   * arrays returned by {@link ByteVectorValues#vectorValue(int)} are copied before the next lookup.
   */
  public float score(
      ByteVectorValues values, int[] ordinals, int count, float[] scores, int scoreOffset)
      throws IOException {
    if (!runtime.isQualificationOnly()) throw new IOException("staged scoring is unmeasured");
    begin(count, scores, scoreOffset);
    try {
      Objects.checkFromIndexSize(0, count, ordinals.length);
      if (values.dimension() != dimensions)
        throw new IllegalArgumentException("vector dimension mismatch");
      for (int i = 0; i < count; i++) Objects.checkIndex(ordinals[i], values.size());
      rows.clear().limit(Math.multiplyExact(count, dimensions));
      for (int i = 0; i < count; i++) {
        if ((i & 63) == 0) checkpoint();
        byte[] vector = values.vectorValue(ordinals[i]);
        if (vector.length != dimensions) throw new IOException("reader vector dimension mismatch");
        rows.put(vector);
      }
      return execute(count, scores, scoreOffset);
    } finally {
      busy = false;
    }
  }

  /**
   * Scores a contiguous ordinal range directly from a mapped, unquantized byte-vector slice. The
   * owning leaf is pinned through execution and released before publication; unsupported mappings
   * fail explicitly without replay. Values must belong to the supplied leaf. No input copy or
   * mapping lifetime is transferred to the frame.
   */
  public float scoreContiguous(
      LeafReader reader,
      ByteVectorValues values,
      int firstOrdinal,
      int count,
      float[] scores,
      int scoreOffset)
      throws IOException {
    begin(count, scores, scoreOffset);
    try {
      Objects.requireNonNull(reader);
      float maximum;
      try (var pin = LeafReaderPin.tryAcquire(reader)) {
        if (pin == null) throw new IOException("vector reader closed");
        Objects.checkFromIndexSize(firstOrdinal, count, values.size());
        if (values.dimension() != dimensions || values.getVectorByteLength() != dimensions)
          throw new IllegalArgumentException("unquantized byte-vector layout required");
        if (count == 0) return Float.NEGATIVE_INFINITY;
        if (!(values instanceof HasIndexSlice sliced) || sliced.getSlice() == null)
          throw new UnavailableMappingException();
        ByteBuffer data =
            MappedSlices.slice(
                sliced.getSlice(),
                Math.multiplyExact((long) firstOrdinal, dimensions),
                Math.multiplyExact((long) count, dimensions));
        if (data == null || !data.isDirect()) throw new UnavailableMappingException();
        maximum = compute(data, count);
      }
      return publish(maximum, count, scores, scoreOffset);
    } finally {
      busy = false;
    }
  }

  static final class UnavailableMappingException extends IOException {}

  private void begin(int count, float[] scores, int offset) throws IOException {
    checkThread();
    if (closed || failed || busy) throw new IllegalStateException("prepared scoring unavailable");
    Objects.checkFromIndexSize(offset, count, scores.length);
    if (count > maximumRows) throw new IllegalArgumentException("prepared batch capacity exceeded");
    checkpoint();
    busy = true;
  }

  private void checkpoint() throws IOException {
    runtime.checkpoint();
    if (!Objects.equals(taskOwner, runtime.currentOwner()))
      throw new IOException("prepared scoring request owner changed");
    if (scope.isCancelled()) throw new IOException("prepared scoring cancelled");
  }

  private float execute(int count, float[] scores, int offset) throws IOException {
    checkpoint();
    if (count == 0) return Float.NEGATIVE_INFINITY;
    rows.flip();
    return publish(compute(rows, count), count, scores, offset);
  }

  private float compute(ByteBuffer data, int count) throws IOException {
    checkpoint();
    output.clear().limit(count * 4);
    parameters.putLong(0, count);
    var batch =
        new AccelerationProvider.Batch(
            Capability.VECTOR_I8,
            List.of(
                query,
                new AccelerationProvider.Input(
                    data, (long) count * dimensions, AccelerationProvider.Encoding.BYTES)),
            new AccelerationProvider.Output(output, count),
            parameters,
            scratch,
            metadata);
    try {
      if (!runtime.isQualificationOnly()) {
        if (!runtime.executeMappedByte(batch, scope, consumerContext))
          throw new IOException("measured I8 execution unavailable");
      } else {
        var result = kernel.execute(batch);
        if (result == null
            || result.status() != AccelerationProvider.Status.OK
            || result.detail() != 0
            || result.outputElements() != count)
          throw new IOException("prepared I8 kernel did not return complete success: " + result);
      }
    } catch (IOException | RuntimeException | Error failure) {
      failed = true; // Never reuse a frame after an unknown-state native failure or attempt replay.
      throw failure;
    }
    float maximum = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < count; i++) maximum = Math.max(maximum, output.getFloat(i * 4));
    return maximum;
  }

  private float publish(float maximum, int count, float[] scores, int offset) throws IOException {
    checkpoint();
    for (int i = 0; i < count; i++) scores[offset + i] = output.getFloat(i * 4);
    return maximum;
  }

  private static long align8(long bytes) {
    if (bytes < 0) throw new IllegalArgumentException("negative workspace span");
    return Math.addExact(bytes, 7) & ~7L;
  }

  private void checkThread() {
    if (thread != Thread.currentThread())
      throw new IllegalStateException("prepared scoring belongs to another thread");
  }

  /** Owner-thread recovery after a diagnosed close failure; charges remain held until success. */
  public void retryFailedReclamations() {
    checkThread();
    if (!closed) throw new IllegalStateException("close prepared scoring before recovery");
    scope.retryFailedReclamations();
  }

  @Override
  public void close() {
    checkThread();
    if (busy) throw new IllegalStateException("prepared scoring call still active");
    if (closed) return;
    closed = true;
    try {
      heap.close();
    } finally {
      scope.close();
    }
  }
}
