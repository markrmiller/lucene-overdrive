/* Licensed to the Apache Software Foundation (ASF) under one or more
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
package org.apache.lucene.codecs.hnsw;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.acceleration.AccelerationProvider;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.CallFrame;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.MappedSlices;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.Constants;
import org.apache.lucene.util.hnsw.HasKnnVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.apache.lucene.util.quantization.BaseQuantizedByteVectorValues;
import org.apache.lucene.util.quantization.LegacyQuantizedByteVectorValues;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;

/**
 * Retains the exact deployed Java scorer, including SIMD, while dispatching admitted signed-byte,
 * float and scalar-quantized query batches. Bounded native frames share one publication boundary.
 * Supplier and graph-construction scorers remain Java-only.
 *
 * @lucene.experimental
 */
public final class AcceleratedFlatVectorsScorer implements FlatVectorsScorer {
  private final FlatVectorsScorer delegate;
  private final AccelerationRuntime runtime;

  public AcceleratedFlatVectorsScorer(FlatVectorsScorer delegate, AccelerationRuntime runtime) {
    this.delegate = Objects.requireNonNull(delegate);
    this.runtime = Objects.requireNonNull(runtime);
  }

  @Override
  public int exactScalarFloatLimit() {
    return delegate.exactScalarFloatLimit();
  }

  @Override
  public String toString() {
    return delegate.toString();
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(
      VectorSimilarityFunction similarity, KnnVectorValues values, float[] target)
      throws IOException {
    RandomVectorScorer java = delegate.getRandomVectorScorer(similarity, values, target);
    if (values instanceof LegacyQuantizedByteVectorValues legacy
        && legacy.getScalarQuantizer().getBits() > 0
        && legacy.getScalarQuantizer().getBits() <= 7
        && target.length > 0
        && target.length <= 16384
        && runtime.supports(Capability.QUANTIZED_RESCORE)) {
      byte[] query = new byte[target.length];
      float correction =
          ScalarQuantizedVectorScorer.quantizeQuery(
              target, query, similarity, legacy.getScalarQuantizer());
      return new QueryScorer(
          java,
          values,
          legacy.copy(),
          query,
          similarity,
          correction,
          legacy.getScalarQuantizer().getConstantMultiplier());
    }
    if (values instanceof FloatVectorValues floats
        && target.length > 0
        && target.length <= delegate.exactScalarFloatLimit()
        && runtime.supports(Capability.VECTOR_F32)) {
      return new FloatQueryScorer(java, values, floats.copy(), target, similarity);
    }
    runtime.recordJava(Capability.VECTOR_F32, AccelerationRuntime.Reason.UNSUPPORTED);
    return java;
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(
      VectorSimilarityFunction similarity, KnnVectorValues values, byte[] target)
      throws IOException {
    RandomVectorScorer java = delegate.getRandomVectorScorer(similarity, values, target);
    if (values instanceof ByteVectorValues bytes
        && !(values instanceof BaseQuantizedByteVectorValues)
        && target.length > 0
        && target.length <= 32768
        && runtime.supports(Capability.VECTOR_I8)) {
      return new QueryScorer(java, values, bytes.copy(), target, similarity, 0, 0);
    }
    return java;
  }

  @Override
  public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
      VectorSimilarityFunction similarity, KnnVectorValues values) throws IOException {
    return supplier(delegate.getRandomVectorScorerSupplier(similarity, values));
  }

  /**
   * Prototype: scores an already quantized Lucene104 query against rows read in place from the
   * mapped field region (kernel 30), gathering rows by ordinal. Unmapped values, other shapes or a
   * runtime without the kernel keep the Java scorer.
   */
  public RandomVectorScorer wrapQuantized(
      RandomVectorScorer java,
      QuantizedByteVectorValues values,
      byte[] query,
      OptimizedScalarQuantizer.QuantizationResult corrections,
      VectorSimilarityFunction similarity)
      throws IOException {
    if (!runtime.supports(Capability.OSQ_GATHER)
        || values.dimension() < 1
        || values.dimension() > 32768
        || !(values instanceof HasIndexSlice sliced)
        || sliced.getSlice() == null) return java;
    var encoding = values.getScalarEncoding();
    int dimension = values.dimension();
    if (query.length
        != (encoding.isAsymmetric()
            ? encoding.getQueryPackedLength(dimension)
            : encoding.getDiscreteDimensions(dimension))) return java;
    long stride = encoding.getDocPackedLength(dimension) + 16L;
    ByteBuffer region =
        MappedSlices.slice(sliced.getSlice(), 0, Math.multiplyExact(stride, values.size()));
    if (region == null) return java;
    return new GatherQueryScorer(
        java, values, region, query.clone(), corrections, similarity, stride);
  }

  /** One ordinary native call per bulk batch over mapped rows and a staged query. */
  private final class GatherQueryScorer implements RandomVectorScorer, HasKnnVectorValues {
    private final RandomVectorScorer java;
    private final KnnVectorValues original;
    private final ByteBuffer region, query;
    private final long[] parameters = new long[8];

    GatherQueryScorer(
        RandomVectorScorer java,
        QuantizedByteVectorValues values,
        ByteBuffer region,
        byte[] query,
        OptimizedScalarQuantizer.QuantizationResult corrections,
        VectorSimilarityFunction similarity,
        long stride)
        throws IOException {
      this.java = java;
      this.original = values;
      this.region = region;
      this.query = ByteBuffer.wrap(query);
      parameters[1] = values.dimension();
      parameters[2] = stride;
      parameters[3] = similarity.ordinal();
      parameters[4] = values.getScalarEncoding().getWireNumber();
      parameters[5] =
          Integer.toUnsignedLong(Float.floatToRawIntBits(corrections.lowerInterval()))
              | (long) Float.floatToRawIntBits(corrections.upperInterval()) << 32;
      parameters[6] =
          Integer.toUnsignedLong(Float.floatToRawIntBits(corrections.additionalCorrection()))
              | (long) corrections.quantizedComponentSum() << 32;
      parameters[7] = Integer.toUnsignedLong(Float.floatToRawIntBits(values.getCentroidDP()));
    }

    @Override
    public float bulkScore(int[] nodes, float[] scores, int count) throws IOException {
      runtime.checkpoint();
      Objects.checkFromIndexSize(0, count, nodes.length);
      Objects.checkFromIndexSize(0, count, scores.length);
      if (count == 0) return Float.NEGATIVE_INFINITY;
      if (!runtime.eligible(Capability.OSQ_GATHER, count))
        return java.bulkScore(nodes, scores, count);
      for (int i = 0; i < count; i++) Objects.checkIndex(nodes[i], maxOrd());
      try (CallFrame frame =
          count * 8L + query.remaining() <= CallFrame.DATA_BYTES ? runtime.acquireFrame() : null) {
        if (frame != null) return frameScore(frame, nodes, scores, count);
      }
      try (RequestMemoryScope scope = runtime.newCallScope()) {
        ByteBuffer ordinals = scope.allocate(count * 4L).order(ByteOrder.LITTLE_ENDIAN);
        ordinals.asIntBuffer().put(nodes, 0, count);
        ByteBuffer output = scope.allocate(count * 4L).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer p = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer nativeQuery = scope.allocate(query.remaining());
        nativeQuery.put(0, query, 0, query.remaining());
        p.putLong(0, count);
        for (int i = 1; i < 8; i++) p.putLong(i * 8, parameters[i]);
        var batch =
            new AccelerationProvider.Batch(
                Capability.OSQ_GATHER,
                List.of(
                    new AccelerationProvider.Input(
                        nativeQuery, nativeQuery.remaining(), AccelerationProvider.Encoding.BYTES),
                    new AccelerationProvider.Input(
                        region, region.remaining(), AccelerationProvider.Encoding.BYTES),
                    new AccelerationProvider.Input(
                        ordinals, count, AccelerationProvider.Encoding.I32)),
                new AccelerationProvider.Output(output, count),
                p,
                scope.allocate(0),
                scope.allocate(256));
        if (!runtime.execute(batch, count, count * 8L + query.remaining(), scope))
          return java.bulkScore(nodes, scores, count);
        runtime.checkpoint();
        output.asFloatBuffer().get(0, scores, 0, count);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) max = Math.max(max, scores[i]);
        return max;
      } catch (RequestMemoryScope.AdmissionException denied) {
        if (!denied.isCapacityFailure()) {
          runtime.checkpoint();
          throw new IOException("native scoring scope stopped", denied);
        }
        runtime.checkpoint();
        runtime.recordJava(Capability.OSQ_GATHER, AccelerationRuntime.Reason.BUDGET_DENIED);
        return java.bulkScore(nodes, scores, count);
      }
    }

    /** The last frame batch; reused while the frame and count repeat, so the bridge's view is. */
    private CallFrame batchFrame;

    private AccelerationProvider.Batch frameBatch;
    private ByteBuffer frameOrdinals, frameOutput, frameQuery;

    /** Fills this thread's reusable frame: no per-call ledger reservation or allocation. */
    private float frameScore(CallFrame frame, int[] nodes, float[] scores, int count)
        throws IOException {
      if (frame != batchFrame || frameBatch == null || frameBatch.output().elements() != count) {
        frameOrdinals = frame.data(0, count * 4);
        frameOutput = frame.data(count * 4, count * 4);
        frameQuery = frame.data(count * 8, query.remaining());
        frameBatch =
            new AccelerationProvider.Batch(
                Capability.OSQ_GATHER,
                List.of(
                    new AccelerationProvider.Input(
                        frameQuery, frameQuery.remaining(), AccelerationProvider.Encoding.BYTES),
                    new AccelerationProvider.Input(
                        region, region.remaining(), AccelerationProvider.Encoding.BYTES),
                    new AccelerationProvider.Input(
                        frameOrdinals, count, AccelerationProvider.Encoding.I32)),
                new AccelerationProvider.Output(frameOutput, count),
                frame.parameters(),
                frame.scratch(0),
                frame.metadata(256));
        batchFrame = frame;
      }
      frameOrdinals.asIntBuffer().put(0, nodes, 0, count);
      // A different scorer may have used this thread's frame since our previous call.
      frameQuery.put(0, query, 0, query.remaining());
      // Another scorer on this thread may have written the shared parameter block.
      ByteBuffer p = frameBatch.parameters();
      p.putLong(0, count);
      for (int i = 1; i < 8; i++) p.putLong(i * 8, parameters[i]);
      if (!runtime.execute(frameBatch, count, count * 8L + query.remaining(), frame.scope()))
        return java.bulkScore(nodes, scores, count);
      runtime.checkpoint();
      frameOutput.asFloatBuffer().get(0, scores, 0, count);
      float max = Float.NEGATIVE_INFINITY;
      for (int i = 0; i < count; i++) max = Math.max(max, scores[i]);
      return max;
    }

    @Override
    public RandomVectorScorer forGraphTraversal() {
      // Each bulk call still passes the runtime's exact shape/profile qualification checks.
      return this;
    }

    @Override
    public float score(int node) throws IOException {
      runtime.checkpoint();
      return java.score(node);
    }

    @Override
    public int maxOrd() {
      return java.maxOrd();
    }

    @Override
    public int ordToDoc(int ord) {
      return java.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits accept) {
      return java.getAcceptOrds(accept);
    }

    @Override
    public KnnVectorValues values() {
      return original;
    }
  }

  private RandomVectorScorerSupplier supplier(RandomVectorScorerSupplier java) {
    return new RandomVectorScorerSupplier() {
      @Override
      public RandomVectorScorerSupplier copy() throws IOException {
        return supplier(java.copy());
      }

      @Override
      public UpdateableRandomVectorScorer scorer() throws IOException {
        // No native query/index state or scratch is shared between supplier copies.
        return java.scorer();
      }
    };
  }

  /** A whole caller batch has one publication boundary, even when native frames are partitioned. */
  private abstract class BulkScorer implements RandomVectorScorer, HasKnnVectorValues {
    final RandomVectorScorer java;
    final KnnVectorValues original;
    final Capability capability;
    final int dimension, rowBytes;

    BulkScorer(
        RandomVectorScorer java,
        KnnVectorValues original,
        Capability capability,
        int dimension,
        int rowBytes) {
      this.java = java;
      this.original = original;
      this.capability = capability;
      this.dimension = dimension;
      this.rowBytes = rowBytes;
    }

    abstract AccelerationProvider.Batch stage(
        RequestMemoryScope scope, int[] nodes, int offset, int count, ByteBuffer output)
        throws IOException;

    @Override
    public RandomVectorScorer forGraphTraversal() {
      return java.forGraphTraversal();
    }

    @Override
    public float score(int node) throws IOException {
      runtime.checkpoint();
      return java.score(node);
    }

    @Override
    public int maxOrd() {
      return java.maxOrd();
    }

    @Override
    public int ordToDoc(int ord) {
      return java.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits accept) {
      return java.getAcceptOrds(accept);
    }

    @Override
    public KnnVectorValues values() {
      return original;
    }

    @Override
    public float bulkScore(int[] nodes, float[] scores, int count) throws IOException {
      runtime.checkpoint();
      Objects.checkFromIndexSize(0, count, nodes.length);
      Objects.checkFromIndexSize(0, count, scores.length);
      if (count == 0) return Float.NEGATIVE_INFINITY;
      if (!runtime.eligible(capability, count)) return java.bulkScore(nodes, scores, count);
      for (int i = 0; i < count; i++) Objects.checkIndex(nodes[i], maxOrd());
      // The aggregate private output is charged for the entire call; input frames are reclaimed
      // after each synchronous kernel. No prefix reaches the collector on a later-frame failure.
      try (RequestMemoryScope parent = runtime.newCallScope()) {
        ByteBuffer output =
            parent
                .allocate(Math.multiplyExact((long) count, Float.BYTES))
                .order(ByteOrder.LITTLE_ENDIAN);
        int desired =
            Math.min(
                4096,
                Math.max(runtime.config().minimumBatch(), (1 << 20) / Math.max(1, rowBytes + 16)));
        int partitions = Math.toIntExact(((long) count + desired - 1) / desired);
        int chunk = Math.toIntExact(((long) count + partitions - 1) / partitions);
        for (int offset = 0; offset < count; ) {
          int size = Math.min(chunk, count - offset);
          boolean complete;
          try (RequestMemoryScope frame = runtime.newScope(parent)) {
            runtime.checkpoint();
            ByteBuffer slice =
                output
                    .duplicate()
                    .position(offset * Float.BYTES)
                    .limit((offset + size) * Float.BYTES)
                    .slice()
                    .order(ByteOrder.LITTLE_ENDIAN);
            var batch = stage(frame, nodes, offset, size, slice);
            complete =
                runtime.execute(batch, size, (long) size * rowBytes + rowBytes + size * 20L, frame);
          } catch (RequestMemoryScope.AdmissionException denied) {
            if (!denied.isCapacityFailure()) {
              runtime.checkpoint();
              throw new IOException("native scoring scope stopped", denied);
            }
            if (size <= runtime.config().minimumBatch()) throw denied;
            chunk = Math.max(1, size / 2);
            continue;
          }
          if (!complete) return java.bulkScore(nodes, scores, count);
          offset += size;
        }
        runtime.checkpoint();
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
          scores[i] = output.getFloat(i * Float.BYTES);
          max = Math.max(max, scores[i]);
        }
        return max;
      } catch (RequestMemoryScope.AdmissionException denied) {
        if (!denied.isCapacityFailure()) {
          runtime.checkpoint();
          throw new IOException("native scoring scope stopped", denied);
        }
        runtime.checkpoint();
        runtime.recordJava(capability, AccelerationRuntime.Reason.BUDGET_DENIED);
        return java.bulkScore(nodes, scores, count);
      }
    }
  }

  private final class QueryScorer extends BulkScorer {
    private final ByteVectorValues data;
    private final byte[] query;
    private final VectorSimilarityFunction similarity;
    private final float queryCorrection, multiplier;

    QueryScorer(
        RandomVectorScorer java,
        KnnVectorValues original,
        ByteVectorValues data,
        byte[] query,
        VectorSimilarityFunction similarity,
        float queryCorrection,
        float multiplier) {
      super(
          java,
          original,
          data instanceof LegacyQuantizedByteVectorValues
              ? Capability.QUANTIZED_RESCORE
              : Capability.VECTOR_I8,
          query.length,
          query.length);
      this.data = data;
      this.query = query;
      this.similarity = similarity;
      this.queryCorrection = queryCorrection;
      this.multiplier = multiplier;
    }

    @Override
    AccelerationProvider.Batch stage(
        RequestMemoryScope scope, int[] nodes, int offset, int count, ByteBuffer output)
        throws IOException {
      ByteBuffer q = scope.allocate(query.length);
      q.put(query).flip();
      int byteCount = Math.multiplyExact(count, query.length);
      ByteBuffer vectors = scope.allocate(byteCount);
      ByteBuffer correction =
          scope.allocate(Math.multiplyExact(count, Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer metadata = scope.allocate(256);
      ByteBuffer scratch = scope.allocate(0);
      for (int i = 0; i < count; i++) {
        if ((i & 63) == 0) runtime.checkpoint();
        byte[] vector = data.vectorValue(nodes[offset + i]);
        if (vector.length != query.length)
          throw new IllegalArgumentException("vector staging dimensions");
        vectors.put(vector);
        if (data instanceof LegacyQuantizedByteVectorValues legacy)
          correction.putFloat(legacy.getScoreCorrectionConstant(nodes[offset + i]));
      }
      vectors.flip();
      if (capability == Capability.QUANTIZED_RESCORE) correction.flip();
      parameters
          .putLong(0, count)
          .putLong(8, query.length)
          .putLong(16, query.length)
          .putLong(24, similarity.ordinal())
          .putLong(40, Integer.toUnsignedLong(Float.floatToRawIntBits(multiplier)))
          .putLong(48, Integer.toUnsignedLong(Float.floatToRawIntBits(queryCorrection)));
      var inputs = new java.util.ArrayList<AccelerationProvider.Input>();
      inputs.add(
          new AccelerationProvider.Input(q, query.length, AccelerationProvider.Encoding.BYTES));
      inputs.add(
          new AccelerationProvider.Input(vectors, byteCount, AccelerationProvider.Encoding.BYTES));
      if (capability == Capability.QUANTIZED_RESCORE)
        inputs.add(
            new AccelerationProvider.Input(correction, count, AccelerationProvider.Encoding.F32));
      return new AccelerationProvider.Batch(
          capability,
          inputs,
          new AccelerationProvider.Output(output, count),
          parameters,
          scratch,
          metadata);
    }
  }

  private final class FloatQueryScorer extends BulkScorer {
    private final FloatVectorValues data;
    private final float[] query;
    private final VectorSimilarityFunction similarity;

    FloatQueryScorer(
        RandomVectorScorer java,
        KnnVectorValues original,
        FloatVectorValues data,
        float[] query,
        VectorSimilarityFunction similarity) {
      super(java, original, Capability.VECTOR_F32, query.length, query.length * Float.BYTES);
      this.data = data;
      this.query = query;
      this.similarity = similarity;
    }

    @Override
    AccelerationProvider.Batch stage(
        RequestMemoryScope scope, int[] nodes, int offset, int count, ByteBuffer output)
        throws IOException {
      ByteBuffer q = scope.allocate(rowBytes).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : query) q.putFloat(value);
      q.flip();
      ByteBuffer rows =
          scope.allocate(Math.multiplyExact(count, rowBytes)).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < count; i++) {
        if ((i & 63) == 0) runtime.checkpoint();
        float[] row = data.vectorValue(nodes[offset + i]);
        if (row.length != dimension)
          throw new IllegalArgumentException("vector staging dimensions");
        for (float value : row) rows.putFloat(value);
      }
      rows.flip();
      ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      parameters
          .putLong(0, count)
          .putLong(8, dimension)
          .putLong(16, dimension)
          .putLong(24, similarity.ordinal())
          .putLong(32, Constants.HAS_FAST_SCALAR_FMA ? 1 : 0);
      return new AccelerationProvider.Batch(
          capability,
          List.of(
              new AccelerationProvider.Input(q, dimension, AccelerationProvider.Encoding.F32),
              new AccelerationProvider.Input(
                  rows, (long) count * dimension, AccelerationProvider.Encoding.F32)),
          new AccelerationProvider.Output(output, count),
          parameters,
          scope.allocate(0),
          scope.allocate(256));
    }
  }
}
