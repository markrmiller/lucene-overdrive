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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SmallFloat;

/**
 * Diagnostic prototype of fused BM25 block scoring (kernel 29): one native call reads each doc's
 * one-byte norm from the mapped norms file, looks up the norm-inverse cache and computes {@code
 * weight - weight / (1 + freq * inverse)} with the same float operations as {@code BM25Similarity}.
 * It replaces {@code NumericDocValues#longValues} plus the bulk similarity scorer for one postings
 * block. Docs and frequencies are staged; norms are read in place.
 *
 * <p>Owner-thread only. The leaf is pinned during each call. Native failures are never replayed.
 *
 * @lucene.experimental
 */
public final class Bm25Block implements AutoCloseable {
  /** Largest block one call accepts. */
  public static final int MAXIMUM_BLOCK = 512;

  private final AccelerationProvider.Kernel kernel;
  private final RequestMemoryScope scope;
  private final LeafReader leaf;
  private final Thread thread = Thread.currentThread();
  private final AccelerationProvider.Input norms, cache;
  private final ByteBuffer block, output, parameters, scratch, metadata;
  private final IntBuffer docView;
  private final FloatBuffer freqView, scoreView;
  private AccelerationProvider.Batch batch;
  private int batchSize = -1;
  private boolean closed, failed;

  /**
   * Prepares a block scorer for one term of {@code similarity}, or returns null when the field's
   * norms are not dense one-byte norms in a mapped file. The weight and norm cache are built with
   * the same float expressions as {@link BM25Similarity#scorer}.
   */
  public static Bm25Block prepare(
      AccelerationRuntime runtime,
      LeafReader leaf,
      String field,
      BM25Similarity similarity,
      float boost,
      CollectionStatistics collection,
      TermStatistics term)
      throws IOException {
    Objects.requireNonNull(runtime);
    // A subclass may override idf or the average length; only the exact formulas are mirrored.
    if (similarity.getClass() != BM25Similarity.class) return null;
    Bm25Block result = null;
    try (var pin = LeafReaderPin.tryAcquire(leaf)) {
      if (pin == null) throw new IOException("norms reader closed");
      NumericDocValues values = leaf.getNormValues(field);
      if (!(values instanceof HasIndexSlice sliced) || sliced.getSlice() == null) return null;
      ByteBuffer mapped = MappedSlices.slice(sliced.getSlice(), 0, leaf.maxDoc());
      if (mapped == null || !mapped.isDirect()) return null;
      float weight = boost * similarity.idfExplain(collection, term).getValue().floatValue();
      float avgdl = (float) (collection.sumTotalTermFreq() / (double) collection.docCount());
      float k1 = similarity.getK1(), b = similarity.getB();
      float[] cache = new float[256];
      for (int i = 0; i < cache.length; i++) {
        float length = SmallFloat.byte4ToInt((byte) i);
        cache[i] = 1f / (k1 * ((1 - b) + b * length / avgdl));
      }
      var kernel = runtime.diagnosticKernel(Capability.BM25_BLOCK);
      result = new Bm25Block(kernel, runtime.newQualificationScope(), leaf, mapped, weight, cache);
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

  private Bm25Block(
      AccelerationProvider.Kernel kernel,
      RequestMemoryScope scope,
      LeafReader leaf,
      ByteBuffer mapped,
      float weight,
      float[] normCache)
      throws IOException {
    this.kernel = kernel;
    this.scope = scope;
    this.leaf = leaf;
    try {
      norms =
          new AccelerationProvider.Input(
              mapped, leaf.maxDoc(), AccelerationProvider.Encoding.BYTES);
      ByteBuffer cached = scope.allocate(1024).order(ByteOrder.LITTLE_ENDIAN);
      cached.asFloatBuffer().put(normCache);
      cache = new AccelerationProvider.Input(cached, 256, AccelerationProvider.Encoding.F32);
      block = scope.allocate(MAXIMUM_BLOCK * 8L).order(ByteOrder.LITTLE_ENDIAN);
      output = scope.allocate(MAXIMUM_BLOCK * 4L).order(ByteOrder.LITTLE_ENDIAN);
      parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 8; i++) parameters.putLong(i * 8, 0);
      parameters
          .putLong(8, 1)
          .putLong(16, leaf.maxDoc())
          .putLong(24, Integer.toUnsignedLong(Float.floatToRawIntBits(weight)));
      scratch = scope.allocate(0);
      metadata = scope.allocate(256);
      docView = block.asIntBuffer();
      freqView = block.asFloatBuffer();
      scoreView = output.asFloatBuffer();
    } catch (RuntimeException | Error failure) {
      scope.close();
      throw failure;
    }
  }

  /** Scores {@code size} docs with their frequencies into {@code scores}. */
  public void score(int size, int[] docs, float[] freqs, float[] scores) throws IOException {
    if (thread != Thread.currentThread())
      throw new IllegalStateException("block scorer belongs to another thread");
    if (closed || failed) throw new IllegalStateException("block scorer unavailable");
    Objects.checkFromIndexSize(0, size, MAXIMUM_BLOCK + 1);
    if (size == 0) return;
    docView.put(0, docs, 0, size);
    freqView.put(size, freqs, 0, size);
    if (size != batchSize) {
      batch =
          new AccelerationProvider.Batch(
              Capability.BM25_BLOCK,
              List.of(
                  new AccelerationProvider.Input(
                      block.duplicate().limit(size * 8),
                      size * 8L,
                      AccelerationProvider.Encoding.BYTES),
                  norms,
                  cache),
              new AccelerationProvider.Output(output.duplicate().limit(size * 4), size),
              parameters,
              scratch,
              metadata);
      parameters.putLong(0, size);
      batchSize = size;
    }
    try (var pin = LeafReaderPin.tryAcquire(leaf)) {
      if (pin == null) throw new IOException("norms reader closed");
      var result = kernel.execute(batch);
      if (result == null
          || result.status() != AccelerationProvider.Status.OK
          || result.outputElements() != size) {
        failed = true;
        throw new IOException("BM25 block kernel did not return complete success: " + result);
      }
    } catch (IOException | RuntimeException | Error failure) {
      failed = true;
      throw failure;
    }
    scoreView.get(0, scores, 0, size);
  }

  @Override
  public void close() {
    if (closed) return;
    closed = true;
    scope.close();
  }
}
