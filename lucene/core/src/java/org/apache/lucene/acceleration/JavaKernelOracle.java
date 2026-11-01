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
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.ScalarQuantizedVectorSimilarity;

/** Bounded Java schemas and shadow reference. Writes only the supplied unpublished output. */
public final class JavaKernelOracle {
  private JavaKernelOracle() {}

  private static final long MAX = 16L << 20;
  private static final long BYTES = 128L << 20;

  private static ByteBuffer view(ByteBuffer b) {
    return b.duplicate().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static long[] params(AccelerationProvider.Batch b) {
    ByteBuffer p = view(b.parameters());
    long[] values = new long[8];
    for (int i = 0; i < 8; i++) values[i] = p.getLong(i * 8);
    return values;
  }

  /** Requires {@code mask} clear in each of the first {@code n} bytes, eight at a time. */
  private static void requireNoBits(ByteBuffer bytes, long n, int mask) {
    ByteBuffer data = view(bytes);
    long wide = (mask & 0xFFL) * 0x0101010101010101L;
    int i = 0;
    for (; i + 8 <= n; i += 8) require((data.getLong(i) & wide) == 0);
    for (; i < n; i++) require((data.get(i) & mask) == 0);
  }

  private static void require(boolean condition) {
    if (!condition) throw new IllegalArgumentException("invalid acceleration kernel schema");
  }

  private static long mul(long a, long b) {
    require(a >= 0 && b >= 0);
    return Math.multiplyExact(a, b);
  }

  private static void input(
      AccelerationProvider.Batch b, int index, AccelerationProvider.Encoding encoding, long count) {
    require(index < b.inputs().size());
    var in = b.inputs().get(index);
    require(
        in.encoding() == encoding
            && in.elements() >= count
            && in.byteSize() >= mul(count, encoding.width));
  }

  private static void shape(AccelerationProvider.Batch b, int inputs, long count, int width) {
    require(
        b.inputs().size() == inputs
            && b.output().elements() == count
            && b.output().bytes().remaining() >= mul(count, width));
  }

  /**
   * Conservative physical heap bound for one reference execution, including primitive arrays,
   * nested matrix references and headers, and quantizer interval temporaries. This is separate from
   * the native clone frame. Overflow denotes an unrepresentable admission request.
   */
  public static long referenceMemoryBytes(AccelerationProvider.Batch batch) {
    long frame =
        Math.addExact(
            384, Math.addExact(batch.output().bytes().remaining(), batch.scratch().remaining()));
    long[] parameters = params(batch);
    if (parameters[0] < 0) throw new IllegalArgumentException("negative reference shape");
    long elements = Math.addExact(batch.output().elements(), parameters[0]);
    if (batch.capability() == Capability.MULTI_INTERSECT) {
      if (parameters[1] < 0) throw new IllegalArgumentException("negative set count");
      elements = Math.addExact(elements, parameters[1]);
    }
    for (var input : batch.inputs()) {
      frame = Math.addExact(frame, input.byteSize());
      elements = Math.addExact(elements, input.elements());
    }
    return Math.addExact(
        Math.multiplyExact(
            Math.addExact(
                frame,
                Math.max(
                    4096L,
                    Math.multiplyExact(
                        256L,
                        org.apache.lucene.util.RamUsageEstimator.NUM_BYTES_OBJECT_ALIGNMENT))),
            8),
        Math.multiplyExact(
            elements,
            Math.max(
                64L,
                Math.multiplyExact(
                    4L, org.apache.lucene.util.RamUsageEstimator.NUM_BYTES_OBJECT_ALIGNMENT))));
  }

  /**
   * Finite-result domain of the exact float profile. Coordinate magnitudes leave a factor-four
   * margin for squared differences and accumulation; positive split-product cosine norms also
   * exclude zero and underflow. A false result selects the existing Java behavior, including its
   * assertions. Structural validation must precede this data-only check.
   */
  public static boolean supportsFloatDomain(AccelerationProvider.Batch batch) {
    if (batch.capability() != Capability.VECTOR_F32 && batch.capability() != Capability.MAXSIM)
      return true;
    long[] p = params(batch);
    int dimensions = Math.toIntExact(p[1]);
    if (dimensions <= 0) return false;
    boolean cosine = p[3] == 2;
    return safeFloatRows(
            view(batch.inputs().get(0).bytes()),
            batch.capability() == Capability.VECTOR_F32 ? 1 : Math.toIntExact(p[4]),
            dimensions,
            dimensions,
            cosine)
        && safeFloatRows(
            view(batch.inputs().get(1).bytes()),
            Math.toIntExact(batch.capability() == Capability.VECTOR_F32 ? p[0] : p[2]),
            dimensions,
            Math.toIntExact(batch.capability() == Capability.VECTOR_F32 ? p[2] : p[1]),
            cosine);
  }

  private static boolean safeFloatRows(
      ByteBuffer rows, int count, int dimensions, int stride, boolean cosine) {
    double maximum = Math.sqrt((double) Float.MAX_VALUE / (16.0 * dimensions));
    for (int row = 0; row < count; row++) {
      float norm = 0;
      for (int d = 0; d < dimensions; d++) {
        float value = rows.getFloat(Math.toIntExact(((long) row * stride + d) * 4));
        if (!Float.isFinite(value) || Math.abs((double) value) > maximum) return false;
        if (cosine) norm += value * value;
      }
      if (cosine && norm == 0) return false;
    }
    return true;
  }

  /** Checks complete operation descriptors and data constraints without writing output. */
  public static void validate(AccelerationProvider.Batch batch) throws IOException {
    for (var input : batch.inputs())
      if (input.span() != null && batch.capability() != Capability.MASKED_FACETS)
        throw new IllegalArgumentException("borrowed typed input unsupported for capability");
    long[] p = params(batch);
    long n = p[0];
    require(n >= 0);
    require(
        batch.inputs().size() >= 1
            && batch.inputs().size() <= 3
            && batch.scratch().remaining() <= BYTES);
    for (var in : batch.inputs()) require(in.elements() >= 0 && in.byteSize() <= BYTES);
    require(batch.output().bytes().remaining() <= BYTES);
    switch (batch.capability()) {
      case TEST_COPY -> {
        input(batch, 0, AccelerationProvider.Encoding.I64, n);
        shape(batch, 1, n, 8);
      }
      case VECTOR_I8, QUANTIZED_RESCORE -> {
        require(p[1] > 0 && p[1] <= 32768 && p[2] >= p[1] && p[3] >= 0 && p[3] <= 3);
        long rows = mul(n, p[2]);
        input(batch, 0, AccelerationProvider.Encoding.BYTES, p[1]);
        input(batch, 1, AccelerationProvider.Encoding.BYTES, rows);
        shape(batch, batch.capability() == Capability.VECTOR_I8 ? 2 : 3, n, 4);
        if (batch.capability() == Capability.QUANTIZED_RESCORE) {
          input(batch, 2, AccelerationProvider.Encoding.F32, n);
          for (int k = 0; k < 2; k++) {
            ByteBuffer data = view(batch.inputs().get(k).bytes());
            for (int i = 0; i < data.remaining(); i++) require(data.get(i) >= 0);
          }
        }
      }
      case VECTOR_F32 -> {
        require(
            p[1] > 0
                && p[1] <= 32768
                && p[2] >= p[1]
                && p[3] >= 0
                && p[3] <= 3
                && p[4] >= 0
                && p[4] <= 1);
        input(batch, 0, AccelerationProvider.Encoding.F32, p[1]);
        input(batch, 1, AccelerationProvider.Encoding.F32, mul(n, p[2]));
        shape(batch, 2, n, 4);
      }
      case MAXSIM -> {
        require(
            p[1] > 0
                && p[1] <= 32768
                && p[2] >= 0
                && p[3] >= 0
                && p[3] <= 3
                && p[4] >= 0
                && p[5] >= 0
                && p[6] >= 0
                && p[6] <= 1);
        input(batch, 0, AccelerationProvider.Encoding.F32, mul(p[4], p[1]));
        input(batch, 1, AccelerationProvider.Encoding.F32, mul(p[2], p[1]));
        input(batch, 2, AccelerationProvider.Encoding.I64, n + 1);
        shape(batch, 3, n, 4);
        ByteBuffer offsets = view(batch.inputs().get(2).bytes());
        require(offsets.getLong(0) == 0 && offsets.getLong(Math.toIntExact(n * 8)) == p[2]);
        for (int i = 0; i < n; i++) require(offsets.getLong(i * 8) <= offsets.getLong((i + 1) * 8));
      }
      case DOCSET_INTERSECT,
          DOCSET_UNION,
          DOCSET_DIFFERENCE,
          DOCSET_CARDINALITY,
          DOCSET_INTERSECT_COUNT,
          MULTI_INTERSECT -> {
        require(p[2] >= 0 && p[2] <= mul(n, 64) && (n == 0 || p[2] > (n - 1) * 64));
        boolean multi = batch.capability() == Capability.MULTI_INTERSECT,
            card = batch.capability() == Capability.DOCSET_CARDINALITY,
            counted = card || batch.capability() == Capability.DOCSET_INTERSECT_COUNT;
        require(!multi || p[1] > 0);
        input(batch, 0, AccelerationProvider.Encoding.I64, multi ? mul(n, p[1]) : n);
        if (!multi && !card) input(batch, 1, AccelerationProvider.Encoding.I64, n);
        shape(batch, multi || card ? 1 : 2, counted ? 1 : n, 8);
      }
      case DOCVALUES_DECODE, DOCVALUES_AGGREGATE -> {
        require(p[1] >= 8 && p[1] <= 64 && p[1] % 8 == 0);
        long stride = p[2] == 0 ? p[1] / 8 : p[2];
        require(stride >= p[1] / 8);
        input(batch, 0, AccelerationProvider.Encoding.BYTES, mul(n, stride));
        if (batch.capability() == Capability.DOCVALUES_DECODE) shape(batch, 1, n, 8);
        else {
          require(batch.inputs().size() <= 2);
          if (batch.inputs().size() == 2)
            input(batch, 1, AccelerationProvider.Encoding.I64, (n + 63) / 64);
          shape(batch, batch.inputs().size(), 6, 8);
        }
      }
      case DOCVALUES_RANGE -> {
        input(batch, 0, AccelerationProvider.Encoding.I64, n);
        shape(batch, 1, (n + 63) / 64, 8);
      }
      case DOCVALUES_RANGE_FILTER -> {
        input(batch, 0, AccelerationProvider.Encoding.I64, n);
        input(batch, 1, AccelerationProvider.Encoding.I64, (n + 63) / 64);
        shape(batch, 2, (n + 63) / 64, 8);
      }
      case POSTINGS_SPLIT -> {
        require(
            p[1] >= 1
                && p[1] <= 31
                && p[2] >= 1
                && p[2] <= 31
                && p[3] >= 0
                && p[3] <= 0xffffffffL
                && p[4] >= 0
                && p[4] <= 0xffffffffL);
        input(batch, 0, AccelerationProvider.Encoding.I32, n);
        shape(batch, 1, mul(n, (p[1] - 1) / p[2] + 2), 4);
      }
      case QUANTIZE_PACK, QUANTIZE_TRANSPOSE, QUANTIZE_BINARY, QUANTIZE_DIBIT -> {
        int bits =
            batch.capability() == Capability.QUANTIZE_BINARY
                ? 1
                : batch.capability() == Capability.QUANTIZE_DIBIT ? 2 : 4;
        input(batch, 0, AccelerationProvider.Encoding.BYTES, n);
        ByteBuffer in = view(batch.inputs().get(0).bytes());
        for (int i = 0; i < n; i++) require((in.get(i) & 255) < (1 << bits));
        if (batch.capability() == Capability.QUANTIZE_PACK) {
          require(n % 2 == 0);
          shape(batch, 1, n / 2, 1);
        } else shape(batch, 1, mul((n + 7) / 8, bits), 1);
      }
      case QUANTIZE_UNPACK -> {
        input(batch, 0, AccelerationProvider.Encoding.BYTES, n);
        shape(batch, 1, mul(n, 2), 1);
      }
      case FACETS_COUNT, SPARSE_FACETS -> {
        require(p[1] >= 0);
        input(batch, 0, AccelerationProvider.Encoding.I32, n);
        shape(batch, 1, batch.capability() == Capability.FACETS_COUNT ? p[1] : 1 + mul(n, 2), 4);
        ByteBuffer data = view(batch.inputs().get(0).bytes());
        int last = -1;
        for (int i = 0; i < n; i++) {
          int v = data.getInt(i * 4);
          require(v == -1 || (v >= 0 && v < p[1]));
          if (v != -1 && batch.capability() == Capability.SPARSE_FACETS) {
            require(v >= last);
            last = v;
          }
        }
      }
      case MASKED_FACETS -> {
        // p: 0 words, 1 buckets, 2 from bit, 3 to bit, 4 signed column offset, 5 width, 6 column
        // length. Ordinals are checked against p[1] while counting (a rejected call restores the
        // output), not by a pre-scan, so validation stays O(1) per call.
        require(p[1] >= 1 && (p[5] == 1 || p[5] == 2 || p[5] == 4) && p[6] >= 0);
        require(p[2] >= 0 && p[2] <= p[3] && p[3] <= mul(n, 64));
        require(
            p[2] == p[3] || (Math.addExact(p[2], p[4]) >= 0 && Math.addExact(p[3], p[4]) <= p[6]));
        input(batch, 0, AccelerationProvider.Encoding.I64, n);
        input(batch, 1, AccelerationProvider.Encoding.BYTES, mul(p[6], p[5]));
        shape(batch, 2, p[1], 4);
      }
      case SORTED_INTERSECT -> {
        require(p[1] >= 0 && p[2] >= 0);
        input(batch, 0, AccelerationProvider.Encoding.I32, n);
        input(batch, 1, AccelerationProvider.Encoding.I32, p[1]);
        shape(batch, 2, 1 + Math.min(n, p[1]), 4);
        for (int k = 0; k < 2; k++) {
          ByteBuffer data = view(batch.inputs().get(k).bytes());
          int last = -1;
          long len = k == 0 ? n : p[1];
          for (int i = 0; i < len; i++) {
            int v = data.getInt(i * 4);
            require(v > last && v < p[2]);
            last = v;
          }
        }
      }
      case MODERN_QUANTIZED_RESCORE -> {
        require(p[1] > 0 && p[1] <= 32768 && p[3] >= 0 && p[3] <= 3 && p[4] >= 0 && p[4] <= 4);
        long q = p[4] == 1 ? (p[1] + 1) / 2 * 2 : p[4] < 3 ? p[1] : 4 * ((p[1] + 7) / 8),
            d = p[4] == 1 ? (p[1] + 1) / 2 : p[4] < 3 ? p[1] : (p[4] - 2) * ((p[1] + 7) / 8);
        require(p[2] >= d);
        input(batch, 0, AccelerationProvider.Encoding.BYTES, q);
        input(batch, 1, AccelerationProvider.Encoding.BYTES, mul(n, p[2]));
        input(batch, 2, AccelerationProvider.Encoding.BYTES, 32 + mul(n, 16));
        shape(batch, 3, n, 4);
        if (p[4] == 1 || p[4] == 2) {
          ByteBuffer data = view(batch.inputs().get(0).bytes());
          for (int i = 0; i < q; i++) require((data.get(i) & 255) < (p[4] == 1 ? 16 : 128));
        }
      }
      case OSQ_GATHER -> {
        require(p[1] > 0 && p[1] <= 32768 && p[3] >= 0 && p[3] <= 3 && p[4] >= 0 && p[4] <= 4);
        long q = p[4] == 1 ? (p[1] + 1) / 2 * 2 : p[4] < 3 ? p[1] : 4 * ((p[1] + 7) / 8),
            d = p[4] == 1 ? (p[1] + 1) / 2 : p[4] < 3 ? p[1] : (p[4] - 2) * ((p[1] + 7) / 8);
        require(p[2] == d + 16);
        input(batch, 0, AccelerationProvider.Encoding.BYTES, q);
        input(batch, 2, AccelerationProvider.Encoding.I32, n);
        shape(batch, 3, n, 4);
        ByteBuffer ordinals = view(batch.inputs().get(2).bytes());
        long rows = 0;
        for (int i = 0; i < n; i++) {
          long ordinal = Integer.toUnsignedLong(ordinals.getInt(i * 4));
          rows = Math.max(rows, ordinal + 1);
        }
        input(batch, 1, AccelerationProvider.Encoding.BYTES, mul(rows, p[2]));
        if (p[4] == 1 || p[4] == 2)
          requireNoBits(batch.inputs().get(0).bytes(), q, p[4] == 1 ? 0xF0 : 0x80);
      }
      case OPTIMIZED_QUANTIZE -> {
        require(n > 0 && p[1] >= 1 && p[1] <= 8 && p[2] >= 0 && p[2] <= 3 && p[4] >= 0);
        input(batch, 0, AccelerationProvider.Encoding.F32, n);
        input(batch, 1, AccelerationProvider.Encoding.F32, n);
        shape(batch, 2, mul(n, 5) + 16, 1);
        require(batch.scratch().remaining() >= mul(n, 4));
      }
      case VECTOR_STATISTICS -> {
        require(p[1] > 0 && p[1] <= 32768 && p[2] >= p[1]);
        input(batch, 0, AccelerationProvider.Encoding.F32, mul(n, p[2]));
        shape(batch, 1, p[1], 4);
      }
      case POSTINGS_PREFIX -> {
        input(batch, 0, AccelerationProvider.Encoding.I32, n);
        shape(batch, 1, n, 4);
      }
      case BM25_BLOCK -> {
        require(p[1] == 1 && p[2] >= 0 && p[2] <= Integer.MAX_VALUE);
        input(batch, 0, AccelerationProvider.Encoding.BYTES, mul(n, 8));
        input(batch, 1, AccelerationProvider.Encoding.BYTES, p[2]);
        input(batch, 2, AccelerationProvider.Encoding.F32, 256);
        shape(batch, 3, n, 4);
        ByteBuffer block = view(batch.inputs().get(0).bytes());
        for (int i = 0; i < n; i++) require(Integer.toUnsignedLong(block.getInt(i * 4)) < p[2]);
      }
      case POSTINGS_BLOCK -> {
        require(n == 256 && p[1] >= 1 && p[1] <= 32);
        input(batch, 0, AccelerationProvider.Encoding.BYTES, p[1] * 32);
        shape(batch, 1, 256, 4);
      }
      case FLOATING_STATS -> {
        input(batch, 0, AccelerationProvider.Encoding.BYTES, mul(n, 8));
        input(batch, 1, AccelerationProvider.Encoding.BYTES, 48);
        if (batch.inputs().size() == 3)
          input(batch, 2, AccelerationProvider.Encoding.I64, (n + 63) / 64);
        shape(batch, batch.inputs().size(), 48, 1);
        require(view(batch.inputs().get(1).bytes()).getLong(0) >= 0);
      }
      case COMPRESSION -> {
        require(p[1] >= 0);
        input(batch, 0, AccelerationProvider.Encoding.BYTES, n);
        shape(batch, 1, p[1], 1);
        require(batch.scratch().remaining() >= p[1]);
        validateLz4(view(batch.inputs().get(0).bytes()), Math.toIntExact(n), Math.toIntExact(p[1]));
      }
    }
  }

  private static byte[] bytes(ByteBuffer b, int count) {
    byte[] a = new byte[count];
    b.duplicate().get(a);
    return a;
  }

  private static long[] longs(ByteBuffer b, int count) {
    long[] a = new long[count];
    for (int i = 0; i < count; i++) a[i] = b.getLong(i * 8);
    return a;
  }

  private static int[] ints(ByteBuffer b, int count) {
    int[] a = new int[count];
    for (int i = 0; i < count; i++) a[i] = b.getInt(i * 4);
    return a;
  }

  private static float[] floats(ByteBuffer b, int offset, int count) {
    float[] a = new float[count];
    for (int i = 0; i < count; i++) a[i] = b.getFloat((offset + i) * 4);
    return a;
  }

  private static void put(ByteBuffer b, long[] a) {
    for (int i = 0; i < a.length; i++) b.putLong(i * 8, a[i]);
  }

  private static void put(ByteBuffer b, int[] a) {
    for (int i = 0; i < a.length; i++) b.putInt(i * 4, a[i]);
  }

  private static float fp(long bits) {
    return Float.intBitsToFloat((int) bits);
  }

  /** Executes the established Java behavior for qualified shadow comparison. */
  public static AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
      throws IOException {
    validate(batch);
    long[] p = params(batch);
    int n = Math.toIntExact(p[0]),
        dim = Math.toIntExact(p[1]),
        stride =
            batch.capability() == Capability.POSTINGS_PREFIX ? (int) p[2] : Math.toIntExact(p[2]);
    if (n > MAX
        || batch.output().elements() > MAX
        || batch.capability() == Capability.MULTI_INTERSECT && p[1] > MAX)
      return new AccelerationProvider.Result(AccelerationProvider.Status.UNSUPPORTED, 0, 0);
    ByteBuffer
        a = batch.inputs().get(0).bytes() == null ? null : view(batch.inputs().get(0).bytes()),
        b =
            batch.inputs().size() > 1 && batch.inputs().get(1).bytes() != null
                ? view(batch.inputs().get(1).bytes())
                : null,
        out = view(batch.output().bytes());
    VectorSimilarityFunction similarity =
        VectorSimilarityFunction.values()[
            batch.capability() == Capability.OPTIMIZED_QUANTIZE
                ? (int) p[2]
                : batch.capability() == Capability.VECTOR_I8
                        || batch.capability() == Capability.QUANTIZED_RESCORE
                        || batch.capability() == Capability.VECTOR_F32
                        || batch.capability() == Capability.MAXSIM
                        || batch.capability() == Capability.MODERN_QUANTIZED_RESCORE
                        || batch.capability() == Capability.OSQ_GATHER
                    ? (int) p[3]
                    : 0];
    switch (batch.capability()) {
      case TEST_COPY -> put(out, longs(a, n));
      case VECTOR_I8, QUANTIZED_RESCORE -> {
        byte[] q = bytes(a, dim);
        var scorer =
            batch.capability() == Capability.QUANTIZED_RESCORE
                ? ScalarQuantizedVectorSimilarity.fromVectorSimilarity(
                    similarity, fp(p[5]), (byte) 7)
                : null;
        ByteBuffer c = scorer == null ? null : view(batch.inputs().get(2).bytes());
        for (int r = 0; r < n; r++) {
          ByteBuffer row = b.duplicate();
          row.position(r * stride);
          byte[] d = bytes(row, dim);
          out.putFloat(
              r * 4,
              scorer == null
                  ? similarity.compare(q, d)
                  : scorer.score(q, fp(p[6]), d, c.getFloat(r * 4)));
        }
      }
      case VECTOR_F32 -> {
        float[] q = floats(a, 0, dim);
        for (int r = 0; r < n; r++)
          out.putFloat(r * 4, similarity.compare(q, floats(b, r * stride, dim)));
      }
      case MAXSIM -> {
        ByteBuffer offsets = view(batch.inputs().get(2).bytes());
        float[][] q = new float[Math.toIntExact(p[4])][];
        for (int i = 0; i < q.length; i++) q[i] = floats(a, i * dim, dim);
        for (int r = 0; r < n; r++) {
          int start = Math.toIntExact(offsets.getLong(r * 8)),
              end = Math.toIntExact(offsets.getLong((r + 1) * 8));
          float[][] rows = new float[end - start][];
          for (int i = start; i < end; i++) rows[i - start] = floats(b, i * dim, dim);
          out.putFloat(r * 4, BlockOperations.maxSim(q, rows, similarity));
        }
      }
      case DOCSET_INTERSECT, DOCSET_UNION, DOCSET_DIFFERENCE ->
          put(
              out,
              BlockOperations.bitset(null, batch.capability(), longs(a, n), longs(b, n), stride));
      case DOCSET_CARDINALITY ->
          out.putLong(0, BlockOperations.cardinality(null, longs(a, n), stride));
      case DOCSET_INTERSECT_COUNT ->
          out.putLong(0, BlockOperations.intersectionCount(null, longs(a, n), longs(b, n), stride));
      case MULTI_INTERSECT -> {
        long[][] sets = new long[dim][];
        for (int s = 0; s < dim; s++) {
          ByteBuffer v = a.duplicate();
          v.position(s * n * 8);
          sets[s] = longs(v.slice().order(ByteOrder.LITTLE_ENDIAN), n);
        }
        put(out, BlockOperations.intersectMany(null, sets, stride));
      }
      case DOCVALUES_DECODE ->
          put(
              out,
              BlockOperations.decode(null, bytes(a, a.remaining()), 0, n, dim, stride, p[5], p[6]));
      case DOCVALUES_AGGREGATE -> {
        var state =
            BlockOperations.aggregate(
                null,
                bytes(a, a.remaining()),
                0,
                n,
                dim,
                stride,
                p[5],
                p[6],
                b == null ? null : longs(b, (n + 63) / 64),
                p[3],
                p[4]);
        put(
            out,
            new long[] {
              state.count(),
              state.sum(),
              state.minimum(),
              state.maximum(),
              state.hasValue() ? 1 : 0,
              state.overflow() ? 1 : 0
            });
      }
      case DOCVALUES_RANGE -> put(out, BlockOperations.range(null, longs(a, n), p[3], p[4]));
      case DOCVALUES_RANGE_FILTER ->
          put(
              out,
              BlockOperations.rangeFilter(
                  null, longs(a, n), longs(b, Math.toIntExact((n + 63) / 64)), p[3], p[4]));
      case POSTINGS_SPLIT ->
          put(out, BlockOperations.split(null, ints(a, n), dim, stride, (int) p[3], (int) p[4]));
      case QUANTIZE_PACK -> out.put(BlockOperations.packNibbles(null, bytes(a, n)));
      case QUANTIZE_TRANSPOSE, QUANTIZE_BINARY, QUANTIZE_DIBIT ->
          out.put(
              BlockOperations.transposeBits(
                  null,
                  bytes(a, n),
                  batch.capability() == Capability.QUANTIZE_BINARY
                      ? 1
                      : batch.capability() == Capability.QUANTIZE_DIBIT ? 2 : 4));
      case QUANTIZE_UNPACK -> out.put(BlockOperations.unpackNibbles(null, bytes(a, n)));
      case FACETS_COUNT -> put(out, BlockOperations.histogram(null, ints(a, n), dim));
      case MASKED_FACETS -> {
        // Overwrites the output; an out-of-range ordinal is rejected before any output write.
        int width = (int) p[5];
        int[] counts = new int[dim];
        for (long bit = p[2]; bit < p[3]; bit++) {
          if ((batch.inputs().get(0).littleEndianLong((bit >>> 6) * 8) >>> bit & 1) == 0) continue;
          int at = Math.toIntExact(bit + p[4]);
          long v = 0;
          for (int i = 0; i < width; i++)
            v |=
                Byte.toUnsignedLong(batch.inputs().get(1).byteAt((long) at * width + i)) << (i * 8);
          require(v < dim);
          counts[(int) v]++;
        }
        put(out, counts);
      }
      case SORTED_INTERSECT -> {
        int[] v = BlockOperations.intersectSorted(null, ints(a, n), ints(b, dim), stride);
        zero(out);
        out.putInt(0, v.length);
        for (int i = 0; i < v.length; i++) out.putInt((i + 1) * 4, v[i]);
      }
      case SPARSE_FACETS -> {
        int[] v = BlockOperations.sparseHistogram(null, ints(a, n), dim);
        zero(out);
        out.putInt(0, v.length / 2);
        for (int i = 0; i < v.length; i++) out.putInt((i + 1) * 4, v[i]);
      }
      case MODERN_QUANTIZED_RESCORE -> {
        ByteBuffer cor = view(batch.inputs().get(2).bytes());
        for (int r = 0; r < n; r++)
          out.putFloat(
              r * 4, modern(a, b, r * stride, cor, 32 + r * 16, cor, dim, (int) p[4], similarity));
      }
      case OSQ_GATHER -> {
        ByteBuffer ordinals = view(batch.inputs().get(2).bytes());
        ByteBuffer header = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        header.putLong(0, p[5]).putLong(8, p[6]).putLong(16, p[7]);
        int codes = Math.toIntExact(stride - 16L);
        for (int r = 0; r < n; r++) {
          int row = Math.multiplyExact(ordinals.getInt(r * 4), stride);
          out.putFloat(
              r * 4, modern(a, b, row, b, row + codes, header, dim, (int) p[4], similarity));
        }
      }
      case OPTIMIZED_QUANTIZE -> {
        var v =
            BlockOperations.quantize(
                null,
                floats(a, 0, n),
                floats(b, 0, n),
                (byte) dim,
                similarity,
                fp(p[3]),
                Math.toIntExact(p[4]));
        for (int i = 0; i < n; i++) out.putFloat(i * 4, v.centered()[i]);
        out.position(n * 4);
        out.put(v.quantized());
        out.putFloat(v.lowerInterval())
            .putFloat(v.upperInterval())
            .putFloat(v.additionalCorrection())
            .putInt(v.quantizedComponentSum());
      }
      case VECTOR_STATISTICS -> {
        float[][] rows = new float[n][];
        for (int r = 0; r < n; r++) rows[r] = floats(a, r * stride, dim);
        float[] centroid = BlockOperations.centroid(null, rows, dim);
        for (int i = 0; i < dim; i++) out.putFloat(i * 4, centroid[i]);
      }
      case POSTINGS_PREFIX -> put(out, BlockOperations.prefixSum(null, ints(a, n), stride));
      case BM25_BLOCK -> {
        ByteBuffer cache = view(batch.inputs().get(2).bytes());
        float weight = fp(p[3]);
        for (int i = 0; i < n; i++) {
          float inverse = cache.getFloat((b.get(a.getInt(i * 4)) & 0xFF) * 4);
          out.putFloat(i * 4, weight - weight / (1f + a.getFloat((n + i) * 4) * inverse));
        }
      }
      case POSTINGS_BLOCK -> decodeFor(a, dim, out);
      case FLOATING_STATS -> {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = a.getDouble(i * 8);
        var initial =
            new BlockOperations.FloatingStats(
                b.getLong(0),
                b.getDouble(8),
                b.getDouble(16),
                b.getDouble(24),
                b.getDouble(32),
                b.getDouble(40));
        var s =
            BlockOperations.floatingStats(
                null,
                v,
                batch.inputs().size() == 3
                    ? longs(view(batch.inputs().get(2).bytes()), (n + 63) / 64)
                    : null,
                initial);
        // NaN is written canonically: which NaN operand propagates is not part of the contract.
        out.putLong(s.count())
            .putLong(Double.doubleToLongBits(s.sum()))
            .putLong(Double.doubleToLongBits(s.min()))
            .putLong(Double.doubleToLongBits(s.max()))
            .putLong(Double.doubleToLongBits(s.mean()))
            .putLong(Double.doubleToLongBits(s.variance()));
      }
      case COMPRESSION -> out.put(BlockOperations.decompressLz4(null, bytes(a, n), dim));
    }
    return new AccelerationProvider.Result(
        AccelerationProvider.Status.OK, batch.output().elements(), 0);
  }

  private static void zero(ByteBuffer b) {
    for (int i = 0; i < b.remaining(); i++) b.put(i, (byte) 0);
  }

  private static int code(ByteBuffer v, int base, int i, int dim, int encoding, boolean query) {
    if (encoding == 0 || encoding == 2 || query && encoding == 1) return v.get(base + i) & 255;
    if (encoding == 1)
      return i < (dim + 1) / 2
          ? (v.get(base + i) & 255) >>> 4
          : v.get(base + i - (dim + 1) / 2) & 15;
    int blocks = (dim + 7) / 8, result = 0, planes = query ? 4 : encoding == 3 ? 1 : 2;
    for (int p = 0; p < planes; p++)
      result |= ((v.get(base + p * blocks + i / 8) >> (7 - i % 8)) & 1) << p;
    return result;
  }

  private static float modern(
      ByteBuffer query,
      ByteBuffer rows,
      int docAt,
      ByteBuffer rowCorrections,
      int at,
      ByteBuffer header,
      int dim,
      int encoding,
      VectorSimilarityFunction similarity) {
    int discrete = encoding == 1 ? (dim + 1) / 2 * 2 : encoding >= 3 ? (dim + 7) / 8 * 8 : dim;
    int dp = 0;
    for (int i = 0; i < discrete; i++)
      dp +=
          code(query, 0, i, discrete, encoding, true)
              * code(rows, docAt, i, discrete, encoding, false);
    float ax = rowCorrections.getFloat(at),
        ay = header.getFloat(0),
        lx =
            (rowCorrections.getFloat(at + 4) - ax)
                * (1f
                    / (encoding == 0
                        ? 255f
                        : encoding == 2 ? 127f : encoding == 1 ? 15f : encoding == 3 ? 1f : 3f)),
        ly = (header.getFloat(4) - ay) * (1f / (encoding == 0 ? 255f : encoding == 2 ? 127f : 15f));
    float score =
        ax * ay * dim
            + ay * lx * rowCorrections.getInt(at + 12)
            + ax * ly * header.getInt(12)
            + lx * ly * dp;
    if (similarity == VectorSimilarityFunction.EUCLIDEAN) {
      score = header.getFloat(8) + rowCorrections.getFloat(at + 8) - 2 * score;
      return 1f / (1f + Math.max(score, 0f));
    }
    score += header.getFloat(8) + rowCorrections.getFloat(at + 8) - header.getFloat(16);
    return similarity == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
        ? (score < 0 ? 1f / (1f + -1f * score) : score + 1f)
        : (1f + Math.max(-1f, Math.min(1f, score))) / 2f;
  }

  static void decodeFor(ByteBuffer src, int bits, ByteBuffer dst) {
    int primitive = bits <= 8 ? 8 : bits <= 16 ? 16 : 32,
        lanes = 32 / primitive,
        words = bits * 8,
        q = primitive / bits,
        rem = primitive % bits,
        mask = bits == 32 ? -1 : (1 << bits) - 1;
    for (int lane = 0; lane < lanes; lane++) {
      int base = lane * (256 / lanes), shift = (lanes - 1 - lane) * primitive;
      for (int group = 0; group < q; group++)
        for (int w = 0; w < words; w++)
          dst.putInt(
              (base + group * words + w) * 4,
              (src.getInt(w * 4) >>> (shift + primitive - (group + 1) * bits)) & mask);
      int at = q * words, slot = 0, used = 0;
      while (at < 256 / lanes) {
        int v = 0, needed = bits;
        while (needed > 0) {
          int take = Math.min(needed, rem - used);
          v =
              (v << take)
                  | ((src.getInt(slot * 4) >>> (shift + rem - used - take)) & ((1 << take) - 1));
          needed -= take;
          used += take;
          if (used == rem) {
            slot++;
            used = 0;
          }
        }
        dst.putInt((base + at++) * 4, v);
      }
    }
  }

  private static void validateLz4(ByteBuffer src, int size, int length) throws IOException {
    int i = 0, j = 0;
    while (i < size) {
      int token = src.get(i++) & 255, literals = token >>> 4;
      if (literals == 15) {
        int x;
        do {
          require(i < size);
          x = src.get(i++) & 255;
          literals = Math.addExact(literals, x);
        } while (x == 255);
      }
      require(literals <= size - i && literals <= length - j);
      i += literals;
      j += literals;
      if (j == length) {
        require(i == size);
        return;
      }
      require(size - i >= 2);
      int offset = src.getShort(i) & 65535;
      i += 2;
      require(offset > 0 && offset <= j);
      int match = token & 15;
      if (match == 15) {
        int x;
        do {
          require(i < size);
          x = src.get(i++) & 255;
          match = Math.addExact(match, x);
        } while (x == 255);
      }
      match = Math.addExact(match, 4);
      require(match <= length - j);
      j += match;
    }
    require(j == length);
  }
}
