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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Constants;
import org.apache.lucene.util.SmallFloat;
import org.apache.lucene.util.VectorUtil;

/** Deterministic startup known-answer tests; no frozen runtime or request discovery is involved. */
final class KernelFixtures {
  private KernelFixtures() {}

  /** Packed-nibble fixtures deliberately append one zero coordinate to odd dimensions. */
  private static byte[] zeroPad(byte[] values) {
    byte[] padded = new byte[values.length + 1];
    System.arraycopy(values, 0, padded, 0, values.length);
    return padded;
  }

  record Fixture(BlockOperations.Operation operation, byte[] expected) {}

  /** Each fixture owns a separate bounded scope so qualification does not accumulate scratch. */
  static boolean verify(
      Capability capability,
      AccelerationProvider.Kernel kernel,
      org.apache.lucene.util.IOSupplier<RequestMemoryScope> scopes)
      throws IOException {
    List<Fixture> fixtures = fixtures(capability);
    if (fixtures.isEmpty()) return false;
    for (Fixture fixture : fixtures) {
      try (var scope = scopes.get()) {
        BlockOperations.Operation operation = fixture.operation();
        AccelerationProvider.Estimate estimate =
            kernel.estimate(
                new AccelerationProvider.Shape(
                    operation.items(),
                    Math.toIntExact(operation.parameters()[1]),
                    operation.stagedBytes(),
                    new HashSet<>(
                        operation.inputs().stream().map(BlockOperations.Source::encoding).toList()),
                    AccelerationProvider.LayoutKind.NATIVE_BUFFER,
                    1,
                    true));
        if (!estimate.supported() || estimate.scratchBytes() > operation.scratchBytes())
          return false;
        AccelerationProvider.Batch batch = operation.stage(scope);
        // A deterministic poison catches skipped initialization, including empty aggregate fields.
        ByteBuffer output = batch.output().bytes().duplicate();
        for (int i = 0; i < output.remaining(); i++) output.put(i, (byte) 0xa5);
        AccelerationProvider.Result result = kernel.execute(batch);
        if (result == null
            || result.status() != AccelerationProvider.Status.OK
            || result.outputElements() != operation.outputElements()
            || result.detail() != 0) return false;
        if (output.remaining() != fixture.expected().length) return false;
        for (int i = 0; i < fixture.expected().length; i++)
          if (output.get(i) != fixture.expected()[i]) return false;
      }
    }
    return true;
  }

  static List<Fixture> fixtures(Capability capability) throws IOException {
    List<Fixture> fixtures = new ArrayList<>();
    for (boolean empty : new boolean[] {false, true}) {
      switch (capability) {
        case TEST_COPY -> {
          long[] input =
              empty ? new long[0] : new long[] {-1L, 0, Long.MIN_VALUE, 0x0102030405060708L};
          fixtures.add(
              longs(
                  new BlockOperations.Operation(
                      capability,
                      input.length,
                      BlockOperations.parameters(input.length),
                      List.of(BlockOperations.longs(input, 0, input.length)),
                      input.length,
                      8),
                  input));
        }
        case DOCSET_INTERSECT,
            DOCSET_UNION,
            DOCSET_DIFFERENCE,
            DOCSET_CARDINALITY,
            DOCSET_INTERSECT_COUNT -> {
          int bits = empty ? 0 : 130;
          long[] a = empty ? new long[0] : new long[] {-1, 0x1020304050607080L, -1};
          long[] b = empty ? new long[0] : new long[] {2, 0x0102030405060708L, -1};
          long[] params = BlockOperations.parameters(a.length);
          params[2] = bits;
          List<BlockOperations.Source> sources =
              capability == Capability.DOCSET_CARDINALITY
                  ? List.of(BlockOperations.longs(a, 0, a.length))
                  : List.of(
                      BlockOperations.longs(a, 0, a.length), BlockOperations.longs(b, 0, b.length));
          long[] expected =
              capability == Capability.DOCSET_CARDINALITY
                  ? new long[] {BlockOperations.cardinality(null, a, bits)}
                  : capability == Capability.DOCSET_INTERSECT_COUNT
                      ? new long[] {BlockOperations.intersectionCount(null, a, b, bits)}
                      : BlockOperations.bitset(null, capability, a, b, bits);
          fixtures.add(
              longs(
                  new BlockOperations.Operation(
                      capability, a.length, params, sources, expected.length, 8),
                  expected));
        }
        case DOCVALUES_DECODE, DOCVALUES_AGGREGATE -> {
          for (int bits : new int[] {8, 16, 24, 32, 40, 48, 56, 64}) {
            int count = empty ? 0 : 5, stride = bits / 8 + 1;
            byte[] input = new byte[count * stride];
            new Random(0x1052L + bits).nextBytes(input);
            long[] params = BlockOperations.parameters(count);
            params[1] = bits;
            params[2] = stride;
            params[3] = Long.MIN_VALUE;
            params[4] = Long.MAX_VALUE;
            params[5] = 3;
            params[6] = -7;
            List<BlockOperations.Source> sources =
                List.of(BlockOperations.bytes(input, 0, input.length));
            long[] expected;
            if (capability == Capability.DOCVALUES_DECODE)
              expected = BlockOperations.decode(null, input, 0, count, bits, stride, 3, -7);
            else {
              var aggregate =
                  BlockOperations.aggregate(
                      null,
                      input,
                      0,
                      count,
                      bits,
                      stride,
                      3,
                      -7,
                      null,
                      Long.MIN_VALUE,
                      Long.MAX_VALUE);
              expected = aggregateWords(aggregate);
            }
            fixtures.add(
                longs(
                    new BlockOperations.Operation(
                        capability, count, params, sources, expected.length, 8),
                    expected));
          }
          if (capability == Capability.DOCVALUES_AGGREGATE && !empty) {
            ByteBuffer input = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
            input.putLong(Long.MAX_VALUE).putLong(1).putLong(-1);
            long[] live = {3};
            long[] params = BlockOperations.parameters(3);
            params[1] = 64;
            params[2] = 8;
            params[3] = Long.MIN_VALUE;
            params[4] = Long.MAX_VALUE;
            params[5] = 1;
            var aggregate =
                BlockOperations.aggregate(
                    null, input.array(), 0, 3, 64, 8, 1, 0, live, Long.MIN_VALUE, Long.MAX_VALUE);
            fixtures.add(
                longs(
                    new BlockOperations.Operation(
                        capability,
                        3,
                        params,
                        List.of(
                            BlockOperations.bytes(input.array(), 0, 24),
                            BlockOperations.longs(live, 0, 1)),
                        6,
                        8),
                    aggregateWords(aggregate)));
          }
        }
        case DOCVALUES_RANGE_FILTER -> {
          long[] input = new long[empty ? 0 : 130];
          for (int i = 0; i < input.length; i++) input[i] = (i % 2 == 0 ? i : -i) * 3L;
          long[] selection = empty ? new long[0] : new long[] {-1L, 0, 0x5555555555555555L};
          long[] params = BlockOperations.parameters(input.length);
          params[3] = -150;
          params[4] = 200;
          long[] expected = BlockOperations.rangeFilter(null, input, selection, -150, 200);
          fixtures.add(
              longs(
                  new BlockOperations.Operation(
                      capability,
                      input.length,
                      params,
                      List.of(
                          BlockOperations.longs(input, 0, input.length),
                          BlockOperations.longs(selection, 0, selection.length)),
                      expected.length,
                      8),
                  expected));
        }
        case DOCVALUES_RANGE -> {
          long[] input =
              empty ? new long[0] : new long[] {Long.MIN_VALUE, -7, 0, 7, Long.MAX_VALUE};
          long[] params = BlockOperations.parameters(input.length);
          params[3] = -7;
          params[4] = 7;
          long[] expected = BlockOperations.range(null, input, -7, 7);
          fixtures.add(
              longs(
                  new BlockOperations.Operation(
                      capability,
                      input.length,
                      params,
                      List.of(BlockOperations.longs(input, 0, input.length)),
                      expected.length,
                      8),
                  expected));
        }
        case POSTINGS_SPLIT -> {
          int[] input = empty ? new int[0] : new int[] {0xabcdef12, 0x12345678, -1, 0};
          long[] params = BlockOperations.parameters(input.length);
          params[1] = 24;
          params[2] = 8;
          params[3] = 255;
          params[4] = 255;
          int[] expected = BlockOperations.split(null, input, 24, 8, 255, 255);
          fixtures.add(
              ints(
                  new BlockOperations.Operation(
                      capability,
                      input.length,
                      params,
                      List.of(BlockOperations.ints(input, 0, input.length)),
                      expected.length,
                      4),
                  expected));
        }
        case FACETS_COUNT -> {
          int[] input = empty ? new int[0] : new int[] {2, 0, -1, 2, 3, -1};
          long[] params = BlockOperations.parameters(input.length);
          params[1] = 5;
          int[] expected = BlockOperations.histogram(null, input, 5);
          fixtures.add(
              ints(
                  new BlockOperations.Operation(
                      capability,
                      input.length,
                      params,
                      List.of(BlockOperations.ints(input, 0, input.length)),
                      expected.length,
                      4),
                  expected));
        }
        case QUANTIZE_PACK, QUANTIZE_TRANSPOSE -> {
          byte[] input =
              empty
                  ? new byte[0]
                  : capability == Capability.QUANTIZE_PACK
                      ? new byte[] {0, 1, 7, 15, 15, 2, 8, 0}
                      : new byte[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15};
          byte[] expected =
              capability == Capability.QUANTIZE_PACK
                  ? BlockOperations.packNibbles(null, input)
                  : BlockOperations.transposeNibbles(null, input);
          fixtures.add(
              new Fixture(
                  new BlockOperations.Operation(
                      capability,
                      input.length,
                      BlockOperations.parameters(input.length),
                      List.of(BlockOperations.bytes(input, 0, input.length)),
                      expected.length,
                      1),
                  expected));
        }
        case VECTOR_I8, QUANTIZED_RESCORE -> {
          for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
            byte[] query =
                capability == Capability.VECTOR_I8
                    ? new byte[] {1, -2, 3, -4, 127}
                    : new byte[] {1, 2, 3, 4, 127};
            byte[][] documents =
                empty
                    ? new byte[0][]
                    : new byte[][] {
                      query,
                      capability == Capability.VECTOR_I8
                          ? new byte[] {-1, 2, -3, 4, -128}
                          : new byte[] {127, 4, 3, 2, 1},
                      new byte[] {4, 3, 2, 1, 9}
                    };
            float[] corrections = new float[documents.length];
            for (int i = 0; i < corrections.length; i++) corrections[i] = i % 2 == 0 ? .25f : -.75f;
            fixtures.add(
                vector(capability, query, documents, corrections, similarity, .01f, .125f));
          }
        }
        case SORTED_INTERSECT,
            MULTI_INTERSECT,
            SPARSE_FACETS,
            MODERN_QUANTIZED_RESCORE,
            OPTIMIZED_QUANTIZE,
            VECTOR_STATISTICS,
            POSTINGS_PREFIX,
            QUANTIZE_UNPACK,
            QUANTIZE_DIBIT,
            QUANTIZE_BINARY,
            POSTINGS_BLOCK,
            FLOATING_STATS,
            BM25_BLOCK,
            OSQ_GATHER,
            MASKED_FACETS,
            VECTOR_F32,
            MAXSIM,
            COMPRESSION -> {
          if (!empty) fixtures.addAll(extended(capability));
        }
      }
    }
    if (capability == Capability.VECTOR_I8) {
      fixtures.add(
          vector(
              capability,
              new byte[5],
              new byte[][] {new byte[5], {1, -2, 3, -4, 127}},
              new float[2],
              VectorSimilarityFunction.COSINE,
              1,
              0));
    }
    if (capability == Capability.QUANTIZED_RESCORE) {
      Random random = new Random(0x105213);
      for (int iteration = 0; iteration < 12; iteration++) {
        byte[] query = new byte[1 + random.nextInt(63)];
        byte[][] documents = new byte[3][query.length];
        for (int i = 0; i < query.length; i++) query[i] = (byte) random.nextInt(128);
        for (byte[] document : documents)
          for (int i = 0; i < document.length; i++) document[i] = (byte) random.nextInt(128);
        float[] corrections = {
          random.nextFloat() - .5f, random.nextFloat() - .5f, random.nextFloat() - .5f
        };
        for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values())
          fixtures.add(
              vector(
                  capability,
                  query,
                  documents,
                  corrections,
                  similarity,
                  random.nextFloat() / 100,
                  random.nextFloat() - .5f));
      }
    }
    return List.copyOf(fixtures);
  }

  private static Fixture vector(
      Capability capability,
      byte[] query,
      byte[][] documents,
      float[] corrections,
      VectorSimilarityFunction similarity,
      float multiplier,
      float queryCorrection)
      throws IOException {
    long[] params = BlockOperations.parameters(documents.length);
    params[1] = query.length;
    params[2] = query.length;
    params[3] = similarity.ordinal();
    params[5] = Integer.toUnsignedLong(Float.floatToRawIntBits(multiplier));
    params[6] = Integer.toUnsignedLong(Float.floatToRawIntBits(queryCorrection));
    BlockOperations.Source rows =
        new BlockOperations.Source(
            (long) documents.length * query.length,
            AccelerationProvider.Encoding.BYTES,
            b -> {
              for (byte[] document : documents) b.put(document);
            });
    List<BlockOperations.Source> sources =
        capability == Capability.VECTOR_I8
            ? List.of(BlockOperations.bytes(query, 0, query.length), rows)
            : List.of(
                BlockOperations.bytes(query, 0, query.length),
                rows,
                BlockOperations.floats(corrections));
    float[] expected =
        capability == Capability.VECTOR_I8
            ? BlockOperations.byteScores(null, query, documents, similarity)
            : BlockOperations.quantizedRescore(
                null, query, documents, corrections, similarity, multiplier, queryCorrection);
    ByteBuffer encoded = ByteBuffer.allocate(expected.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : expected) encoded.putInt(Float.floatToRawIntBits(value));
    return new Fixture(
        new BlockOperations.Operation(
            capability, documents.length, params, sources, expected.length, 4),
        encoded.array());
  }

  private static Fixture oracle(BlockOperations.Operation operation) throws IOException {
    List<AccelerationProvider.Input> inputs = new ArrayList<>();
    for (var source : operation.inputs()) {
      ByteBuffer b =
          ByteBuffer.allocate(Math.toIntExact(source.bytes())).order(ByteOrder.LITTLE_ENDIAN);
      source.writer().write(b);
      b.clear();
      inputs.add(new AccelerationProvider.Input(b, source.elements(), source.encoding()));
    }
    ByteBuffer
        output =
            ByteBuffer.allocate(Math.toIntExact(operation.outputBytes()))
                .order(ByteOrder.LITTLE_ENDIAN),
        params = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 8; i++) params.putLong(i * 8, operation.parameters()[i]);
    var batch =
        new AccelerationProvider.Batch(
            operation.capability(),
            inputs,
            new AccelerationProvider.Output(output, operation.outputElements()),
            params,
            ByteBuffer.allocate(Math.toIntExact(operation.scratchBytes())),
            ByteBuffer.allocate(256));
    var result = JavaKernelOracle.execute(batch);
    if (result.status() != AccelerationProvider.Status.OK)
      throw new IOException("unsupported Java KAT shape");
    return new Fixture(operation, output.array());
  }

  private static List<Fixture> extended(Capability cap) throws IOException {
    List<Fixture> result = new ArrayList<>();
    long[] p;
    switch (cap) {
      case SORTED_INTERSECT -> {
        int[] a = {1, 3, 7, 19}, b = {3, 6, 7};
        p = BlockOperations.parameters(a.length);
        p[1] = b.length;
        p[2] = 20;
        result.add(
            oracle(
                new BlockOperations.Operation(
                    cap,
                    a.length,
                    p,
                    List.of(
                        BlockOperations.ints(a, 0, a.length), BlockOperations.ints(b, 0, b.length)),
                    4,
                    4)));
      }
      case MULTI_INTERSECT -> {
        long[] a = {-1, -1, 2, -1, 3, -1};
        p = BlockOperations.parameters(2);
        p[1] = 3;
        p[2] = 65;
        result.add(
            oracle(
                new BlockOperations.Operation(
                    cap, 2, p, List.of(BlockOperations.longs(a, 0, a.length)), 2, 8)));
      }
      case MASKED_FACETS -> {
        // Two words; counted bits 3..100 map to column slots bit - 2; one-, two- and four-byte
        // ordinals.
        long[] words = {0b1011_1001_1000L | (1L << 63), 0x8000_0000_0000_0011L};
        for (int width : new int[] {1, 2, 4}) {
          int columnLength = 120;
          byte[] column = new byte[columnLength * width];
          ByteBuffer c = ByteBuffer.wrap(column).order(ByteOrder.LITTLE_ENDIAN);
          for (int i = 0; i < columnLength; i++) {
            int v = (i * 7 + 3) % 9;
            if (width == 1) c.put(i, (byte) v);
            else if (width == 2) c.putShort(i * 2, (short) v);
            else c.putInt(i * 4, v);
          }
          p = BlockOperations.parameters(words.length);
          p[1] = 9;
          p[2] = 3;
          p[3] = 101;
          p[4] = -2;
          p[5] = width;
          p[6] = columnLength;
          result.add(
              oracle(
                  new BlockOperations.Operation(
                      cap,
                      words.length,
                      p,
                      List.of(
                          BlockOperations.longs(words, 0, words.length),
                          BlockOperations.bytes(column, 0, column.length)),
                      9,
                      4)));
        }
      }
      case SPARSE_FACETS -> {
        int[] a = {-1, 1, 1, 7, 7, 7};
        p = BlockOperations.parameters(a.length);
        p[1] = 8;
        result.add(
            oracle(
                new BlockOperations.Operation(
                    cap,
                    a.length,
                    p,
                    List.of(BlockOperations.ints(a, 0, a.length)),
                    1 + a.length * 2,
                    4)));
      }
      case QUANTIZE_UNPACK, QUANTIZE_DIBIT, QUANTIZE_BINARY -> {
        byte[] a =
            cap == Capability.QUANTIZE_UNPACK
                ? new byte[] {0x12, 0x3f}
                : cap == Capability.QUANTIZE_DIBIT
                    ? new byte[] {0, 1, 2, 3, 0, 1, 2, 3, 3}
                    : new byte[] {1, 0, 1, 1, 0, 0, 0, 1, 1};
        p = BlockOperations.parameters(a.length);
        int size =
            cap == Capability.QUANTIZE_UNPACK
                ? a.length * 2
                : (a.length + 7) / 8 * (cap == Capability.QUANTIZE_DIBIT ? 2 : 1);
        result.add(
            oracle(
                new BlockOperations.Operation(
                    cap, a.length, p, List.of(BlockOperations.bytes(a, 0, a.length)), size, 1)));
      }
      case VECTOR_F32 -> {
        java.util.TreeSet<Integer> dimensions = new java.util.TreeSet<>();
        int limit = VectorUtil.exactScalarFloatLimit();
        for (int d = 1; d <= Math.min(limit, 32); d++) dimensions.add(d);
        for (int d : new int[] {33, 63, 64, 65, 128, 384, 513, 32768})
          if (d <= limit) dimensions.add(d);
        for (int d : dimensions)
          for (var metric : VectorSimilarityFunction.values()) {
            float[] q = new float[d];
            float[][] rows = new float[3][d];
            Random random = new Random(0x1052 + d);
            for (int i = 0; i < d; i++) {
              q[i] = (random.nextFloat() - .5f) * 2;
              rows[0][i] = q[i];
              rows[1][i] = -q[i];
              rows[2][i] = (random.nextFloat() - .5f) * 4;
            }
            p = BlockOperations.parameters(rows.length);
            p[1] = d;
            p[2] = d;
            p[3] = metric.ordinal();
            p[4] = Constants.HAS_FAST_SCALAR_FMA ? 1 : 0;
            result.add(
                oracle(
                    new BlockOperations.Operation(
                        cap,
                        rows.length,
                        p,
                        List.of(BlockOperations.floats(q), BlockOperations.matrix(rows, d)),
                        rows.length,
                        4)));
          }
      }
      case MAXSIM -> {
        java.util.TreeSet<Integer> dimensions = new java.util.TreeSet<>();
        int limit = VectorUtil.exactScalarFloatLimit();
        for (int d = 1; d <= Math.min(limit, 32); d++) dimensions.add(d);
        for (int d : new int[] {33, 63, 64, 65, 128, 384, 513, 32768})
          if (d <= limit) dimensions.add(d);
        for (int dim : dimensions)
          for (var metric : VectorSimilarityFunction.values()) {
            float[][] q = new float[2][dim], rows = new float[3][dim];
            Random random = new Random(0x101 + dim);
            for (int d = 0; d < dim; d++) {
              q[0][d] = (random.nextFloat() - .5f) * 2;
              q[1][d] = -q[0][d];
              rows[0][d] = q[0][d];
              rows[1][d] = -q[0][d];
              rows[2][d] = (random.nextFloat() - .5f) * 4;
            }
            long[] offsets = {0, 2, 2, 3};
            p = BlockOperations.parameters(3);
            p[1] = dim;
            p[2] = 3;
            p[3] = metric.ordinal();
            p[4] = 2;
            p[5] = 6L * dim;
            p[6] = Constants.HAS_FAST_SCALAR_FMA ? 1 : 0;
            result.add(
                oracle(
                    new BlockOperations.Operation(
                        cap,
                        3,
                        p,
                        List.of(
                            BlockOperations.matrix(q, dim),
                            BlockOperations.matrix(rows, dim),
                            BlockOperations.longs(offsets, 0, offsets.length)),
                        3,
                        4)));
          }
        for (var metric : VectorSimilarityFunction.values()) {
          float[][] q = {{-1, -2, -3}}, rows = {{1, 2, 3}, {3, 2, 1}};
          long[] offsets = {0, 1, 2};
          p = BlockOperations.parameters(2);
          p[1] = 3;
          p[2] = 2;
          p[3] = metric.ordinal();
          p[4] = 1;
          p[5] = 6;
          p[6] = Constants.HAS_FAST_SCALAR_FMA ? 1 : 0;
          result.add(
              oracle(
                  new BlockOperations.Operation(
                      cap,
                      2,
                      p,
                      List.of(
                          BlockOperations.matrix(q, 3),
                          BlockOperations.matrix(rows, 3),
                          BlockOperations.longs(offsets, 0, offsets.length)),
                      2,
                      4)));
          long[] emptyOffsets = {0, 0, 0};
          p = BlockOperations.parameters(2);
          p[1] = 3;
          p[3] = metric.ordinal();
          p[6] = Constants.HAS_FAST_SCALAR_FMA ? 1 : 0;
          result.add(
              oracle(
                  new BlockOperations.Operation(
                      cap,
                      2,
                      p,
                      List.of(
                          BlockOperations.matrix(new float[0][], 3),
                          BlockOperations.matrix(new float[0][], 3),
                          BlockOperations.longs(emptyOffsets, 0, emptyOffsets.length)),
                      2,
                      4)));
          if (metric != VectorSimilarityFunction.COSINE) {
            float[][] zeros = {{0f, -0f, 0f}, {-0f, 0f, -0f}};
            p = BlockOperations.parameters(2);
            p[1] = 3;
            p[2] = 2;
            p[3] = metric.ordinal();
            p[4] = 2;
            p[5] = 12;
            p[6] = Constants.HAS_FAST_SCALAR_FMA ? 1 : 0;
            result.add(
                oracle(
                    new BlockOperations.Operation(
                        cap,
                        2,
                        p,
                        List.of(
                            BlockOperations.matrix(zeros, 3),
                            BlockOperations.matrix(zeros, 3),
                            BlockOperations.longs(offsets, 0, offsets.length)),
                        2,
                        4)));
          }
        }
      }
      case OSQ_GATHER -> {
        // Two mapped rows [codes | lower, upper, additional, sum], gathered out of order.
        for (int encoding = 0; encoding <= 4; encoding++)
          for (var metric : VectorSimilarityFunction.values()) {
            int d = 19, max = encoding == 0 ? 255 : encoding == 2 ? 127 : 15;
            byte[] q = new byte[d], doc = new byte[d], other = new byte[d];
            for (int i = 0; i < d; i++) {
              int levels = encoding >= 3 ? encoding == 3 ? 2 : 4 : max;
              q[i] = (byte) ((i * 7) % max);
              doc[i] = (byte) ((i * 3) % levels);
              other[i] = (byte) ((i * 5 + 1) % levels);
            }
            int qsum = 0;
            for (byte v : q) qsum += v & 255;
            if (encoding == 1) {
              doc = BlockOperations.packNibbles(null, zeroPad(doc));
              other = BlockOperations.packNibbles(null, zeroPad(other));
              q = zeroPad(q);
            } else if (encoding >= 3) {
              q = BlockOperations.transposeBits(null, q, 4);
              doc = BlockOperations.transposeBits(null, doc, encoding == 3 ? 1 : 2);
              other = BlockOperations.transposeBits(null, other, encoding == 3 ? 1 : 2);
            }
            int stride = doc.length + 16;
            ByteBuffer region = ByteBuffer.allocate(2 * stride).order(ByteOrder.LITTLE_ENDIAN);
            region.put(other).putFloat(.5f).putFloat(.875f).putFloat(-.25f).putInt(11);
            region.put(doc).putFloat(-.25f).putFloat(.5f).putFloat(.375f).putInt(23);
            int[] ordinals = {1, 0, 1};
            p = BlockOperations.parameters(ordinals.length);
            p[1] = d;
            p[2] = stride;
            p[3] = metric.ordinal();
            p[4] = encoding;
            p[5] =
                Integer.toUnsignedLong(Float.floatToRawIntBits(-.5f))
                    | (long) Float.floatToRawIntBits(.75f) << 32;
            p[6] = Integer.toUnsignedLong(Float.floatToRawIntBits(.125f)) | (long) qsum << 32;
            p[7] = Integer.toUnsignedLong(Float.floatToRawIntBits(.25f));
            result.add(
                oracle(
                    new BlockOperations.Operation(
                        cap,
                        ordinals.length,
                        p,
                        List.of(
                            BlockOperations.bytes(q, 0, q.length),
                            BlockOperations.bytes(region.array(), 0, region.capacity()),
                            BlockOperations.ints(ordinals, 0, ordinals.length)),
                        ordinals.length,
                        4)));
          }
      }
      case BM25_BLOCK -> {
        // A full SIMD body with docs at the norms end (gather shift path), then a scalar case.
        float[] cache = new float[256];
        for (int i = 0; i < cache.length; i++)
          cache[i] = 1f / (1.2f * ((1 - 0.75f) + 0.75f * SmallFloat.byte4ToInt((byte) i) / 7.5f));
        for (int length : new int[] {23, 3}) {
          int n = length == 3 ? 5 : 19;
          byte[] norms = new byte[length];
          for (int i = 0; i < length; i++) norms[i] = (byte) (i * 37 + 11);
          ByteBuffer block = ByteBuffer.allocate(n * 8).order(ByteOrder.LITTLE_ENDIAN);
          for (int i = 0; i < n; i++) block.putInt(i * 4, (i * 7 + length - 1) % length);
          for (int i = 0; i < n; i++) block.putFloat((n + i) * 4, 1 + i % 5);
          p = BlockOperations.parameters(n);
          p[1] = 1;
          p[2] = length;
          p[3] = Integer.toUnsignedLong(Float.floatToRawIntBits(2.375f));
          result.add(
              oracle(
                  new BlockOperations.Operation(
                      cap,
                      n,
                      p,
                      List.of(
                          BlockOperations.bytes(block.array(), 0, n * 8),
                          BlockOperations.bytes(norms, 0, length),
                          BlockOperations.floats(cache)),
                      n,
                      4)));
        }
      }
      case MODERN_QUANTIZED_RESCORE -> {
        for (int encoding = 0; encoding <= 4; encoding++)
          for (var metric : VectorSimilarityFunction.values()) {
            int d = 16, max = encoding == 0 ? 255 : encoding == 2 ? 127 : 15;
            byte[] q = new byte[d], doc = new byte[d];
            for (int i = 0; i < d; i++) {
              q[i] = (byte) ((i * 7) % max);
              doc[i] = (byte) ((i * 3) % (encoding >= 3 ? encoding == 3 ? 2 : 4 : max));
            }
            int qsum = 0, dsum = 0;
            for (byte v : q) qsum += v & 255;
            for (byte v : doc) dsum += v & 255;
            if (encoding == 1) doc = BlockOperations.packNibbles(null, doc);
            else if (encoding >= 3) {
              q = BlockOperations.transposeBits(null, q, 4);
              doc = BlockOperations.transposeBits(null, doc, encoding == 3 ? 1 : 2);
            }
            ByteBuffer c = ByteBuffer.allocate(48).order(ByteOrder.LITTLE_ENDIAN);
            c.putFloat(-.5f)
                .putFloat(.75f)
                .putFloat(.125f)
                .putInt(qsum)
                .putFloat(.25f)
                .putInt(0)
                .putLong(0);
            c.putFloat(-.25f).putFloat(.5f).putFloat(.375f).putInt(dsum);
            p = BlockOperations.parameters(1);
            p[1] = d;
            p[2] = doc.length;
            p[3] = metric.ordinal();
            p[4] = encoding;
            result.add(
                oracle(
                    new BlockOperations.Operation(
                        cap,
                        1,
                        p,
                        List.of(
                            BlockOperations.bytes(q, 0, q.length),
                            BlockOperations.bytes(doc, 0, doc.length),
                            BlockOperations.bytes(c.array(), 0, c.capacity())),
                        1,
                        4)));
          }
      }
      case OPTIMIZED_QUANTIZE -> {
        for (byte bits : new byte[] {1, 2, 4, 7, 8})
          for (var metric : VectorSimilarityFunction.values())
            for (int d : new int[] {3, 17, 128}) {
              float[] v = new float[d], c = new float[d];
              Random random = new Random(0x104 + d);
              for (int i = 0; i < d; i++) {
                v[i] = (random.nextFloat() - .5f) * 2;
                c[i] = (random.nextFloat() - .5f) * .5f;
              }
              if (metric == VectorSimilarityFunction.COSINE) {
                VectorUtil.l2normalize(v);
                VectorUtil.l2normalize(c);
              }
              p = BlockOperations.parameters(d);
              p[1] = bits;
              p[2] = metric.ordinal();
              p[3] = Integer.toUnsignedLong(Float.floatToRawIntBits(.1f));
              p[4] = 5;
              result.add(
                  oracle(
                      new BlockOperations.Operation(
                          cap,
                          d,
                          p,
                          List.of(BlockOperations.floats(v), BlockOperations.floats(c)),
                          d * 5 + 16,
                          1,
                          d * 4L)));
            }
      }
      case VECTOR_STATISTICS -> {
        float[][] v = {{1, 2, 3}, {-2, 5, 9}, {.1f, -.2f, .3f}};
        p = BlockOperations.parameters(v.length);
        p[1] = 3;
        p[2] = 3;
        result.add(
            oracle(
                new BlockOperations.Operation(
                    cap, v.length, p, List.of(BlockOperations.matrix(v, 3)), 3, 4)));
      }
      case POSTINGS_PREFIX -> {
        int[] v = {Integer.MAX_VALUE, 1, -1, 17};
        p = BlockOperations.parameters(v.length);
        p[2] = 0xffffffffL;
        result.add(
            oracle(
                new BlockOperations.Operation(
                    cap, v.length, p, List.of(BlockOperations.ints(v, 0, v.length)), v.length, 4)));
      }
      case POSTINGS_BLOCK -> {
        for (int bits = 1; bits <= 32; bits++) {
          byte[] data = new byte[bits * 32];
          new Random(0x104 + bits).nextBytes(data);
          p = BlockOperations.parameters(256);
          p[1] = bits;
          result.add(
              oracle(
                  new BlockOperations.Operation(
                      cap, 256, p, List.of(BlockOperations.bytes(data, 0, data.length)), 256, 4)));
        }
      }
      case FLOATING_STATS -> {
        for (boolean nonfinite : new boolean[] {false, true}) {
          double[] v =
              nonfinite
                  ? new double[] {-0.0, 0.0, Double.POSITIVE_INFINITY, Double.NaN}
                  : new double[] {1, 2, 3, -7, .125};
          ByteBuffer values = ByteBuffer.allocate(v.length * 8).order(ByteOrder.LITTLE_ENDIAN);
          for (double x : v) values.putDouble(x);
          ByteBuffer state = ByteBuffer.allocate(48).order(ByteOrder.LITTLE_ENDIAN);
          state
              .putLong(0)
              .putDouble(0)
              .putDouble(Double.MAX_VALUE)
              .putDouble(-Double.MAX_VALUE)
              .putDouble(0)
              .putDouble(0);
          p = BlockOperations.parameters(v.length);
          result.add(
              oracle(
                  new BlockOperations.Operation(
                      cap,
                      v.length,
                      p,
                      List.of(
                          BlockOperations.bytes(values.array(), 0, values.capacity()),
                          BlockOperations.bytes(state.array(), 0, state.capacity())),
                      48,
                      1)));
        }
      }
      case COMPRESSION -> {
        byte[] data = {0x30, 'a', 'b', 'c', 3, 0, 0x50, 'a', 'b', 'c', 'd', 'e'};
        p = BlockOperations.parameters(data.length);
        p[1] = 12;
        result.add(
            oracle(
                new BlockOperations.Operation(
                    cap,
                    data.length,
                    p,
                    List.of(BlockOperations.bytes(data, 0, data.length)),
                    12,
                    1,
                    12)));
      }
      case TEST_COPY,
          VECTOR_I8,
          DOCSET_INTERSECT,
          DOCSET_UNION,
          DOCSET_DIFFERENCE,
          DOCSET_CARDINALITY,
          DOCSET_INTERSECT_COUNT,
          DOCVALUES_DECODE,
          DOCVALUES_RANGE,
          DOCVALUES_RANGE_FILTER,
          POSTINGS_SPLIT,
          QUANTIZE_PACK,
          FACETS_COUNT,
          DOCVALUES_AGGREGATE,
          QUANTIZED_RESCORE,
          QUANTIZE_TRANSPOSE ->
          throw new AssertionError(cap);
    }
    return result;
  }

  private static Fixture longs(BlockOperations.Operation operation, long[] expected) {
    ByteBuffer encoded = ByteBuffer.allocate(expected.length * 8).order(ByteOrder.LITTLE_ENDIAN);
    for (long value : expected) encoded.putLong(value);
    return new Fixture(operation, encoded.array());
  }

  private static Fixture ints(BlockOperations.Operation operation, int[] expected) {
    ByteBuffer encoded = ByteBuffer.allocate(expected.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (int value : expected) encoded.putInt(value);
    return new Fixture(operation, encoded.array());
  }

  private static long[] aggregateWords(BlockOperations.Aggregate aggregate) {
    return new long[] {
      aggregate.count(),
      aggregate.sum(),
      aggregate.minimum(),
      aggregate.maximum(),
      aggregate.hasValue() ? 1 : 0,
      aggregate.overflow() ? 1 : 0
    };
  }
}
