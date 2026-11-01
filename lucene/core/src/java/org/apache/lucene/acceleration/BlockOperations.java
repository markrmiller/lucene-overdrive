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
import java.util.List;
import java.util.Objects;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.Constants;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.ScalarQuantizedVectorSimilarity;

/**
 * Bounded, storage-independent block operations with complete Java implementations. Native inputs
 * and unpublished output are copied directly into admitted staging. Caller-owned input and Java
 * result arrays remain on the existing heap path and are not enrolled in the native ledger.
 * Supplying a null runtime chooses Java without initializing the process registry.
 *
 * @lucene.experimental
 */
public final class BlockOperations {
  private BlockOperations() {}

  private static final long MAX_ELEMENTS = 16L * 1024 * 1024;
  private static final long MAX_BYTES = 128L * 1024 * 1024;

  @FunctionalInterface
  interface Writer {
    void write(ByteBuffer buffer);
  }

  record Source(long elements, AccelerationProvider.Encoding encoding, Writer writer) {
    long bytes() {
      return Math.multiplyExact(elements, encoding.width);
    }
  }

  record Operation(
      Capability capability,
      long items,
      long[] parameters,
      List<Source> inputs,
      long outputElements,
      int outputWidth,
      long scratchBytes) {
    Operation(
        Capability capability,
        long items,
        long[] parameters,
        List<Source> inputs,
        long outputElements,
        int outputWidth) {
      this(capability, items, parameters, inputs, outputElements, outputWidth, 0);
    }

    long outputBytes() {
      return Math.multiplyExact(outputElements, outputWidth);
    }

    long stagedBytes() {
      long bytes = Math.addExact(Math.addExact(outputBytes(), scratchBytes), 320);
      for (Source input : inputs) bytes = Math.addExact(bytes, input.bytes());
      return bytes;
    }

    AccelerationProvider.Batch stage(RequestMemoryScope scope) {
      List<AccelerationProvider.Input> views = new ArrayList<>(inputs.size());
      for (Source input : inputs) {
        ByteBuffer storage = scope.allocate(input.bytes()).order(ByteOrder.LITTLE_ENDIAN);
        input.writer().write(storage);
        storage.clear();
        views.add(new AccelerationProvider.Input(storage, input.elements(), input.encoding()));
      }
      ByteBuffer output = scope.allocate(outputBytes()).order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer params = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < 8; i++) params.putLong(i * 8, parameters[i]);
      return new AccelerationProvider.Batch(
          capability,
          views,
          new AccelerationProvider.Output(output, outputElements),
          params,
          scope.allocate(scratchBytes),
          scope.allocate(256));
    }
  }

  static boolean nativeCall(AccelerationRuntime runtime, Operation operation, Writer publish)
      throws IOException {
    if (runtime == null) return false;
    runtime.checkpoint();
    if (!runtime.eligible(operation.capability(), operation.items())) return false;
    if (!runtime.supports(operation.capability(), "generic")) {
      runtime.recordJava(operation.capability(), AccelerationRuntime.Reason.NO_MEASURED_WIN);
      return false;
    }
    boolean unsupported =
        operation.items() > MAX_ELEMENTS
            || operation.outputElements() > MAX_ELEMENTS
            || operation.stagedBytes() > MAX_BYTES;
    if (operation.capability() == Capability.VECTOR_I8
        || operation.capability() == Capability.QUANTIZED_RESCORE) {
      unsupported |=
          operation.parameters()[1] == 0
              || operation.parameters()[1]
                  > (operation.capability() == Capability.VECTOR_I8 ? 32768 : 16384)
              || operation.inputs().get(1).elements() > MAX_ELEMENTS;
    }
    if (unsupported) {
      runtime.recordJava(operation.capability(), AccelerationRuntime.Reason.UNSUPPORTED);
      return false;
    }
    try (RequestMemoryScope scope = runtime.newCallScope()) {
      runtime.checkpoint();
      AccelerationProvider.Batch batch = operation.stage(scope);
      runtime.checkpoint();
      if (!runtime.execute(batch, operation.items(), operation.stagedBytes(), scope)) return false;
      runtime.checkpoint();
      publish.write(batch.output().bytes().duplicate().clear().order(ByteOrder.LITTLE_ENDIAN));
      return true;
    } catch (RequestMemoryScope.AdmissionException denied) {
      if (!denied.isCapacityFailure()) {
        runtime.checkpoint();
        throw new IOException("native staging unavailable: " + denied.reason(), denied);
      }
      runtime.checkpoint();
      runtime.recordJava(operation.capability(), AccelerationRuntime.Reason.BUDGET_DENIED);
      return false;
    }
  }

  static Source bytes(byte[] values, int offset, int length) {
    Objects.checkFromIndexSize(offset, length, values.length);
    return new Source(
        length, AccelerationProvider.Encoding.BYTES, b -> b.put(values, offset, length));
  }

  static Source longs(long[] values, int offset, int length) {
    Objects.checkFromIndexSize(offset, length, values.length);
    return new Source(
        length,
        AccelerationProvider.Encoding.I64,
        b -> {
          for (int i = 0; i < length; i++) b.putLong(i * 8, values[offset + i]);
        });
  }

  static Source ints(int[] values, int offset, int length) {
    Objects.checkFromIndexSize(offset, length, values.length);
    return new Source(
        length,
        AccelerationProvider.Encoding.I32,
        b -> {
          for (int i = 0; i < length; i++) b.putInt(i * 4, values[offset + i]);
        });
  }

  static Source floats(float[] values) {
    return new Source(
        values.length,
        AccelerationProvider.Encoding.F32,
        b -> {
          for (int i = 0; i < values.length; i++) b.putFloat(i * 4, values[i]);
        });
  }

  static long[] parameters(long items) {
    long[] params = new long[8];
    params[0] = items;
    return params;
  }

  private static void readLongs(ByteBuffer buffer, long[] output, int offset, int count) {
    for (int i = 0; i < count; i++) output[offset + i] = buffer.getLong(i * 8);
  }

  private static void readInts(ByteBuffer buffer, int[] output, int offset, int count) {
    for (int i = 0; i < count; i++) output[offset + i] = buffer.getInt(i * 4);
  }

  private static void readFloats(ByteBuffer buffer, float[] output) {
    for (int i = 0; i < output.length; i++) output[i] = buffer.getFloat(i * 4);
  }

  /** Copies words through the registered test kernel, with an exact Java alternative. */
  public static long[] copy(AccelerationRuntime runtime, long[] input) throws IOException {
    long[] output = new long[input.length];
    Operation operation =
        new Operation(
            Capability.TEST_COPY,
            input.length,
            parameters(input.length),
            List.of(longs(input, 0, input.length)),
            input.length,
            8);
    if (!nativeCall(runtime, operation, b -> readLongs(b, output, 0, output.length)))
      System.arraycopy(input, 0, output, 0, input.length);
    return output;
  }

  /** Byte-aligned little-endian values, decoded with Java long wrapping multiply/add semantics. */
  public static long[] decode(
      AccelerationRuntime runtime,
      byte[] input,
      int offset,
      int count,
      int bits,
      int stride,
      long multiplier,
      long base)
      throws IOException {
    decodeSpan(input, offset, count, bits, stride);
    long[] output = new long[count];
    decodeInto(runtime, input, offset, count, bits, stride, multiplier, base, output, 0);
    return output;
  }

  public static long[] decode(
      byte[] input, int offset, int count, int bits, int stride, long multiplier, long base)
      throws IOException {
    return decode(
        AccelerationRuntime.system(), input, offset, count, bits, stride, multiplier, base);
  }

  /**
   * Decodes into an existing result buffer, publishing native staging only after complete success.
   */
  public static void decodeInto(
      AccelerationRuntime runtime,
      byte[] input,
      int offset,
      int count,
      int bits,
      int stride,
      long multiplier,
      long base,
      long[] output,
      int outputOffset)
      throws IOException {
    int span = decodeSpan(input, offset, count, bits, stride);
    int actualStride = stride == 0 ? bits / 8 : stride;
    Objects.checkFromIndexSize(outputOffset, count, output.length);
    long[] params = parameters(count);
    params[1] = bits;
    params[2] = actualStride;
    params[5] = multiplier;
    params[6] = base;
    Operation operation =
        new Operation(
            Capability.DOCVALUES_DECODE,
            count,
            params,
            List.of(bytes(input, offset, span)),
            count,
            8);
    if (nativeCall(runtime, operation, b -> readLongs(b, output, outputOffset, count))) return;
    for (int i = 0; i < count; i++)
      output[outputOffset + i] =
          decoded(input, offset + i * actualStride, bits / 8, multiplier, base);
  }

  private static int decodeSpan(byte[] input, int offset, int count, int bits, int stride) {
    if (bits < 8 || bits > 64 || (bits & 7) != 0 || count < 0)
      throw new IllegalArgumentException("expected byte-aligned width in [8,64]");
    int actualStride = stride == 0 ? bits / 8 : stride;
    if (actualStride < bits / 8) throw new IllegalArgumentException("stride shorter than payload");
    int span = Math.toIntExact(Math.multiplyExact((long) count, actualStride));
    Objects.checkFromIndexSize(offset, span, input.length);
    return span;
  }

  private static long decoded(byte[] input, int offset, int width, long multiplier, long base) {
    long raw = 0;
    for (int j = 0; j < width; j++) raw |= (input[offset + j] & 255L) << (j * 8);
    return raw * multiplier + base;
  }

  /** Inclusive signed-long range matches as dense words, with zero unused tail bits. */
  public static long[] range(AccelerationRuntime runtime, long[] values, long minimum, long maximum)
      throws IOException {
    long[] output = new long[(int) (((long) values.length + 63) >>> 6)];
    long[] params = parameters(values.length);
    params[3] = minimum;
    params[4] = maximum;
    Operation operation =
        new Operation(
            Capability.DOCVALUES_RANGE,
            values.length,
            params,
            List.of(longs(values, 0, values.length)),
            output.length,
            8);
    if (!nativeCall(runtime, operation, b -> readLongs(b, output, 0, output.length))) {
      for (int i = 0; i < values.length; i++)
        if (values[i] >= minimum && values[i] <= maximum) output[i >>> 6] |= 1L << i;
    }
    return output;
  }

  /**
   * Fused {@code selection & range(values)}: bit {@code i} is set when selection bit {@code i} is
   * set and {@code minimum <= values[i] <= maximum}. Values under zero selection words are not
   * read; selection bits at or beyond {@code values.length} are cleared.
   */
  public static long[] rangeFilter(
      AccelerationRuntime runtime, long[] values, long[] selection, long minimum, long maximum)
      throws IOException {
    int words = (int) (((long) values.length + 63) >>> 6);
    checkWords(selection, words);
    long[] output = new long[words];
    long[] params = parameters(values.length);
    params[3] = minimum;
    params[4] = maximum;
    Operation operation =
        new Operation(
            Capability.DOCVALUES_RANGE_FILTER,
            values.length,
            params,
            List.of(longs(values, 0, values.length), longs(selection, 0, words)),
            words,
            8);
    if (!nativeCall(runtime, operation, b -> readLongs(b, output, 0, output.length))) {
      for (int w = 0; w < words; w++) {
        long selected = selection[w];
        if (w == words - 1 && (values.length & 63) != 0) selected &= (1L << values.length) - 1;
        long bits = 0;
        for (; selected != 0; selected &= selected - 1) {
          int bit = Long.numberOfTrailingZeros(selected), i = (w << 6) + bit;
          if (values[i] >= minimum && values[i] <= maximum) bits |= 1L << bit;
        }
        output[w] = bits;
      }
    }
    return output;
  }

  public static long[] rangeFilter(long[] values, long[] selection, long minimum, long maximum)
      throws IOException {
    return rangeFilter(AccelerationRuntime.system(), values, selection, minimum, maximum);
  }

  public static long[] range(long[] values, long minimum, long maximum) throws IOException {
    return range(AccelerationRuntime.system(), values, minimum, maximum);
  }

  /**
   * Word-wise intersection, union, or difference; inputs are immutable and tail bits are masked.
   */
  public static long[] bitset(
      AccelerationRuntime runtime, Capability operation, long[] left, long[] right, int bitLength)
      throws IOException {
    if (operation != Capability.DOCSET_INTERSECT
        && operation != Capability.DOCSET_UNION
        && operation != Capability.DOCSET_DIFFERENCE)
      throw new IllegalArgumentException("not bit algebra");
    int words = wordCount(bitLength);
    checkWords(left, words);
    checkWords(right, words);
    long[] output = new long[words];
    long[] params = parameters(words);
    params[2] = bitLength;
    Operation call =
        new Operation(
            operation,
            words,
            params,
            List.of(longs(left, 0, words), longs(right, 0, words)),
            words,
            8);
    if (!nativeCall(runtime, call, b -> readLongs(b, output, 0, output.length))) {
      for (int i = 0; i < words; i++)
        output[i] =
            operation == Capability.DOCSET_INTERSECT
                ? left[i] & right[i]
                : operation == Capability.DOCSET_UNION ? left[i] | right[i] : left[i] & ~right[i];
      maskTail(output, bitLength);
    }
    return output;
  }

  public static long[] bitset(Capability operation, long[] left, long[] right, int bitLength)
      throws IOException {
    return bitset(AccelerationRuntime.system(), operation, left, right, bitLength);
  }

  public static long cardinality(AccelerationRuntime runtime, long[] input, int bitLength)
      throws IOException {
    int words = wordCount(bitLength);
    checkWords(input, words);
    long[] params = parameters(words);
    params[2] = bitLength;
    long[] output = new long[1];
    Operation call =
        new Operation(
            Capability.DOCSET_CARDINALITY, words, params, List.of(longs(input, 0, words)), 1, 8);
    if (nativeCall(runtime, call, b -> output[0] = b.getLong(0))) return output[0];
    long result = 0;
    for (int i = 0; i < words; i++)
      result += Long.bitCount(input[i] & (i == words - 1 ? tailMask(bitLength) : -1L));
    return result;
  }

  /**
   * Population count of {@code left & right} over {@code bitLength} bits, without materializing the
   * intersection.
   */
  public static long intersectionCount(
      AccelerationRuntime runtime, long[] left, long[] right, int bitLength) throws IOException {
    int words = wordCount(bitLength);
    checkWords(left, words);
    checkWords(right, words);
    long[] params = parameters(words);
    params[2] = bitLength;
    long[] output = new long[1];
    Operation call =
        new Operation(
            Capability.DOCSET_INTERSECT_COUNT,
            words,
            params,
            List.of(longs(left, 0, words), longs(right, 0, words)),
            1,
            8);
    if (nativeCall(runtime, call, b -> output[0] = b.getLong(0))) return output[0];
    long result = 0;
    for (int i = 0; i < words; i++)
      result += Long.bitCount(left[i] & right[i] & (i == words - 1 ? tailMask(bitLength) : -1L));
    return result;
  }

  public static long intersectionCount(long[] left, long[] right, int bitLength)
      throws IOException {
    return intersectionCount(AccelerationRuntime.system(), left, right, bitLength);
  }

  private static int wordCount(int bitLength) {
    if (bitLength < 0) throw new IllegalArgumentException("negative bit length");
    return (int) (((long) bitLength + 63) >>> 6);
  }

  private static void checkWords(long[] words, int count) {
    if (words.length < count) throw new IllegalArgumentException("insufficient bit words");
  }

  private static long tailMask(int length) {
    return (length & 63) == 0 ? -1L : (1L << (length & 63)) - 1;
  }

  private static void maskTail(long[] words, int length) {
    if (words.length != 0) words[words.length - 1] &= tailMask(length);
  }

  /** Counts local ordinals; -1 is missing, all other ordinals must be in [0,buckets). */
  public static int[] histogram(AccelerationRuntime runtime, int[] ordinals, int buckets)
      throws IOException {
    if (buckets < 0) throw new IllegalArgumentException("negative bucket count");
    for (int ordinal : ordinals)
      if (ordinal < -1 || ordinal >= buckets) throw new IllegalArgumentException("invalid ordinal");
    int[] output = new int[buckets];
    long[] params = parameters(ordinals.length);
    params[1] = buckets;
    Operation call =
        new Operation(
            Capability.FACETS_COUNT,
            ordinals.length,
            params,
            List.of(ints(ordinals, 0, ordinals.length)),
            buckets,
            4);
    if (!nativeCall(runtime, call, b -> readInts(b, output, 0, output.length)))
      for (int ordinal : ordinals) if (ordinal != -1) output[ordinal]++;
    return output;
  }

  public static int[] histogram(int[] ordinals, int buckets) throws IOException {
    return histogram(AccelerationRuntime.system(), ordinals, buckets);
  }

  /** Aggregate sum wraps as Java long; overflow is sticky over every accepted addition. */
  public record Aggregate(
      long count, long sum, long minimum, long maximum, boolean hasValue, boolean overflow) {}

  public static Aggregate aggregate(
      AccelerationRuntime runtime,
      byte[] input,
      int offset,
      int count,
      int bits,
      int stride,
      long multiplier,
      long base,
      long[] accepted,
      long minimum,
      long maximum)
      throws IOException {
    int span = decodeSpan(input, offset, count, bits, stride);
    int actualStride = stride == 0 ? bits / 8 : stride;
    int words = wordCount(count);
    if (accepted != null) checkWords(accepted, words);
    long[] params = parameters(count);
    params[1] = bits;
    params[2] = actualStride;
    params[3] = minimum;
    params[4] = maximum;
    params[5] = multiplier;
    params[6] = base;
    List<Source> sources =
        accepted == null
            ? List.of(bytes(input, offset, span))
            : List.of(bytes(input, offset, span), longs(accepted, 0, words));
    long[] output = new long[6];
    Operation call = new Operation(Capability.DOCVALUES_AGGREGATE, count, params, sources, 6, 8);
    if (nativeCall(runtime, call, b -> readLongs(b, output, 0, 6))) {
      return new Aggregate(
          output[0], output[1], output[2], output[3], output[4] != 0, output[5] != 0);
    }
    long seen = 0, sum = 0, min = 0, max = 0;
    boolean overflow = false;
    for (int i = 0; i < count; i++) {
      if (accepted != null && (accepted[i >>> 6] & (1L << i)) == 0) continue;
      long value = decoded(input, offset + i * actualStride, bits / 8, multiplier, base);
      if (value < minimum || value > maximum) continue;
      if (seen == 0 || value < min) min = value;
      if (seen == 0 || value > max) max = value;
      long next = sum + value;
      overflow |= ((sum ^ next) & (value ^ next)) < 0;
      sum = next;
      seen++;
    }
    return new Aggregate(seen, sum, min, max, seen != 0, overflow);
  }

  /** Exact Lucene split layout: high groups by column, followed by the low remainder column. */
  public static int[] split(
      AccelerationRuntime runtime,
      int[] input,
      int bits,
      int decrement,
      int mask,
      int remainderMask)
      throws IOException {
    int blocks = splitBlocks(bits, decrement);
    int[] output = new int[Math.multiplyExact(input.length, blocks + 1)];
    splitInto(runtime, input, 0, input.length, bits, decrement, mask, remainderMask, output, 0);
    return output;
  }

  public static int[] split(int[] input, int bits, int decrement, int mask, int remainderMask)
      throws IOException {
    return split(AccelerationRuntime.system(), input, bits, decrement, mask, remainderMask);
  }

  public static void splitInto(
      AccelerationRuntime runtime,
      int[] input,
      int inputOffset,
      int count,
      int bits,
      int decrement,
      int mask,
      int remainderMask,
      int[] output,
      int outputOffset)
      throws IOException {
    int blocks = splitBlocks(bits, decrement);
    int outputCount = Math.multiplyExact(count, blocks + 1);
    Objects.checkFromIndexSize(inputOffset, count, input.length);
    Objects.checkFromIndexSize(outputOffset, outputCount, output.length);
    if (input == output) throw new IllegalArgumentException("split output must not alias input");
    long[] params = parameters(count);
    params[1] = bits;
    params[2] = decrement;
    params[3] = Integer.toUnsignedLong(mask);
    params[4] = Integer.toUnsignedLong(remainderMask);
    Operation call =
        new Operation(
            Capability.POSTINGS_SPLIT,
            count,
            params,
            List.of(ints(input, inputOffset, count)),
            outputCount,
            4);
    if (nativeCall(runtime, call, b -> readInts(b, output, outputOffset, outputCount))) return;
    for (int block = 0; block < blocks; block++)
      for (int i = 0; i < count; i++)
        output[outputOffset + block * count + i] =
            (input[inputOffset + i] >>> (bits - block * decrement)) & mask;
    for (int i = 0; i < count; i++)
      output[outputOffset + blocks * count + i] = input[inputOffset + i] & remainderMask;
  }

  private static int splitBlocks(int bits, int decrement) {
    if (bits < 1 || bits > 31 || decrement < 1 || decrement > 31)
      throw new IllegalArgumentException("invalid split widths");
    return (bits - 1) / decrement + 1;
  }

  /** Packs first-half nibbles high and second-half nibbles low, preserving Lucene's layout. */
  public static byte[] packNibbles(AccelerationRuntime runtime, byte[] input) throws IOException {
    checkNibbles(input);
    if ((input.length & 1) != 0)
      throw new IllegalArgumentException("nibble packing needs even length");
    byte[] output = new byte[input.length / 2];
    Operation call =
        new Operation(
            Capability.QUANTIZE_PACK,
            input.length,
            parameters(input.length),
            List.of(bytes(input, 0, input.length)),
            output.length,
            1);
    if (!nativeCall(runtime, call, b -> b.get(output)))
      for (int i = 0; i < output.length; i++)
        output[i] = (byte) (input[i] << 4 | input[output.length + i]);
    return output;
  }

  /** Four bit planes, high bit first within each byte; trailing bits are zero. */
  public static byte[] transposeNibbles(AccelerationRuntime runtime, byte[] input)
      throws IOException {
    checkNibbles(input);
    int blocks = Math.toIntExact(((long) input.length + 7) >>> 3);
    byte[] output = new byte[Math.multiplyExact(blocks, 4)];
    Operation call =
        new Operation(
            Capability.QUANTIZE_TRANSPOSE,
            input.length,
            parameters(input.length),
            List.of(bytes(input, 0, input.length)),
            output.length,
            1);
    if (!nativeCall(runtime, call, b -> b.get(output))) {
      for (int plane = 0; plane < 4; plane++)
        for (int i = 0; i < input.length; i++)
          output[plane * blocks + (i >>> 3)] |=
              (byte) (((input[i] >>> plane) & 1) << (7 - (i & 7)));
    }
    return output;
  }

  private static void checkNibbles(byte[] input) {
    for (byte value : input)
      if ((value & 255) > 15) throw new IllegalArgumentException("invalid nibble");
  }

  /** Signed byte vector scoring; traversal order and row order stay unchanged. */
  public static float[] byteScores(
      AccelerationRuntime runtime,
      byte[] query,
      byte[][] documents,
      VectorSimilarityFunction similarity)
      throws IOException {
    return vectorScores(runtime, query, documents, similarity, null, 0, 0);
  }

  /** Exact unsigned seven-bit scalar-quantized scoring using Lucene's correction order. */
  public static float[] quantizedRescore(
      AccelerationRuntime runtime,
      byte[] query,
      byte[][] documents,
      float[] corrections,
      VectorSimilarityFunction similarity,
      float multiplier,
      float queryCorrection)
      throws IOException {
    Objects.requireNonNull(corrections);
    if (corrections.length != documents.length)
      throw new IllegalArgumentException("correction count");
    for (byte value : query)
      if (value < 0) throw new IllegalArgumentException("not seven-bit query");
    for (byte[] document : documents)
      for (byte value : document)
        if (value < 0) throw new IllegalArgumentException("not seven-bit vector");
    return vectorScores(
        runtime, query, documents, similarity, corrections, multiplier, queryCorrection);
  }

  private static float[] vectorScores(
      AccelerationRuntime runtime,
      byte[] query,
      byte[][] documents,
      VectorSimilarityFunction similarity,
      float[] corrections,
      float multiplier,
      float queryCorrection)
      throws IOException {
    Objects.requireNonNull(similarity);
    for (byte[] document : documents)
      if (document.length != query.length)
        throw new IllegalArgumentException("vector dimension mismatch");
    long rowsBytes = Math.multiplyExact((long) documents.length, query.length);
    Source rows =
        new Source(
            rowsBytes,
            AccelerationProvider.Encoding.BYTES,
            b -> {
              for (byte[] document : documents) b.put(document);
            });
    Capability capability =
        corrections == null ? Capability.VECTOR_I8 : Capability.QUANTIZED_RESCORE;
    long[] params = parameters(documents.length);
    params[1] = query.length;
    params[2] = query.length;
    params[3] = similarity.ordinal();
    params[5] = Integer.toUnsignedLong(Float.floatToRawIntBits(multiplier));
    params[6] = Integer.toUnsignedLong(Float.floatToRawIntBits(queryCorrection));
    List<Source> sources =
        corrections == null
            ? List.of(bytes(query, 0, query.length), rows)
            : List.of(bytes(query, 0, query.length), rows, floats(corrections));
    float[] output = new float[documents.length];
    Operation call = new Operation(capability, documents.length, params, sources, output.length, 4);
    if (nativeCall(runtime, call, b -> readFloats(b, output))) return output;
    ScalarQuantizedVectorSimilarity quantized =
        corrections == null
            ? null
            : ScalarQuantizedVectorSimilarity.fromVectorSimilarity(
                similarity, multiplier, (byte) 7);
    for (int i = 0; i < output.length; i++)
      output[i] =
          quantized == null
              ? similarity.compare(query, documents[i])
              : quantized.score(query, queryCorrection, documents[i], corrections[i]);
    return output;
  }

  /** Scores dense float rows after exact qualification of the active Java arithmetic profile. */
  public static float[] floatScores(
      AccelerationRuntime runtime,
      float[] query,
      float[][] documents,
      VectorSimilarityFunction similarity)
      throws IOException {
    for (float[] row : documents)
      if (row.length != query.length)
        throw new IllegalArgumentException("vector dimension mismatch");
    long[] p = parameters(documents.length);
    p[1] = query.length;
    p[2] = query.length;
    p[3] = similarity.ordinal();
    p[4] = Constants.HAS_FAST_SCALAR_FMA ? 1 : 0;
    float[] result = new float[documents.length];
    Operation call =
        new Operation(
            Capability.VECTOR_F32,
            documents.length,
            p,
            List.of(floats(query), matrix(documents, query.length)),
            result.length,
            4);
    if (!nativeCall(runtime, call, b -> readFloats(b, result)))
      for (int i = 0; i < result.length; i++) result[i] = similarity.compare(query, documents[i]);
    return result;
  }

  static Source matrix(float[][] rows, int dim) {
    for (float[] row : rows)
      if (row.length != dim) throw new IllegalArgumentException("matrix dimension mismatch");
    return new Source(
        Math.multiplyExact((long) rows.length, dim),
        AccelerationProvider.Encoding.F32,
        b -> {
          for (float[] row : rows) for (float v : row) b.putFloat(v);
        });
  }

  /** MaxSim keeps the existing positive-minimum initializer, including an empty document. */
  public static float maxSim(
      float[][] query, float[][] document, VectorSimilarityFunction similarity) {
    if (document.length == 0) return Float.MIN_VALUE;
    float[] maxima = new float[query.length];
    java.util.Arrays.fill(maxima, Float.MIN_VALUE);
    for (float[] documentToken : document) {
      for (int q = 0; q < query.length; q++) {
        if (query[q].length != documentToken.length)
          throw new IllegalArgumentException("token dimension mismatch");
        maxima[q] = Float.max(maxima[q], similarity.compare(query[q], documentToken));
      }
    }
    float score = 0;
    for (float maximum : maxima) score += maximum;
    return score;
  }

  /** Decoded-matrix MaxSim with document offsets and a finite comparison-work allowance. */
  public static float[] maxSimBatch(
      AccelerationRuntime runtime,
      float[][] query,
      float[][][] documents,
      VectorSimilarityFunction similarity)
      throws IOException {
    int dim = query.length == 0 ? 0 : query[0].length;
    for (float[] q : query)
      if (q.length != dim) throw new IllegalArgumentException("query dimension mismatch");
    long[] offsets = new long[documents.length + 1];
    long tokens = 0;
    for (int r = 0; r < documents.length; r++) {
      for (float[] v : documents[r])
        if (query.length != 0 && v.length != dim)
          throw new IllegalArgumentException("token dimension mismatch");
      tokens = Math.addExact(tokens, documents[r].length);
      offsets[r + 1] = tokens;
    }
    long[] p = parameters(documents.length);
    p[1] = dim;
    p[2] = tokens;
    p[3] = similarity.ordinal();
    p[4] = query.length;
    p[5] = MAX_ELEMENTS;
    p[6] = Constants.HAS_FAST_SCALAR_FMA ? 1 : 0;
    Source rows =
        new Source(
            Math.multiplyExact(tokens, dim),
            AccelerationProvider.Encoding.F32,
            b -> {
              for (float[][] doc : documents)
                for (float[] row : doc) for (float v : row) b.putFloat(v);
            });
    float[] out = new float[documents.length];
    Operation call =
        new Operation(
            Capability.MAXSIM,
            documents.length,
            p,
            List.of(matrix(query, dim), rows, longs(offsets, 0, offsets.length)),
            out.length,
            4);
    if (query.length == 0 || !nativeCall(runtime, call, b -> readFloats(b, out)))
      for (int i = 0; i < out.length; i++) out[i] = maxSim(query, documents[i], similarity);
    return out;
  }

  /** Quantization publishes the centered vector and all four correction fields together. */
  public record Quantization(
      float[] centered,
      byte[] quantized,
      float lowerInterval,
      float upperInterval,
      float additionalCorrection,
      int quantizedComponentSum) {}

  public static Quantization quantize(
      AccelerationRuntime runtime,
      float[] vector,
      float[] centroid,
      byte bits,
      VectorSimilarityFunction similarity,
      float lambda,
      int iterations)
      throws IOException {
    if (vector.length != centroid.length
        || vector.length == 0
        || bits < 1
        || bits > 8
        || iterations < 0) throw new IllegalArgumentException("invalid quantization shape");
    float[] centered = vector.clone();
    byte[] quantized = new byte[vector.length];
    float[] corrections = new float[3];
    int[] sum = new int[1];
    long[] p = parameters(vector.length);
    p[1] = bits;
    p[2] = similarity.ordinal();
    p[3] = Integer.toUnsignedLong(Float.floatToRawIntBits(lambda));
    p[4] = iterations;
    Operation call =
        new Operation(
            Capability.OPTIMIZED_QUANTIZE,
            vector.length,
            p,
            List.of(floats(vector), floats(centroid)),
            vector.length * 5L + 16,
            1,
            vector.length * 4L);
    if (nativeCall(
        runtime,
        call,
        b -> {
          for (int i = 0; i < centered.length; i++) centered[i] = b.getFloat(i * 4);
          b.position(centered.length * 4);
          b.get(quantized);
          int at = centered.length * 5;
          for (int i = 0; i < 3; i++) corrections[i] = b.getFloat(at + i * 4);
          sum[0] = b.getInt(at + 12);
        }))
      return new Quantization(
          centered, quantized, corrections[0], corrections[1], corrections[2], sum[0]);
    var result =
        new OptimizedScalarQuantizer(similarity, lambda, iterations)
            .scalarQuantizeJava(centered, quantized, bits, centroid);
    return new Quantization(
        centered,
        quantized,
        result.lowerInterval(),
        result.upperInterval(),
        result.additionalCorrection(),
        result.quantizedComponentSum());
  }

  /** Packs one, two, or four unsigned bit planes in Lucene's MSB-first byte layout. */
  public static byte[] transposeBits(AccelerationRuntime runtime, byte[] input, int bits)
      throws IOException {
    if (bits == 4) return transposeNibbles(runtime, input);
    if (bits != 1 && bits != 2) throw new IllegalArgumentException("bits must be 1, 2, or 4");
    for (byte v : input)
      if ((v & 255) >= (1 << bits)) throw new IllegalArgumentException("value exceeds bit width");
    int blocks = (input.length + 7) / 8;
    byte[] out = new byte[Math.multiplyExact(blocks, bits)];
    Operation call =
        new Operation(
            bits == 1 ? Capability.QUANTIZE_BINARY : Capability.QUANTIZE_DIBIT,
            input.length,
            parameters(input.length),
            List.of(bytes(input, 0, input.length)),
            out.length,
            1);
    if (!nativeCall(runtime, call, b -> b.get(out)))
      for (int plane = 0; plane < bits; plane++)
        for (int i = 0; i < input.length; i++)
          out[plane * blocks + (i >>> 3)] |= (byte) (((input[i] >>> plane) & 1) << (7 - (i & 7)));
    return out;
  }

  public static byte[] unpackNibbles(AccelerationRuntime runtime, byte[] input) throws IOException {
    byte[] out = new byte[Math.multiplyExact(input.length, 2)];
    Operation call =
        new Operation(
            Capability.QUANTIZE_UNPACK,
            input.length,
            parameters(input.length),
            List.of(bytes(input, 0, input.length)),
            out.length,
            1);
    if (!nativeCall(runtime, call, b -> b.get(out)))
      for (int i = 0; i < input.length; i++) {
        out[i] = (byte) ((input[i] & 255) >>> 4);
        out[input.length + i] = (byte) (input[i] & 15);
      }
    return out;
  }

  /** Intersection of strictly increasing document identifiers, bounded by maxDoc. */
  public static int[] intersectSorted(
      AccelerationRuntime runtime, int[] left, int[] right, int maxDoc) throws IOException {
    for (int[] v : new int[][] {left, right})
      for (int i = 0; i < v.length; i++)
        if (v[i] < 0 || v[i] >= maxDoc || (i != 0 && v[i] <= v[i - 1]))
          throw new IllegalArgumentException("invalid sorted doc IDs");
    int[] result = new int[1 + Math.min(left.length, right.length)];
    long[] p = parameters(left.length);
    p[1] = right.length;
    p[2] = maxDoc;
    Operation call =
        new Operation(
            Capability.SORTED_INTERSECT,
            left.length,
            p,
            List.of(ints(left, 0, left.length), ints(right, 0, right.length)),
            result.length,
            4);
    if (!nativeCall(runtime, call, b -> readInts(b, result, 0, result.length))) {
      int l = 0, r = 0, count = 0;
      while (l < left.length && r < right.length) {
        if (left[l] == right[r]) {
          result[++count] = left[l];
          l++;
          r++;
        } else if (left[l] < right[r]) l++;
        else r++;
      }
      result[0] = count;
    }
    return ArrayUtil.copyOfSubArray(result, 1, 1 + result[0]);
  }

  public static long[] intersectMany(AccelerationRuntime runtime, long[][] sets, int bitLength)
      throws IOException {
    int words = wordCount(bitLength);
    if (sets.length == 0) throw new IllegalArgumentException("empty set collection");
    for (long[] set : sets) checkWords(set, words);
    long[] p = parameters(words);
    p[1] = sets.length;
    p[2] = bitLength;
    Source flat =
        new Source(
            Math.multiplyExact((long) words, sets.length),
            AccelerationProvider.Encoding.I64,
            b -> {
              for (long[] set : sets) for (int i = 0; i < words; i++) b.putLong(set[i]);
            });
    long[] out = new long[words];
    Operation call = new Operation(Capability.MULTI_INTERSECT, words, p, List.of(flat), words, 8);
    if (!nativeCall(runtime, call, b -> readLongs(b, out, 0, words))) {
      java.util.Arrays.fill(out, -1L);
      for (long[] set : sets) for (int i = 0; i < words; i++) out[i] &= set[i];
      maskTail(out, bitLength);
    }
    return out;
  }

  /** Ordered local ordinal/count pairs; missing ordinals are skipped. */
  public static int[] sparseHistogram(AccelerationRuntime runtime, int[] sorted, int buckets)
      throws IOException {
    int last = -1;
    for (int v : sorted) {
      if (v == -1) continue;
      if (v < 0 || v >= buckets || v < last) throw new IllegalArgumentException("unsorted ordinal");
      last = v;
    }
    int[] out = new int[Math.addExact(1, Math.multiplyExact(sorted.length, 2))];
    long[] p = parameters(sorted.length);
    p[1] = buckets;
    Operation call =
        new Operation(
            Capability.SPARSE_FACETS,
            sorted.length,
            p,
            List.of(ints(sorted, 0, sorted.length)),
            out.length,
            4);
    if (!nativeCall(runtime, call, b -> readInts(b, out, 0, out.length))) {
      int pairs = 0;
      last = -1;
      for (int v : sorted) {
        if (v == -1) continue;
        if (last != v) {
          out[1 + 2 * pairs] = v;
          pairs++;
          last = v;
        }
        out[2 * pairs]++;
      }
      out[0] = pairs;
    }
    return ArrayUtil.copyOfSubArray(out, 1, 1 + out[0] * 2);
  }

  /** Ordered float centroid accumulation without normalization. */
  public static float[] centroid(AccelerationRuntime runtime, float[][] vectors, int dimensions)
      throws IOException {
    long[] p = parameters(vectors.length);
    p[1] = dimensions;
    p[2] = dimensions;
    float[] out = new float[dimensions];
    Operation call =
        new Operation(
            Capability.VECTOR_STATISTICS,
            vectors.length,
            p,
            List.of(matrix(vectors, dimensions)),
            dimensions,
            4);
    if (!nativeCall(runtime, call, b -> readFloats(b, out))) {
      for (float[] row : vectors) for (int i = 0; i < dimensions; i++) out[i] += row[i];
      if (vectors.length != 0) for (int i = 0; i < dimensions; i++) out[i] /= vectors.length;
    }
    return out;
  }

  public static int[] prefixSum(AccelerationRuntime runtime, int[] values, int base)
      throws IOException {
    long[] p = parameters(values.length);
    p[2] = Integer.toUnsignedLong(base);
    int[] out = new int[values.length];
    Operation call =
        new Operation(
            Capability.POSTINGS_PREFIX,
            values.length,
            p,
            List.of(ints(values, 0, values.length)),
            out.length,
            4);
    if (!nativeCall(runtime, call, b -> readInts(b, out, 0, out.length))) {
      int sum = base;
      for (int i = 0; i < values.length; i++) out[i] = sum += values[i];
    }
    return out;
  }

  /** Ordered numeric statistics, with variance accumulated as the unnormalized second moment. */
  public record FloatingStats(
      long count, double sum, double min, double max, double mean, double variance) {}

  public static FloatingStats floatingStats(
      AccelerationRuntime runtime, double[] values, long[] accepted, FloatingStats initial)
      throws IOException {
    return floatingStats(runtime, values, values.length, accepted, initial);
  }

  /** Processes only the populated prefix of a reusable statistics buffer. */
  public static FloatingStats floatingStats(
      AccelerationRuntime runtime,
      double[] values,
      int valueCount,
      long[] accepted,
      FloatingStats initial)
      throws IOException {
    Objects.checkFromIndexSize(0, valueCount, values.length);
    if (initial.count() < 0) throw new IllegalArgumentException("negative count");
    if (accepted != null) checkWords(accepted, wordCount(valueCount));
    Source raw =
        new Source(
            valueCount * 8L,
            AccelerationProvider.Encoding.BYTES,
            b -> {
              for (int i = 0; i < valueCount; i++) b.putDouble(values[i]);
            });
    Source state =
        new Source(
            48,
            AccelerationProvider.Encoding.BYTES,
            b ->
                b.putLong(initial.count())
                    .putDouble(initial.sum())
                    .putDouble(initial.min())
                    .putDouble(initial.max())
                    .putDouble(initial.mean())
                    .putDouble(initial.variance()));
    FloatingStats[] out = new FloatingStats[1];
    Operation call =
        new Operation(
            Capability.FLOATING_STATS,
            valueCount,
            parameters(valueCount),
            accepted == null
                ? List.of(raw, state)
                : List.of(raw, state, longs(accepted, 0, wordCount(valueCount))),
            48,
            1);
    if (nativeCall(
        runtime,
        call,
        b ->
            out[0] =
                new FloatingStats(
                    b.getLong(0),
                    b.getDouble(8),
                    b.getDouble(16),
                    b.getDouble(24),
                    b.getDouble(32),
                    b.getDouble(40)))) return out[0];
    long count = initial.count();
    double sum = initial.sum(),
        min = initial.min(),
        max = initial.max(),
        mean = initial.mean(),
        variance = initial.variance();
    for (int i = 0; i < valueCount; i++) {
      if (accepted != null && (accepted[i >>> 6] & (1L << i)) == 0) continue;
      double v = values[i];
      sum += v;
      if (Double.compare(v, min) < 0) min = v;
      if (Double.compare(v, max) > 0) max = v;
      count++;
      double delta = v - mean;
      mean += delta / count;
      variance += delta * (v - mean);
    }
    return new FloatingStats(count, sum, min, max, mean, variance);
  }

  /** Strict full-block LZ4 prototype: no preset dictionary or partially decoded output. */
  public static byte[] decompressLz4(
      AccelerationRuntime runtime, byte[] compressed, int decodedLength) throws IOException {
    if (decodedLength < 0) throw new IllegalArgumentException("negative decoded length");
    byte[] out = new byte[decodedLength];
    long[] p = parameters(compressed.length);
    p[1] = decodedLength;
    Operation call =
        new Operation(
            Capability.COMPRESSION,
            compressed.length,
            p,
            List.of(bytes(compressed, 0, compressed.length)),
            decodedLength,
            1,
            decodedLength);
    if (!nativeCall(runtime, call, b -> b.get(out))) {
      var input = new org.apache.lucene.store.ByteArrayDataInput(compressed);
      int end = org.apache.lucene.util.compress.LZ4.decompress(input, decodedLength, out, 0);
      if (end != decodedLength || input.getPosition() != compressed.length)
        throw new IOException("LZ4 block length mismatch");
    }
    return out;
  }
}
