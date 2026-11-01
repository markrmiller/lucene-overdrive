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
package org.apache.lucene.benchmark.jmh;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.LongFunction;
import org.apache.lucene.acceleration.AccelerationProvider;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.JavaKernelOracle;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.benchmark.jmh.BenchmarkAccelerationSupport.PreparedInputs;
import org.apache.lucene.codecs.lucene104.ForUtil;
import org.apache.lucene.codecs.lucene104.PostingIndexInput;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.quantization.ScalarQuantizedVectorSimilarity;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Prepared kernels against the current Lucene Java path for the same operation. Setup owns all
 * input, output and native workspace allocation; every output element must be bit-identical to the
 * Java output before and after timing. CURRENT_JAVA runs check the same prepared wire layout
 * through {@link JavaKernelOracle} instead of a native pack. Each workload method names its Java
 * baseline.
 */
@SuppressForbidden(reason = "Benchmark identity, actual ISA and memory-drain diagnostics")
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 2,
    jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class PreparedKernelBenchmark {
  private static final int HISTOGRAM_BUCKETS = 1024;
  private static final int MULTI_SETS = 4;
  private static final int FOR_BITS = 7;
  private static final int U7_DIMENSIONS = 384;
  // More than 256 distinct values, so Lucene90 doc values never choose table encoding.
  private static final int MIN_INDEXED_DOCS = 1024;

  @Param({"CURRENT_JAVA", "NATIVE_REFERENCE", "NATIVE_OPTIMIZED", "MEASURED_DISPATCH"})
  public String implementation;

  @Param({
    "CARDINALITY",
    "INTERSECT_COUNT",
    "RANGE",
    "RANGE_FILTER",
    "DECODE",
    "AGGREGATE",
    "PREFIX_SUM",
    "HISTOGRAM",
    "MULTI_AND",
    "INTERSECT_SORTED",
    "NIBBLE_PACK",
    "NIBBLE_UNPACK",
    "FOR_DECODE",
    "U7_RESCORE"
  })
  public String op;

  /** Elements or words per operation; FOR_DECODE is one fixed block and U7_RESCORE uses rows. */
  @Param({"256", "4096", "65536"})
  public int size;

  private Workload workload;
  private final Deque<Closeable> resources = new ArrayDeque<>();
  private Path tempPath;
  private AccelerationRuntime runtime;
  private RequestMemoryScope scope;
  private AccelerationProvider.Kernel kernel;
  private AccelerationProvider.Batch prepared;
  private ByteBuffer kernelOutput;
  private long invocations;
  private boolean measuredDispatch;
  private JavaPath preparedJava;
  private LeafReader numericReader;
  private NumericDocValues numericDocValues;

  /** The current Java path: writes and returns its complete output. */
  @FunctionalInterface
  private interface JavaPath {
    Object run() throws IOException;
  }

  /** One input view, written once in setup in the kernel's little-endian wire encoding. */
  private record Staged(
      AccelerationProvider.Encoding encoding, long elements, Consumer<ByteBuffer> writer) {}

  /** Java baseline state plus the wire layout of the same data for one kernel id. */
  private record Workload(
      Capability capability,
      long[] parameters,
      List<Staged> inputs,
      long outputElements,
      int outputWidth,
      int dimensions,
      JavaPath currentJava,
      Consumer<ByteBuffer> verify) {

    AccelerationProvider.Shape shape() {
      long bytes = Math.addExact(320, outputElements * outputWidth);
      var encodings = EnumSet.noneOf(AccelerationProvider.Encoding.class);
      int alignment = 1;
      for (Staged input : inputs) {
        bytes += input.elements() * input.encoding().width;
        encodings.add(input.encoding());
        alignment = Math.max(alignment, input.encoding().width);
      }
      return new AccelerationProvider.Shape(
          parameters[0],
          Math.toIntExact(parameters[1]),
          bytes,
          encodings,
          AccelerationProvider.LayoutKind.NATIVE_BUFFER,
          alignment,
          true);
    }
  }

  @Setup(Level.Trial)
  public void setup() throws IOException {
    try {
      workload = createWorkload(new PreparedInputs());
      prepareCurrentJava();
      workload.currentJava().run();
      if (implementation.equals("CURRENT_JAVA") || implementation.equals("PREPARED_JAVA")) {
        if (implementation.equals("PREPARED_JAVA")) {
          if (preparedJava == null)
            throw new IllegalStateException(
                "prepared Java alternative only supports RANGE_FILTER/AGGREGATE");
          preparedJava.run();
        }
        if (!AccelerationRuntime.system().inventory().selected().isEmpty())
          throw new IllegalStateException("current Java baseline must not select native kernels");
        prepared = prepare(bytes -> ByteBuffer.allocate(Math.toIntExact(bytes)), 0);
        runOracle();
        workload.verify().accept(kernelOutput);
        return;
      }
      runtime = BenchmarkAccelerationSupport.open(implementation, workload.capability());
      measuredDispatch = implementation.equals("MEASURED_DISPATCH");
      kernel = measuredDispatch ? null : runtime.diagnosticKernel(workload.capability());
      var estimate =
          measuredDispatch
              ? runtime.admit(workload.capability(), workload.shape())
              : kernel.estimate(workload.shape());
      if (!estimate.supported() || estimate.stagingBytes() != 0)
        throw new IllegalStateException("unsupported prepared shape or implicit staging");
      scope = measuredDispatch ? runtime.newScope() : runtime.newQualificationScope();
      prepared = prepare(scope::allocate, estimate.scratchBytes());
      runKernel();
      workload.verify().accept(kernelOutput);
      System.out.println("Prepared kernel identity: " + runtime.inventory().selected());
    } catch (IOException | RuntimeException | Error failure) {
      try {
        release();
      } catch (IOException | RuntimeException suppressed) {
        failure.addSuppressed(suppressed);
      }
      throw failure;
    }
  }

  private Workload createWorkload(PreparedInputs random) throws IOException {
    return switch (op) {
      case "CARDINALITY" -> cardinality(random);
      case "INTERSECT_COUNT" -> intersectCount(random);
      case "RANGE" -> range(random);
      case "RANGE_FILTER" -> rangeFilter(random);
      case "DECODE" -> decode(random);
      case "AGGREGATE" -> aggregate(random);
      case "PREFIX_SUM" -> prefixSum(random);
      case "HISTOGRAM" -> histogram(random);
      case "MULTI_AND" -> multiAnd(random);
      case "INTERSECT_SORTED" -> intersectSorted(random);
      case "NIBBLE_PACK" -> nibblePack(random);
      case "NIBBLE_UNPACK" -> nibbleUnpack(random);
      case "FOR_DECODE" -> forDecode(random);
      case "U7_RESCORE" -> u7Rescore(random);
      default -> throw new IllegalArgumentException("unknown op: " + op);
    };
  }

  /** Kernel 6 against {@link FixedBitSet#cardinality()}, which requires clear ghost bits. */
  private Workload cardinality(PreparedInputs random) {
    int words = size, bitLength = words * Long.SIZE - 7;
    long[] bits = randomWords(random, words, bitLength);
    FixedBitSet set = new FixedBitSet(bits, bitLength);
    long[] count = new long[1];
    return new Workload(
        Capability.DOCSET_CARDINALITY,
        parameters(words, 0, bitLength),
        List.of(longs(bits)),
        1,
        Long.BYTES,
        0,
        () -> {
          count[0] = set.cardinality();
          return count;
        },
        output -> expectLongs(count, output));
  }

  /** Kernel 27 against {@link FixedBitSet#intersectionCount}, which requires clear ghost bits. */
  private Workload intersectCount(PreparedInputs random) {
    int words = size, bitLength = words * Long.SIZE - 7;
    long[] left = randomWords(random, words, bitLength);
    long[] right = randomWords(random, words, bitLength);
    FixedBitSet a = new FixedBitSet(left, bitLength), b = new FixedBitSet(right, bitLength);
    long[] count = new long[1];
    return new Workload(
        Capability.DOCSET_INTERSECT_COUNT,
        parameters(words, 0, bitLength),
        List.of(longs(left), longs(right)),
        1,
        Long.BYTES,
        0,
        () -> {
          count[0] = FixedBitSet.intersectionCount(a, b);
          return count;
        },
        output -> expectLongs(count, output));
  }

  /**
   * Kernel 8 against a word-at-a-time Java loop with the BlockOperations.range semantics. Lucene
   * has no public path from a decoded long[] to dense match words.
   */
  private Workload range(PreparedInputs random) {
    int n = size;
    long[] values = new long[n];
    for (int i = 0; i < n; i++) values[i] = random.nextLong();
    long minimum = -(1L << 62), maximum = 1L << 62;
    long[] words = new long[(n + 63) / 64];
    return new Workload(
        Capability.DOCVALUES_RANGE,
        parameters(n, 0, 0, minimum, maximum),
        List.of(longs(values)),
        words.length,
        Long.BYTES,
        0,
        () -> {
          for (int w = 0; w < words.length; w++) {
            long word = 0;
            for (int i = w << 6, end = Math.min(n, i + 64); i < end; i++)
              if (values[i] >= minimum && values[i] <= maximum) word |= 1L << i;
            words[w] = word;
          }
          return words;
        },
        output -> expectLongs(words, output));
  }

  /**
   * Kernel 28 against today's two steps: the RANGE loop above into match words, then {@link
   * FixedBitSet#and} with the selection.
   */
  private Workload rangeFilter(PreparedInputs random) {
    int n = size;
    long[] values = new long[n];
    for (int i = 0; i < n; i++) values[i] = random.nextLong();
    long minimum = -(1L << 62), maximum = 1L << 62;
    // About half the selection words are zero, so native can skip their values; the rest are
    // random.
    long[] selection = new long[(n + 63) / 64];
    for (int w = 0; w < selection.length; w++)
      selection[w] = (random.nextByte() & 1) == 0 ? 0 : random.nextLong();
    if ((n & 63) != 0) selection[selection.length - 1] &= (1L << (n & 63)) - 1;
    FixedBitSet selected = new FixedBitSet(selection, n);
    long[] words = new long[selection.length];
    FixedBitSet matches = new FixedBitSet(words, n);
    preparedJava =
        () -> {
          for (int w = 0; w < words.length; w++) {
            long selectedWord = selection[w], word = 0;
            if (selectedWord != 0) {
              for (int i = w << 6, end = Math.min(n, i + 64); i < end; i++)
                if (values[i] >= minimum && values[i] <= maximum) word |= 1L << i;
            }
            words[w] = word & selectedWord;
          }
          return words;
        };
    return new Workload(
        Capability.DOCVALUES_RANGE_FILTER,
        parameters(n, 0, 0, minimum, maximum),
        List.of(longs(values), longs(selection)),
        words.length,
        Long.BYTES,
        0,
        () -> {
          for (int w = 0; w < words.length; w++) {
            long word = 0;
            for (int i = w << 6, end = Math.min(n, i + 64); i < end; i++)
              if (values[i] >= minimum && values[i] <= maximum) word |= 1L << i;
            words[w] = word;
          }
          matches.and(selected);
          return words;
        },
        output -> expectLongs(words, output));
  }

  /**
   * Kernel 7 (width 32, multiplier 1, delta 0) against {@link NumericDocValues#longValues} on a
   * dense mapped Lucene90 segment, whose contiguous bulk path copies the mapped bytes and decodes
   * them through the active vectorization provider.
   */
  private Workload decode(PreparedInputs random) throws IOException {
    int n = size;
    long[] values = numericValues(random, Integer.SIZE);
    numericReader = indexNumeric(values);
    int[] docs = contiguousDocs(n);
    long[] decoded = new long[n];
    return new Workload(
        Capability.DOCVALUES_DECODE,
        parameters(n, Integer.SIZE, Integer.BYTES, 0, 0, 1, 0),
        List.of(bytes(littleEndian(values, n, Integer.BYTES))),
        n,
        Long.BYTES,
        0,
        () -> {
          numericDocValues.longValues(n, docs, decoded, 0);
          return decoded;
        },
        output -> expectLongs(decoded, output));
  }

  /**
   * Kernel 12 (width 64, identity transform, no selection) against the same {@link
   * NumericDocValues#longValues} bulk decode followed by the BlockOperations.aggregate Java loop.
   * Lucene has no fused decode/filter/aggregate path.
   */
  private Workload aggregate(PreparedInputs random) throws IOException {
    int n = size;
    long[] values = numericValues(random, Long.SIZE);
    numericReader = indexNumeric(values);
    int[] docs = contiguousDocs(n);
    long[] decoded = new long[n];
    long[] state = new long[6];
    long minimum = 1L << 61, maximum = 3L << 61;
    long[] preparedValues = Arrays.copyOf(values, n);
    preparedJava = () -> aggregateState(preparedValues, minimum, maximum, state);
    return new Workload(
        Capability.DOCVALUES_AGGREGATE,
        parameters(n, Long.SIZE, Long.BYTES, minimum, maximum, 1, 0),
        List.of(bytes(littleEndian(values, n, Long.BYTES))),
        6,
        Long.BYTES,
        0,
        () -> {
          numericDocValues.longValues(n, docs, decoded, 0);
          return aggregateState(decoded, minimum, maximum, state);
        },
        output -> expectLongs(state, output));
  }

  /** Same fused reduction over an already prepared heap column, without DocValues decoding. */
  private static long[] aggregateState(long[] values, long minimum, long maximum, long[] state) {
    long count = 0, sum = 0, min = 0, max = 0;
    boolean overflow = false;
    for (int i = 0; i < values.length; i++) {
      long value = values[i];
      if (value < minimum || value > maximum) continue;
      if (count == 0 || value < min) min = value;
      if (count == 0 || value > max) max = value;
      long next = sum + value;
      overflow |= ((sum ^ next) & (value ^ next)) < 0;
      sum = next;
      count++;
    }
    state[0] = count;
    state[1] = sum;
    state[2] = min;
    state[3] = max;
    state[4] = count != 0 ? 1 : 0;
    state[5] = overflow ? 1 : 0;
    return state;
  }

  /**
   * Kernel 21 against the Java loop of the package-private Lucene104PostingsReader.prefixSum,
   * writing a separate output like the kernel rather than in place.
   */
  private Workload prefixSum(PreparedInputs random) {
    int n = size, base = 12345;
    int[] deltas = new int[n];
    for (int i = 0; i < n; i++) deltas[i] = random.nextByte() & 127;
    int[] sums = new int[n];
    return new Workload(
        Capability.POSTINGS_PREFIX,
        parameters(n, 0, base),
        List.of(ints(deltas)),
        n,
        Integer.BYTES,
        0,
        () -> {
          int sum = base;
          for (int i = 0; i < n; i++) {
            sum += deltas[i];
            sums[i] = sum;
          }
          return sums;
        },
        output -> expectInts(sums, output));
  }

  /**
   * Kernel 11 against the per-document {@code counts[ordinal]++} loop of Lucene facet counting,
   * zeroing the reused histogram first because the kernel produces a fresh one.
   */
  private Workload histogram(PreparedInputs random) {
    int n = size;
    int[] ordinals = new int[n];
    for (int i = 0; i < n; i++)
      ordinals[i] =
          (random.nextByte() & 15) == 0
              ? -1
              : (int) Long.remainderUnsigned(random.nextLong(), HISTOGRAM_BUCKETS);
    int[] counts = new int[HISTOGRAM_BUCKETS];
    return new Workload(
        Capability.FACETS_COUNT,
        parameters(n, HISTOGRAM_BUCKETS),
        List.of(ints(ordinals)),
        HISTOGRAM_BUCKETS,
        Integer.BYTES,
        0,
        () -> {
          Arrays.fill(counts, 0);
          for (int ordinal : ordinals) if (ordinal != -1) counts[ordinal]++;
          return counts;
        },
        output -> expectInts(counts, output));
  }

  /** Kernel 16 (four sets) against a copy of the first set and {@link FixedBitSet#and} per set. */
  private Workload multiAnd(PreparedInputs random) {
    int words = size, bitLength = words * Long.SIZE - 7;
    FixedBitSet[] sets = new FixedBitSet[MULTI_SETS];
    long[] flat = new long[MULTI_SETS * words];
    for (int s = 0; s < MULTI_SETS; s++) {
      long[] bits = randomWords(random, words, bitLength);
      sets[s] = new FixedBitSet(bits, bitLength);
      System.arraycopy(bits, 0, flat, s * words, words);
    }
    FixedBitSet result = new FixedBitSet(bitLength);
    return new Workload(
        Capability.MULTI_INTERSECT,
        parameters(words, MULTI_SETS, bitLength),
        List.of(longs(flat)),
        words,
        Long.BYTES,
        MULTI_SETS,
        () -> {
          System.arraycopy(sets[0].getBits(), 0, result.getBits(), 0, words);
          for (int s = 1; s < MULTI_SETS; s++) result.and(sets[s]);
          return result.getBits();
        },
        output -> expectLongs(result.getBits(), output));
  }

  /**
   * Kernel 15 against the two-pointer merge of the BlockOperations.intersectSorted Java fallback,
   * into the kernel's fixed {@code [count, matches..., zero padding]} layout.
   */
  private Workload intersectSorted(PreparedInputs random) {
    int n = size;
    int[] left = sortedDocs(random, n), right = sortedDocs(random, n);
    int maxDoc = Math.max(left[n - 1], right[n - 1]) + 1;
    int[] result = new int[1 + n];
    return new Workload(
        Capability.SORTED_INTERSECT,
        parameters(n, n, maxDoc),
        List.of(ints(left), ints(right)),
        result.length,
        Integer.BYTES,
        0,
        () -> {
          int l = 0, r = 0, count = 0;
          while (l < n && r < n) {
            if (left[l] == right[r]) {
              result[++count] = left[l];
              l++;
              r++;
            } else if (left[l] < right[r]) l++;
            else r++;
          }
          result[0] = count;
          return result;
        },
        output -> expectInts(result, output));
  }

  /** Kernel 10 against the Java loop of OffHeapScalarQuantizedVectorValues.packNibbles. */
  private Workload nibblePack(PreparedInputs random) {
    int n = size;
    byte[] unpacked = new byte[n];
    for (int i = 0; i < n; i++) unpacked[i] = (byte) (random.nextByte() & 15);
    byte[] packed = new byte[n / 2];
    return new Workload(
        Capability.QUANTIZE_PACK,
        parameters(n),
        List.of(bytes(unpacked)),
        packed.length,
        1,
        0,
        () -> {
          for (int i = 0; i < packed.length; i++)
            packed[i] = (byte) (unpacked[i] << 4 | unpacked[packed.length + i]);
          return packed;
        },
        output -> expectBytes(packed, output));
  }

  /** Kernel 22 against the Java loop of OffHeapScalarQuantizedVectorValues.unpackNibbles. */
  private Workload nibbleUnpack(PreparedInputs random) {
    int n = size;
    byte[] packed = new byte[n];
    random.fill(packed);
    byte[] unpacked = new byte[2 * n];
    return new Workload(
        Capability.QUANTIZE_UNPACK,
        parameters(n),
        List.of(bytes(packed)),
        unpacked.length,
        1,
        0,
        () -> {
          for (int i = 0; i < n; i++) {
            unpacked[i] = (byte) ((packed[i] >> 4) & 0x0F);
            unpacked[n + i] = (byte) (packed[i] & 0x0F);
          }
          return unpacked;
        },
        output -> expectBytes(unpacked, output));
  }

  /**
   * Kernel 25 (7 bits, one 256-value block; {@code size} is ignored) against {@link
   * PostingIndexInput#decode}, Lucene104 ForUtil over the active PostingDecodingUtil, reading a
   * mapped file.
   */
  private Workload forDecode(PreparedInputs random) throws IOException {
    byte[] encoded = new byte[FOR_BITS * 32];
    random.fill(encoded);
    Directory directory = mappedDirectory();
    try (IndexOutput out = directory.createOutput("block", IOContext.DEFAULT)) {
      out.writeBytes(encoded, encoded.length);
      // Trailing bytes so vectorized reads never reach EOF, as in PostingIndexInputBenchmark.
      out.writeBytes(new byte[64], 64);
    }
    IndexInput in = directory.openInput("block", IOContext.DEFAULT);
    resources.push(in);
    PostingIndexInput postings = new PostingIndexInput(in, new ForUtil());
    int[] decoded = new int[ForUtil.BLOCK_SIZE];
    return new Workload(
        Capability.POSTINGS_BLOCK,
        parameters(ForUtil.BLOCK_SIZE, FOR_BITS),
        List.of(bytes(encoded)),
        ForUtil.BLOCK_SIZE,
        Integer.BYTES,
        0,
        () -> {
          in.seek(0);
          postings.decode(FOR_BITS, decoded);
          return decoded;
        },
        output -> expectInts(decoded, output));
  }

  /**
   * Kernel 13 (dot product, 384 dimensions, rows capped at 16M coordinates) against {@link
   * ScalarQuantizedVectorSimilarity} for 7 bits, whose dot product is VectorUtil.uint8DotProduct
   * from the active vectorization provider.
   */
  private Workload u7Rescore(PreparedInputs random) {
    int dims = U7_DIMENSIONS, rows = Math.min(size, (16 << 20) / dims);
    byte[] query = new byte[dims];
    for (int d = 0; d < dims; d++) query[d] = (byte) (random.nextByte() & 127);
    byte[][] documents = new byte[rows][dims];
    byte[] flat = new byte[rows * dims];
    float[] corrections = new float[rows];
    for (int r = 0; r < rows; r++) {
      for (int d = 0; d < dims; d++) documents[r][d] = (byte) (random.nextByte() & 127);
      System.arraycopy(documents[r], 0, flat, r * dims, dims);
      corrections[r] = random.nextByte() / 256f;
    }
    float multiplier = 0x1p-12f, queryCorrection = 0.125f;
    var similarity =
        ScalarQuantizedVectorSimilarity.fromVectorSimilarity(
            VectorSimilarityFunction.DOT_PRODUCT, multiplier, (byte) 7);
    float[] scores = new float[rows];
    return new Workload(
        Capability.QUANTIZED_RESCORE,
        parameters(
            rows,
            dims,
            dims,
            VectorSimilarityFunction.DOT_PRODUCT.ordinal(),
            0,
            Integer.toUnsignedLong(Float.floatToRawIntBits(multiplier)),
            Integer.toUnsignedLong(Float.floatToRawIntBits(queryCorrection))),
        List.of(bytes(query), bytes(flat), floats(corrections)),
        rows,
        Float.BYTES,
        dims,
        () -> {
          for (int r = 0; r < rows; r++)
            scores[r] = similarity.score(query, queryCorrection, documents[r], corrections[r]);
          return scores;
        },
        output -> expectFloats(scores, output));
  }

  /** Random words with the bits past {@code bitLength} clear, as FixedBitSet requires. */
  private static long[] randomWords(PreparedInputs random, int words, int bitLength) {
    long[] bits = new long[words];
    for (int i = 0; i < words; i++) bits[i] = random.nextLong();
    bits[words - 1] &= (1L << (bitLength & 63)) - 1;
    return bits;
  }

  private static int[] sortedDocs(PreparedInputs random, int count) {
    int[] docs = new int[count];
    for (int i = 0, doc = -1; i < count; i++) docs[i] = doc += 1 + (random.nextByte() & 7);
    return docs;
  }

  private static int[] contiguousDocs(int count) {
    int[] docs = new int[count];
    for (int i = 0; i < count; i++) docs[i] = i;
    return docs;
  }

  /**
   * Dense field values with gcd 1, a positive minimum and a full-width maximum, so Lucene90 stores
   * them unchanged at exactly {@code bits} per value in one block.
   */
  private long[] numericValues(PreparedInputs random, int bits) {
    long mask = bits == Long.SIZE ? Long.MAX_VALUE : (1L << bits) - 1;
    long[] values = new long[Math.max(size, MIN_INDEXED_DOCS)];
    for (int i = 0; i < values.length; i++) values[i] = Math.max(1, random.nextLong() & mask);
    values[0] = mask;
    return values;
  }

  private static byte[] littleEndian(long[] values, int count, int width) {
    byte[] raw = new byte[count * width];
    for (int i = 0; i < count; i++)
      for (int j = 0; j < width; j++) raw[i * width + j] = (byte) (values[i] >>> (j * 8));
    return raw;
  }

  /** Indexes one dense numeric field into a single mapped segment. */
  private LeafReader indexNumeric(long[] values) throws IOException {
    Directory directory = mappedDirectory();
    try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
      for (long value : values) {
        Document doc = new Document();
        doc.add(new NumericDocValuesField("value", value));
        writer.addDocument(doc);
      }
      writer.forceMerge(1);
    }
    DirectoryReader reader = DirectoryReader.open(directory);
    resources.push(reader);
    return reader.leaves().get(0).reader();
  }

  private Directory mappedDirectory() throws IOException {
    tempPath = Files.createTempDirectory("preparedKernel");
    Directory directory = MMapDirectory.open(tempPath);
    resources.push(directory);
    return directory;
  }

  private static long[] parameters(long... leading) {
    return Arrays.copyOf(leading, 8);
  }

  private static Staged bytes(byte[] values) {
    return new Staged(AccelerationProvider.Encoding.BYTES, values.length, b -> b.put(values));
  }

  private static Staged longs(long[] values) {
    return new Staged(
        AccelerationProvider.Encoding.I64,
        values.length,
        b -> {
          for (long value : values) b.putLong(value);
        });
  }

  private static Staged ints(int[] values) {
    return new Staged(
        AccelerationProvider.Encoding.I32,
        values.length,
        b -> {
          for (int value : values) b.putInt(value);
        });
  }

  private static Staged floats(float[] values) {
    return new Staged(
        AccelerationProvider.Encoding.F32,
        values.length,
        b -> {
          for (float value : values) b.putFloat(value);
        });
  }

  /** Allocates and writes every view once; the timed method reuses all of them. */
  private AccelerationProvider.Batch prepare(LongFunction<ByteBuffer> allocate, long scratchBytes) {
    List<AccelerationProvider.Input> inputs = new ArrayList<>();
    for (Staged staged : workload.inputs()) {
      ByteBuffer buffer =
          allocate
              .apply(staged.elements() * staged.encoding().width)
              .order(ByteOrder.LITTLE_ENDIAN);
      staged.writer().accept(buffer);
      inputs.add(
          new AccelerationProvider.Input(buffer.flip(), staged.elements(), staged.encoding()));
    }
    kernelOutput =
        allocate
            .apply(workload.outputElements() * workload.outputWidth())
            .order(ByteOrder.LITTLE_ENDIAN);
    ByteBuffer parameters = allocate.apply(64).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 8; i++) parameters.putLong(i * Long.BYTES, workload.parameters()[i]);
    return new AccelerationProvider.Batch(
        workload.capability(),
        inputs,
        new AccelerationProvider.Output(kernelOutput, workload.outputElements()),
        parameters,
        allocate.apply(scratchBytes),
        allocate.apply(256));
  }

  private void runKernel() throws IOException {
    if (measuredDispatch) {
      if (!runtime.execute(prepared, workload.parameters()[0], 0, scope))
        throw new IllegalStateException("qualified prepared " + op + " declined native execution");
    } else {
      check(kernel.execute(prepared));
    }
  }

  private void runOracle() throws IOException {
    check(JavaKernelOracle.execute(prepared));
  }

  private void check(AccelerationProvider.Result result) {
    if (result == null
        || result.status() != AccelerationProvider.Status.OK
        || result.detail() != 0
        || result.outputElements() != workload.outputElements())
      throw new IllegalStateException("prepared " + op + " kernel failed: " + result);
  }

  private static void expectLongs(long[] expected, ByteBuffer actual) {
    for (int i = 0; i < expected.length; i++)
      if (expected[i] != actual.getLong(i * Long.BYTES)) mismatch(i);
  }

  private static void expectInts(int[] expected, ByteBuffer actual) {
    for (int i = 0; i < expected.length; i++)
      if (expected[i] != actual.getInt(i * Integer.BYTES)) mismatch(i);
  }

  private static void expectBytes(byte[] expected, ByteBuffer actual) {
    for (int i = 0; i < expected.length; i++) if (expected[i] != actual.get(i)) mismatch(i);
  }

  private static void expectFloats(float[] expected, ByteBuffer actual) {
    for (int i = 0; i < expected.length; i++)
      if (Float.floatToRawIntBits(expected[i])
          != Float.floatToRawIntBits(actual.getFloat(i * Float.BYTES))) mismatch(i);
  }

  private static void mismatch(int element) {
    throw new IllegalStateException("prepared kernel parity at element " + element);
  }

  /** Acquires a fresh forward-only DocValues iterator outside the timed decode/reduction. */
  @Setup(Level.Invocation)
  public void beforeInvocation() throws IOException {
    if (implementation.equals("CURRENT_JAVA")) prepareCurrentJava();
  }

  private void prepareCurrentJava() throws IOException {
    if (numericReader != null) numericDocValues = numericReader.getNumericDocValues("value");
  }

  /**
   * Reuses inputs, outputs, descriptors and scope. JMH consumes the complete output reference,
   * keeping all stores observable without adding a per-element checksum to the timed operation.
   */
  @Benchmark
  public Object run() throws IOException {
    invocations++;
    if (runtime == null)
      return implementation.equals("PREPARED_JAVA")
          ? preparedJava.run()
          : workload.currentJava().run();
    runKernel();
    return kernelOutput;
  }

  @TearDown(Level.Trial)
  public void finish() throws IOException {
    try {
      prepareCurrentJava();
      workload.currentJava().run();
      if (runtime == null) {
        if (implementation.equals("PREPARED_JAVA")) preparedJava.run();
        runOracle();
      } else runKernel();
      workload.verify().accept(kernelOutput);
      if (measuredDispatch) {
        var metrics = runtime.inventory().counters().get(workload.capability());
        if (metrics == null || metrics.nativeCalls() == 0)
          throw new IllegalStateException("qualified prepared benchmark made no native calls");
        System.out.println("Prepared production metrics: " + metrics);
      }
      System.out.println("Prepared kernel calls: " + invocations);
    } finally {
      release();
    }
    if (runtime != null) BenchmarkAccelerationSupport.verifyDrained(runtime);
  }

  private void release() throws IOException {
    List<Closeable> owned = new ArrayList<>();
    if (scope != null) owned.add(scope::close);
    if (runtime != null) owned.add(runtime::close);
    owned.addAll(resources);
    try {
      IOUtils.close(owned);
    } finally {
      if (tempPath != null) IOUtils.rm(tempPath);
    }
  }
}
