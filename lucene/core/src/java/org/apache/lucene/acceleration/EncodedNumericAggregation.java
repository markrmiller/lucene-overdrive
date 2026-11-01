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
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Set;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.internal.acceleration.EncodedNumericValues;
import org.apache.lucene.util.FixedBitSet;

/**
 * Optional exact integer aggregate over a bounded codec-owned encoded interval.
 *
 * <p>A null result asks the caller to retain its Java path. This prototype supports only dense,
 * unblocked affine 32/64-bit values with a contiguous mapping and an eligible native profile. No
 * decoded column is allocated. The reader is pinned until the synchronous call and publication
 * finish; live docs and an optional full-reader selection are intersected before aggregation.
 *
 * @lucene.experimental
 */
public final class EncodedNumericAggregation {
  private EncodedNumericAggregation() {}

  public static BlockOperations.Aggregate tryAggregate(
      AccelerationRuntime runtime,
      LeafReader reader,
      String field,
      int fromDoc,
      int toDoc,
      FixedBitSet selection,
      long minimum,
      long maximum)
      throws IOException {
    return tryAggregate(
        runtime, reader, field, fromDoc, toDoc, selection, minimum, maximum, "generic");
  }

  /**
   * Aggregates under an exact consumer contract. Ineligible contexts return null before optional
   * codec access or native storage; the completed mapped batch is independently checked again.
   */
  public static BlockOperations.Aggregate tryAggregate(
      AccelerationRuntime runtime,
      LeafReader reader,
      String field,
      int fromDoc,
      int toDoc,
      FixedBitSet selection,
      long minimum,
      long maximum,
      String consumerContext)
      throws IOException {
    Objects.requireNonNull(runtime);
    Objects.requireNonNull(reader);
    Objects.requireNonNull(field);
    Objects.checkFromToIndex(fromDoc, toDoc, reader.maxDoc());
    if (selection != null && selection.length() != reader.maxDoc())
      throw new IllegalArgumentException("selection must cover exactly this reader");
    int rows = toDoc - fromDoc;
    if (rows == 0 || !runtime.supports(Capability.DOCVALUES_AGGREGATE, consumerContext))
      return null;
    try (var pin = LeafReaderPin.tryAcquire(reader)) {
      if (pin == null)
        throw new org.apache.lucene.store.AlreadyClosedException("numeric reader closed");
      var values = reader.getNumericDocValues(field);
      if (!(values instanceof EncodedNumericValues encoded)) return null;
      var source = encoded.encodedSource();
      if (source == null
          || source.input() == null
          || source.rows() != reader.maxDoc()
          || source.bits() != 32 && source.bits() != 64) return null;
      int width = source.bits() / Byte.SIZE;
      long required = Math.multiplyExact((long) source.rows(), width);
      if (source.offset() < 0
          || source.length() < required
          || source.offset() > source.input().length() - source.length()) return null;
      var live = reader.getLiveDocs();
      boolean masked = selection != null || live != null;
      int words = FixedBitSet.bits2words(rows);
      long maskBytes = masked ? (long) words * Long.BYTES : 0;
      if (maskBytes > CallFrame.DATA_BYTES - 6 * Long.BYTES) return null;
      long inputBytes = (long) rows * width;
      var encodings =
          masked
              ? Set.of(AccelerationProvider.Encoding.BYTES, AccelerationProvider.Encoding.I64)
              : Set.of(AccelerationProvider.Encoding.BYTES);
      var shape =
          new AccelerationProvider.Shape(
              rows,
              source.bits(),
              inputBytes + maskBytes + 6 * Long.BYTES + 64 + CallFrame.METADATA_BYTES,
              encodings,
              AccelerationProvider.LayoutKind.NATIVE_BUFFER,
              1,
              true);
      if (!runtime.supports(
          Capability.DOCVALUES_AGGREGATE, shape, minimum, maximum, consumerContext, true))
        return null;
      var input =
          MappedSlices.slice(source.input(), source.offset() + (long) fromDoc * width, inputBytes);
      if (input == null) return null;
      try (var frame = runtime.acquireFrame()) {
        if (frame == null) return null;
        var inputs = new ArrayList<AccelerationProvider.Input>(masked ? 2 : 1);
        inputs.add(
            new AccelerationProvider.Input(input, inputBytes, AccelerationProvider.Encoding.BYTES));
        if (masked) {
          var mask = frame.data(6 * Long.BYTES, Math.toIntExact(maskBytes));
          for (int word = 0; word < words; word++) {
            if ((word & 63) == 0) runtime.checkpoint();
            long selected = 0;
            int first = word << 6;
            int count = Math.min(rows - first, Long.SIZE);
            if (live == null) {
              selected = selectionWord(selection, fromDoc + first, count);
            } else {
              for (int i = first, end = first + count; i < end; i++) {
                int doc = fromDoc + i;
                if ((selection == null || selection.get(doc)) && live.get(doc)) selected |= 1L << i;
              }
            }
            mask.putLong(word * Long.BYTES, selected);
          }
          inputs.add(
              new AccelerationProvider.Input(mask, words, AccelerationProvider.Encoding.I64));
        }
        var parameters = frame.parameters();
        for (int i = 0; i < 64; i += Long.BYTES) parameters.putLong(i, 0);
        parameters.putLong(0, rows);
        parameters.putLong(8, source.bits());
        parameters.putLong(16, width);
        parameters.putLong(24, minimum);
        parameters.putLong(32, maximum);
        parameters.putLong(40, source.multiplier());
        parameters.putLong(48, source.base());
        var output = frame.data(0, 6 * Long.BYTES);
        var batch =
            new AccelerationProvider.Batch(
                Capability.DOCVALUES_AGGREGATE,
                inputs,
                new AccelerationProvider.Output(output, 6),
                parameters,
                frame.scratch(0),
                frame.metadata());
        if (!runtime.execute(batch, rows, maskBytes, frame.scope(), consumerContext)) return null;
        var result = output.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        return new BlockOperations.Aggregate(
            result.getLong(0),
            result.getLong(8),
            result.getLong(16),
            result.getLong(24),
            result.getLong(32) != 0,
            result.getLong(40) != 0);
      }
    }
  }

  // A selection interval can begin between words. Two reads suffice, and tail bits stay zero.
  static long selectionWord(FixedBitSet selection, int fromDoc, int count) {
    if (count < 0 || count > Long.SIZE) throw new IllegalArgumentException("invalid word length");
    Objects.checkFromIndexSize(fromDoc, count, selection.length());
    if (count == 0) return 0;
    long[] words = selection.getBits();
    int index = fromDoc >>> 6, shift = fromDoc & 63;
    long value = words[index] >>> shift;
    if (shift != 0 && index + 1 < words.length) value |= words[index + 1] << (64 - shift);
    return count == Long.SIZE ? value : value & ((1L << count) - 1);
  }
}
