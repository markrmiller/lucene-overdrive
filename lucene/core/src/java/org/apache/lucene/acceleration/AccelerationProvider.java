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
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Startup-only service contract for optional synchronous acceleration packs. The types in this
 * contract depend only on the JDK. A provider must not retain borrowed buffers, allocate request
 * memory, start threads or publish output. These experimental contracts are pinned to the host's
 * bridge build.
 *
 * @lucene.experimental
 */
public interface AccelerationProvider {
  /** Describes the provider without loading a native library. */
  Descriptor descriptor();

  /** Prepares a previously verified pack using approved host loading services. */
  Prepared prepare(HostContext host) throws IOException;

  /** Immutable provider identity. */
  record Descriptor(
      String id,
      String version,
      int priority,
      Set<Capability> capabilities,
      Map<Capability, KernelDescriptor> kernels) {
    public Descriptor(String id, String version, int priority, Set<Capability> capabilities) {
      this(id, version, priority, capabilities, Map.of());
    }

    public Descriptor {
      Objects.requireNonNull(id);
      Objects.requireNonNull(version);
      capabilities = Set.copyOf(capabilities);
      kernels = Map.copyOf(kernels);
      if (!capabilities.containsAll(kernels.keySet()))
        throw new IllegalArgumentException("unadvertised kernel description");
    }
  }

  /** Stable operation/schema/arithmetic identity, independent of an implementation or ISA. */
  record KernelKey(String operation, int schemaMajor, String arithmeticProfile) {
    public KernelKey {
      Objects.requireNonNull(operation);
      Objects.requireNonNull(arithmeticProfile);
      if (operation.isEmpty() || arithmeticProfile.isEmpty() || schemaMajor < 1)
        throw new IllegalArgumentException("invalid kernel identity");
    }

    public static KernelKey of(Capability capability) {
      return new KernelKey(capability.id, 1, capability.arithmeticProfile);
    }
  }

  /** Declared storage/layout contract; no address can outlive its enclosing owner. */
  enum LayoutKind {
    HEAP_BUFFER,
    NATIVE_BUFFER,
    MAPPED_BUFFER,
    CONTIGUOUS,
    STRIDED,
    PACKED,
    SORTED
  }

  /** Complete bounded workspace formula, evaluated with checked long arithmetic. */
  record Workspace(long fixedBytes, long perElementBytes, long perDimensionBytes) {
    public Workspace {
      if (fixedBytes < 0 || perElementBytes < 0 || perDimensionBytes < 0)
        throw new IllegalArgumentException("negative workspace formula");
    }

    public long bytes(Shape shape) {
      return Math.addExact(
          fixedBytes,
          Math.addExact(
              Math.multiplyExact(perElementBytes, shape.elements()),
              Math.multiplyExact(perDimensionBytes, shape.dimensions())));
    }
  }

  /** Immutable, testable implementation claim; qualification remains a separate startup step. */
  record KernelDescriptor(
      KernelKey key,
      Set<Encoding> encodings,
      Set<LayoutKind> layouts,
      long maximumElements,
      int maximumDimensions,
      long maximumBytes,
      int minimumAlignment,
      Set<String> requiredFeatures,
      Workspace scratch,
      Workspace staging,
      boolean threadSafe,
      String testFingerprint) {
    public KernelDescriptor {
      Objects.requireNonNull(key);
      encodings = Set.copyOf(encodings);
      layouts = Set.copyOf(layouts);
      requiredFeatures = Set.copyOf(requiredFeatures);
      Objects.requireNonNull(scratch);
      Objects.requireNonNull(staging);
      Objects.requireNonNull(testFingerprint);
      if (maximumElements < 0
          || maximumDimensions < 0
          || maximumBytes < 0
          || minimumAlignment < 1
          || Integer.bitCount(minimumAlignment) != 1
          || encodings.isEmpty()
          || layouts.isEmpty()
          || !threadSafe
          || !testFingerprint.matches("[0-9a-f]{64}"))
        throw new IllegalArgumentException("invalid kernel description");
    }

    public boolean supports(Shape shape) {
      return shape.elements() <= maximumElements
          && shape.dimensions() <= maximumDimensions
          && shape.bytes() <= maximumBytes
          && shape.alignment() >= minimumAlignment
          && encodings.containsAll(shape.encodings())
          && layouts.contains(shape.layout());
    }

    /** Baseline contract for legacy providers; more restrictive providers override this claim. */
    public static KernelDescriptor reference(Capability capability) {
      return new KernelDescriptor(
          KernelKey.of(capability),
          Set.of(Encoding.values()),
          Set.of(LayoutKind.values()),
          16L << 20,
          Integer.MAX_VALUE,
          128L << 20,
          1,
          Set.of(),
          new Workspace(0, 0, 0),
          new Workspace(0, 0, 0),
          true,
          "0000000000000000000000000000000000000000000000000000000000000000");
    }
  }

  /** Approved startup services. No unbounded allocator is exposed. */
  interface HostContext {
    /** Exact Java bridge identity. */
    String bridgeBuild();

    /** Loads only the library already verified for this provider. */
    Prepared prepareNative(Descriptor descriptor) throws IOException;
  }

  /** Prepared immutable kernel inventory. */
  interface Prepared {
    /** Inventory frozen before serving requests. */
    Set<Capability> capabilities();

    /** Returns a kernel; request-time discovery is forbidden. */
    Kernel bind(Capability capability);

    /** Baseline CPU tier, never inferred from os.arch as an optional ISA guarantee. */
    String isa();

    /** Actual implementation tier for one capability; reference kernels may use a lower tier. */
    default String isa(Capability capability) {
      return isa();
    }

    /** Describes the bound kernel before any request allocations or downcalls. */
    default KernelDescriptor kernelDescriptor(Capability capability) {
      return KernelDescriptor.reference(capability);
    }
  }

  /** A stateless, thread-safe synchronous kernel. */
  interface Kernel {
    /** Pure shape admission. All lengths include only logical spans, not implicit padding. */
    Estimate estimate(Shape shape);

    /** Observed critical-call durations, bounded to this kernel, never a deadline guarantee. */
    default CriticalCallBudget.Snapshot criticalCalls() {
      return CriticalCallBudget.Snapshot.EMPTY;
    }

    /** Writes private caller-owned output only; the host owns publication. */
    Result execute(Batch batch) throws IOException;
  }

  /** Checked shape information shared with both application hosts. */
  record Shape(
      long elements,
      int dimensions,
      long bytes,
      Set<Encoding> encodings,
      LayoutKind layout,
      int alignment,
      boolean nativeAddressable) {
    public Shape(long elements, int dimensions, long bytes) {
      this(elements, dimensions, bytes, Set.of(Encoding.BYTES), LayoutKind.NATIVE_BUFFER, 1, true);
    }

    public Shape {
      encodings = Set.copyOf(encodings);
      Objects.requireNonNull(layout);
      if (elements < 0
          || dimensions < 0
          || bytes < 0
          || alignment < 1
          || Integer.bitCount(alignment) != 1
          || encodings.isEmpty()) {
        throw new IllegalArgumentException("negative shape");
      }
    }
  }

  /** Complete additional workspace bound. */
  record Estimate(boolean supported, long scratchBytes, long stagingBytes, long estimatedNanos) {
    public Estimate(boolean supported, long scratchBytes, long stagingBytes) {
      this(supported, scratchBytes, stagingBytes, 0);
    }

    public Estimate {
      if (scratchBytes < 0 || stagingBytes < 0 || estimatedNanos < 0) {
        throw new IllegalArgumentException("negative estimate");
      }
    }
  }

  /** Registered little-endian input encoding. */
  enum Encoding {
    /** Raw bytes, signedness belongs to the operation. */
    BYTES(1, 1),
    /** 64-bit little-endian integer words. */
    I64(2, 8),
    /** 32-bit little-endian integer words. */
    I32(3, 4),
    /** IEEE binary32 little-endian values. */
    F32(4, 4);

    public final int id;
    public final int width;

    Encoding(int id, int width) {
      this.id = id;
      this.width = width;
    }
  }

  /** Typed heap/native storage borrowed without materialization under its owner's lifetime. */
  interface BorrowedSpan {
    /** Exact logical bytes of this borrowed view. */
    long byteSize();

    /** Whether this view resides in native storage rather than a typed Java array. */
    boolean isNative();

    /** Reads one byte without materializing the backing storage. */
    byte byteAt(long offset);

    /** Actual platform segment, never a raw address. Validated by the FFM bridge. */
    Object foreignSegment();

    /** Reads an unaligned little-endian word for bounded Java validation/reference execution. */
    default long littleEndianLong(long offset) {
      Objects.checkFromIndexSize(offset, 8, byteSize());
      long value = 0;
      for (int i = 0; i < 8; i++) value |= Byte.toUnsignedLong(byteAt(offset + i)) << (i * 8);
      return value;
    }
  }

  /** A view borrowed under an enclosing request scope or input lease. */
  record Input(ByteBuffer bytes, long elements, Encoding encoding, BorrowedSpan span) {
    public Input(ByteBuffer bytes, long elements, Encoding encoding) {
      this(bytes, elements, encoding, null);
    }

    public Input {
      Objects.requireNonNull(encoding);
      if ((bytes == null) == (span == null))
        throw new IllegalArgumentException("one input storage required");
      long capacity = bytes == null ? span.byteSize() : bytes.remaining();
      if (capacity < 0 || elements < 0 || Math.multiplyExact(elements, encoding.width) > capacity)
        throw new IllegalArgumentException("input capacity");
      if (bytes != null) bytes = bytes.slice().asReadOnlyBuffer();
    }

    public static Input borrowed(BorrowedSpan span, long elements, Encoding encoding) {
      return new Input(null, elements, encoding, Objects.requireNonNull(span));
    }

    public long byteSize() {
      return bytes == null ? span.byteSize() : bytes.remaining();
    }

    public boolean isNative() {
      return bytes == null ? span.isNative() : bytes.isDirect();
    }

    public byte byteAt(long offset) {
      Objects.checkIndex(offset, byteSize());
      return bytes == null
          ? span.byteAt(offset)
          : bytes.get(Math.toIntExact(offset + bytes.position()));
    }

    public long littleEndianLong(long offset) {
      if (span != null) return span.littleEndianLong(offset);
      Objects.checkFromIndexSize(offset, 8, byteSize());
      return bytes
          .duplicate()
          .order(java.nio.ByteOrder.LITTLE_ENDIAN)
          .getLong(Math.toIntExact(offset + bytes.position()));
    }
  }

  /** Private output; its contents cannot be published until complete success. */
  record Output(ByteBuffer bytes, long elements) {
    public Output {
      Objects.requireNonNull(bytes);
      if (bytes.isReadOnly() || elements < 0) {
        throw new IllegalArgumentException("output capacity");
      }
      bytes = bytes.slice();
    }
  }

  /** ABI v1 has one logical output and an operation-specific 64-byte parameter schema. */
  record Batch(
      Capability capability,
      List<Input> inputs,
      Output output,
      ByteBuffer parameters,
      ByteBuffer scratch,
      ByteBuffer metadata) {
    public Batch {
      Objects.requireNonNull(capability);
      inputs = List.copyOf(inputs);
      Objects.requireNonNull(output);
      Objects.requireNonNull(parameters);
      Objects.requireNonNull(scratch);
      Objects.requireNonNull(metadata);
      if (parameters.remaining() != 64 || metadata.remaining() < 256) {
        throw new IllegalArgumentException("ABI metadata capacity");
      }
    }
  }

  /** Bounded native status; errors never expose a partially written result. */
  enum Status {
    OK,
    UNSUPPORTED,
    INVALID_INPUT,
    INTERNAL_ERROR
  }

  /** Complete result metadata returned by one synchronous call. */
  record Result(Status status, long outputElements, int detail) {}
}
