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
package org.apache.lucene.internal.acceleration;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.acceleration.AccelerationProvider;
import org.apache.lucene.acceleration.Capability;
import org.apache.lucene.acceleration.CriticalCallBudget;
import org.apache.lucene.acceleration.NativeBridge;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.NativeAllocator;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.VectorUtil;

/** Controlled FFM boundary, reached only on explicitly budgeted JDK25 operation paths. */
public final class FfmBridge implements NativeBridge {
  private static final MemoryLayout INPUT =
      MemoryLayout.structLayout(ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT);
  private static final MemoryLayout OUTPUT =
      MemoryLayout.structLayout(ADDRESS, JAVA_LONG, JAVA_LONG);
  private static final MemoryLayout CALL =
      MemoryLayout.structLayout(
          JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS,
          JAVA_LONG);
  private static final MemoryLayout RESULT =
      MemoryLayout.structLayout(JAVA_INT, JAVA_INT, JAVA_LONG);

  /**
   * Calls whose total buffer bytes stay within this bound use a critical downcall when the runtime
   * offers one. Each bound kernel retires critical execution after an observed slow call; the first
   * call and calls already in flight cannot have a hard duration bound. The kernels never
   * deliberately block, allocate, or call back into Java. With a pack's direct entry, such calls
   * may also pass Java heap inputs and output without staging copies.
   */
  static final long CRITICAL_MAX_BYTES = 256 * 1024;

  private static final AccelerationProvider.Status[] STATUSES =
      AccelerationProvider.Status.values();

  /** Does not invoke any restricted method or allocate memory. */
  public FfmBridge() {}

  /**
   * {@code Linker.Option.critical(heapAccess)} on JDK 22+, absent on the JDK 21 API this compiles
   * to.
   */
  @SuppressForbidden(reason = "Resolves the JDK 22+ critical linker option without a JDK 21 API")
  private static Linker.Option[] criticalOption(boolean heapAccess) {
    try {
      Method critical = Linker.Option.class.getMethod("critical", boolean.class);
      return new Linker.Option[] {(Linker.Option) critical.invoke(null, heapAccess)};
    } catch (ReflectiveOperationException | RuntimeException e) {
      return null;
    }
  }

  /**
   * The ordinary handle, a critical descriptor-only handle, and (when the pack exports it and the
   * JDK offers heap access) a critical handle whose input and output pointers are arguments, so
   * they may be heap segments.
   */
  private record Handles(
      MethodHandle execute,
      MethodHandle critical,
      MethodHandle direct,
      MethodHandle pinned,
      MethodHandle pinnedCritical) {}

  @Override
  public void verifyPlatform(String minimumOs, String minimumLibc) throws IOException {
    if (!minimumOs.isEmpty()
        && NativeBridge.compareVersion(System.getProperty("os.version", "0"), minimumOs) < 0)
      throw new IOException("kernel version below pack minimum");
    if (minimumLibc.isEmpty()) return;
    verifyHost();
    MethodHandle version =
        Linker.nativeLinker()
            .downcallHandle(
                Linker.nativeLinker()
                    .defaultLookup()
                    .find("gnu_get_libc_version")
                    .orElseThrow(() -> new IOException("glibc version symbol unavailable")),
                FunctionDescriptor.of(ADDRESS));
    String actual;
    try {
      MemorySegment address = (MemorySegment) version.invokeExact();
      if (address.address() == 0) throw new IOException("null libc version");
      MemorySegment text = address.reinterpret(32);
      StringBuilder value = new StringBuilder();
      boolean terminated = false;
      for (int i = 0; i < 32; i++) {
        byte c = text.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i);
        if (c == 0) {
          terminated = true;
          break;
        }
        if (c < 32 || c > 126) throw new IOException("invalid libc version string");
        value.append((char) c);
      }
      if (!terminated) throw new IOException("unbounded libc version string");
      actual = value.toString();
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable e) {
      throw new IOException("libc version invocation", e);
    }
    if (NativeBridge.compareVersion(actual, minimumLibc) < 0)
      throw new IOException("glibc version below pack minimum");
  }

  /** Matches ABI feature requirements without loading code or invoking a restricted method. */
  public static boolean supportsRequiredFeatures(Set<String> required, long features) {
    if ((features & ~3L) != 0) return false;
    for (String feature : required) {
      long mask =
          switch (feature) {
            case "x86.avx2", "os.xsave-ymm" -> 1L;
            case "x86.fma" -> 2L;
            default -> 0L;
          };
      if (mask == 0 || (features & mask) == 0) return false;
    }
    return true;
  }

  @Override
  public long allocationSize(long bytes) {
    if (bytes < 0 || bytes > Integer.MAX_VALUE - 8)
      throw new IllegalArgumentException("allocation size");
    return Math.addExact(Math.max(1, bytes), 7) & ~7L;
  }

  @Override
  public NativeAllocator.Allocation allocate(long bytes) {
    return allocate(bytes, false);
  }

  @Override
  public NativeAllocator synchronousAllocator() {
    return new NativeAllocator() {
      @Override
      public long allocationSize(long bytes) {
        return FfmBridge.this.allocationSize(bytes);
      }

      @Override
      public Allocation allocate(long bytes) {
        return FfmBridge.this.allocate(bytes, true);
      }
    };
  }

  private NativeAllocator.Allocation allocate(long bytes, boolean confined) {
    long capacity = allocationSize(bytes);
    Arena arena = confined ? Arena.ofConfined() : Arena.ofShared();
    try {
      MemorySegment memory = arena.allocate(capacity, 8);
      return new NativeAllocator.Allocation() {
        private final AtomicBoolean closed = new AtomicBoolean();

        @Override
        public ByteBuffer buffer() {
          return memory.asByteBuffer();
        }

        @Override
        public long capacity() {
          return capacity;
        }

        @Override
        public boolean isAlive() {
          return memory.scope().isAlive();
        }

        @Override
        public void close() {
          synchronized (closed) {
            if (closed.get()) return;
            arena.close();
            closed.set(true);
          }
        }
      };
    } catch (RuntimeException | Error failure) {
      arena.close();
      throw failure;
    }
  }

  private static void verifyHost() throws IOException {
    if (Runtime.version().feature() < 25
        || ADDRESS.byteSize() != 8
        || ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN)
      throw new IOException("unqualified FFM platform");
    // Module's permission API is not in Lucene's Java21 base compile API.
    try {
      if (!(boolean)
          Module.class.getMethod("isNativeAccessEnabled").invoke(FfmBridge.class.getModule()))
        throw new IOException(
            "NATIVE_ACCESS_DISABLED: enable native access for "
                + (FfmBridge.class.getModule().isNamed()
                    ? FfmBridge.class.getModule().getName()
                    : "ALL-UNNAMED"));
    } catch (ReflectiveOperationException e) {
      throw new IOException("native access check", e);
    }
    if (INPUT.byteSize() != 32
        || OUTPUT.byteSize() != 24
        || CALL.byteSize() != 64
        || RESULT.byteSize() != 16
        || INPUT.byteAlignment() != 8
        || CALL.byteAlignment() != 8) throw new IOException("ABI layout mismatch");
  }

  @Override
  public AccelerationProvider.Prepared prepare(
      Path library, AccelerationProvider.Descriptor descriptor, MemoryLedger ledger)
      throws IOException {
    verifyHost();
    if (!library.isAbsolute()) throw new IOException("library path must be verified and absolute");
    // Arena.global is intentionally JVM lifetime; neither readers nor requests can unload code.
    SymbolLookup lookup = SymbolLookup.libraryLookup(library, Arena.global());
    Linker linker = Linker.nativeLinker();
    MethodHandle negotiate =
        linker.downcallHandle(
            lookup
                .find("ap_negotiate_v1")
                .orElseThrow(() -> new IOException("missing negotiate symbol")),
            FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG));
    boolean baseline =
        System.getProperty("lucene.acceleration.native.maximumIsa", "automatic").equals("baseline");
    MemorySegment executeSymbol =
        lookup
            .find(baseline ? "ap_execute_reference_v1" : "ap_execute_v1")
            .orElseThrow(() -> new IOException("missing execute symbol"));
    FunctionDescriptor executeDescriptor = FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS);
    MethodHandle execute = linker.downcallHandle(executeSymbol, executeDescriptor);
    Linker.Option[] criticalOption = criticalOption(false);
    Linker.Option[] heapOption = criticalOption(true);
    MemorySegment directSymbol =
        lookup
            .find(baseline ? "ap_execute_direct_reference_v1" : "ap_execute_direct_v1")
            .orElse(null);
    MemorySegment pinnedSymbol =
        lookup
            .find(baseline ? "ap_execute_pinned_reference_v1" : "ap_execute_pinned_v1")
            .orElse(null);
    FunctionDescriptor pinnedDescriptor =
        FunctionDescriptor.of(
            JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS);
    Handles handles =
        new Handles(
            execute,
            criticalOption == null
                ? execute
                : linker.downcallHandle(executeSymbol, executeDescriptor, criticalOption),
            directSymbol == null || heapOption == null
                ? null
                : linker.downcallHandle(
                    directSymbol,
                    FunctionDescriptor.of(
                        JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
                    heapOption),
            pinnedSymbol == null ? null : linker.downcallHandle(pinnedSymbol, pinnedDescriptor),
            pinnedSymbol == null || heapOption == null
                ? null
                : linker.downcallHandle(pinnedSymbol, pinnedDescriptor, heapOption));
    final long features;
    try (RequestMemoryScope scope =
        new RequestMemoryScope(
            ledger,
            new MemoryLedger.Owner("pack-startup", "standalone", 0, "pack-startup"),
            this)) {
      MemorySegment info = MemorySegment.ofBuffer(scope.allocate(32));
      info.fill((byte) 0);
      int status;
      try {
        status = (int) negotiate.invokeExact(1, 0, info, 32L);
      } catch (RuntimeException | Error e) {
        throw e;
      } catch (Throwable e) {
        throw new IOException("negotiate invocation", e);
      }
      int target = System.getProperty("os.arch").equals("aarch64") ? 2 : 1;
      features =
          NativeBridge.negotiatedFeatures(
              status,
              info.get(JAVA_INT, 0),
              info.get(JAVA_INT, 4),
              info.get(JAVA_INT, 8),
              info.get(JAVA_INT, 12),
              info.get(JAVA_LONG, 16),
              info.get(JAVA_LONG, 24),
              target);
    }
    EnumSet<Capability> available = EnumSet.noneOf(Capability.class);
    for (Capability capability : descriptor.capabilities()) {
      var claim = descriptor.kernels().get(capability);
      if (claim == null || supportsRequiredFeatures(claim.requiredFeatures(), features))
        available.add(capability);
    }
    Set<Capability> capabilities = Set.copyOf(available);
    return new AccelerationProvider.Prepared() {
      @Override
      public Set<Capability> capabilities() {
        return capabilities;
      }

      @Override
      public String isa() {
        return !baseline && (features & 1) != 0 ? "x86-avx2-os-xsave-ymm" : "baseline-scalar";
      }

      @Override
      public String isa(Capability capability) {
        boolean optimized =
            switch (capability) {
              case VECTOR_I8,
                  DOCSET_INTERSECT,
                  DOCSET_UNION,
                  DOCSET_DIFFERENCE,
                  DOCSET_CARDINALITY,
                  DOCSET_INTERSECT_COUNT,
                  MULTI_INTERSECT,
                  DOCVALUES_RANGE,
                  DOCVALUES_RANGE_FILTER,
                  DOCVALUES_DECODE,
                  DOCVALUES_AGGREGATE,
                  POSTINGS_PREFIX,
                  POSTINGS_BLOCK,
                  QUANTIZE_PACK,
                  QUANTIZE_UNPACK,
                  QUANTIZE_TRANSPOSE,
                  QUANTIZE_DIBIT,
                  QUANTIZE_BINARY,
                  QUANTIZED_RESCORE,
                  MODERN_QUANTIZED_RESCORE,
                  MASKED_FACETS,
                  BM25_BLOCK,
                  OSQ_GATHER ->
                  true;
              default -> false;
            };
        return optimized && !baseline && (features & 1) != 0
            ? "x86-avx2-os-xsave-ymm"
            : "baseline-scalar";
      }

      @Override
      public AccelerationProvider.KernelDescriptor kernelDescriptor(Capability capability) {
        var declared = descriptor.kernels().get(capability);
        return declared != null
            ? declared
            : AccelerationProvider.KernelDescriptor.reference(capability);
      }

      @Override
      public AccelerationProvider.Kernel bind(Capability capability) {
        if (!capabilities.contains(capability))
          throw new IllegalArgumentException("unregistered kernel");
        return new AccelerationProvider.Kernel() {
          private final CriticalCallBudget criticalBudget = new CriticalCallBudget();

          @Override
          public CriticalCallBudget.Snapshot criticalCalls() {
            return criticalBudget.snapshot();
          }

          /** The last batch's checked native view; racy reuse is safe because it is immutable. */
          private BatchView view;

          /** The previous batch, so a second presentation can build the cached view. */
          private AccelerationProvider.Batch seen;

          @Override
          public AccelerationProvider.Estimate estimate(AccelerationProvider.Shape shape) {
            boolean supported =
                shape.elements() <= 16L * 1024 * 1024 && shape.bytes() <= 128L * 1024 * 1024;
            if (capability == Capability.VECTOR_I8 || capability == Capability.QUANTIZED_RESCORE) {
              supported &=
                  shape.dimensions() > 0
                      && shape.dimensions() <= (capability == Capability.VECTOR_I8 ? 32768 : 16384)
                      && shape.elements() * shape.dimensions() <= 16L * 1024 * 1024;
            }
            if (capability == Capability.VECTOR_F32 || capability == Capability.MAXSIM)
              supported &=
                  shape.dimensions() > 0
                      && shape.dimensions() <= VectorUtil.exactScalarFloatLimit();
            if (capability == Capability.MULTI_INTERSECT)
              supported &= shape.dimensions() <= 16L * 1024 * 1024;
            long scratch =
                capability == Capability.OPTIMIZED_QUANTIZE
                    ? Math.multiplyExact(shape.elements(), 4)
                    : capability == Capability.COMPRESSION ? shape.dimensions() : 0;
            supported &= kernelDescriptor(capability).supports(shape);
            if (shape.layout() == AccelerationProvider.LayoutKind.HEAP_BUFFER)
              supported &=
                  handles.pinnedCritical != null
                      && !criticalBudget.disabled()
                      && capability != Capability.VECTOR_F32
                      && capability != Capability.MAXSIM
                      && (capability == Capability.OSQ_GATHER
                          || capability == Capability.BM25_BLOCK
                          || shape.bytes() <= CRITICAL_MAX_BYTES);
            return new AccelerationProvider.Estimate(
                supported, scratch, 0, Math.multiplyExact(shape.elements(), 8));
          }

          @Override
          public AccelerationProvider.Result execute(AccelerationProvider.Batch batch)
              throws IOException {
            if (capability != Capability.MASKED_FACETS) {
              for (int i = 0; i < batch.inputs().size(); i++)
                if (batch.inputs().get(i).span() != null)
                  return new AccelerationProvider.Result(
                      AccelerationProvider.Status.UNSUPPORTED, 0, 0);
            }
            if (batch.capability() != capability)
              throw new IllegalArgumentException("kernel key mismatch");
            if ((capability == Capability.VECTOR_F32 || capability == Capability.MAXSIM)
                && batch.parameters().duplicate().order(ByteOrder.LITTLE_ENDIAN).getLong(8)
                    > VectorUtil.exactScalarFloatLimit())
              return new AccelerationProvider.Result(AccelerationProvider.Status.UNSUPPORTED, 0, 0);
            BatchView current = view;
            if (current != null && current.matches(batch))
              return downcall(handles, current, criticalBudget);
            if (batch != seen) {
              // A batch seen once (e.g. a fresh mapped slice per call) keeps the allocation-light
              // path; only a batch presented again is worth a cached view.
              seen = batch;
              return downcallOnce(handles, batch, criticalBudget);
            }
            view = current = new BatchView(batch);
            return downcall(handles, current, criticalBudget);
          }
        };
      }
    };
  }

  private static MemorySegment nativeView(ByteBuffer buffer) {
    MemorySegment view = MemorySegment.ofBuffer(buffer);
    if (!view.isNative())
      throw new IllegalArgumentException("ordinary downcall requires native staging");
    return view;
  }

  private static MemorySegment inputView(AccelerationProvider.Input input) {
    if (input.span() == null) return dataView(input.bytes());
    Object storage = input.span().foreignSegment();
    if (!(storage instanceof MemorySegment segment)
        || segment.byteSize() != input.byteSize()
        || segment.isNative() != input.isNative()
        || !segment.isReadOnly())
      throw new IllegalArgumentException(
          "borrowed input must describe an exact read-only MemorySegment");
    if (!segment.scope().isAlive()) throw new IllegalStateException("borrowed input scope closed");
    return segment;
  }

  /** Inputs and output may be heap segments; only the direct critical call accepts those. */
  private static MemorySegment dataView(ByteBuffer buffer) {
    return MemorySegment.ofBuffer(buffer);
  }

  /**
   * Inputs a kernel reads only at a bounded number of gathered positions (one norm byte per doc) do
   * not count toward the critical-call size bound; the call's work follows the other inputs.
   */
  private static boolean gathered(Capability capability, int input) {
    return input == 1
        && (capability == Capability.BM25_BLOCK || capability == Capability.OSQ_GATHER);
  }

  private static long address(MemorySegment segment) {
    return segment.byteSize() == 0 || !segment.isNative() ? 0 : segment.address();
  }

  private static MemorySegment argument(MemorySegment segment) {
    return segment == null || segment.byteSize() == 0 ? MemorySegment.NULL : segment;
  }

  private static boolean mapped(MemorySegment segment) {
    return segment != null && segment.isMapped();
  }

  @Override
  public boolean isMapped(AccelerationProvider.Batch batch) {
    if (dataView(batch.output().bytes()).isMapped()
        || dataView(batch.parameters()).isMapped()
        || dataView(batch.scratch()).isMapped()
        || dataView(batch.metadata()).isMapped()) return true;
    for (var input : batch.inputs()) if (inputView(input).isMapped()) return true;
    return false;
  }

  private static boolean overlaps(MemorySegment a, MemorySegment b) {
    if (a.byteSize() == 0 || b.byteSize() == 0 || a.isNative() != b.isNative()) return false;
    if (!a.isNative()) return a.asOverlappingSlice(b).isPresent();
    long aEnd = Math.addExact(a.address(), a.byteSize());
    long bEnd = Math.addExact(b.address(), b.byteSize());
    return a.address() < bEnd && b.address() < aEnd;
  }

  /**
   * Checked native view of one batch, reused while the batch and every buffer's position and limit
   * are unchanged. Views, alias checks and the total size are computed once; every call still
   * rewrites the metadata fields the native side reads, through the liveness-checked metadata
   * segment, so a caller that modified the metadata cannot make native code read stale pointers.
   */
  private static final class BatchView {
    final AccelerationProvider.Batch batch;
    final ByteBuffer[] buffers;
    final int[] positions, limits;
    final MemorySegment metadata, result;
    final long parameters, scratch, scratchBytes, totalBytes;
    final MemorySegment parameterSegment, scratchSegment;
    final MemorySegment output;
    final MemorySegment[] inputs;
    final boolean heap;

    /** Last result, shared while native results repeat; records are immutable. */
    AccelerationProvider.Result last;

    BatchView(AccelerationProvider.Batch batch) {
      int inputCount = batch.inputs().size();
      if (inputCount > 3) throw new IllegalArgumentException("ABI input count");
      this.batch = batch;
      buffers = new ByteBuffer[4 + inputCount];
      buffers[0] = batch.metadata();
      buffers[1] = batch.parameters();
      buffers[2] = batch.scratch();
      buffers[3] = batch.output().bytes();
      for (int i = 0; i < inputCount; i++) buffers[4 + i] = batch.inputs().get(i).bytes();
      positions = new int[buffers.length];
      limits = new int[buffers.length];
      for (int i = 0; i < buffers.length; i++) {
        if (buffers[i] != null) {
          positions[i] = buffers[i].position();
          limits[i] = buffers[i].limit();
        }
      }
      metadata = nativeView(buffers[0]);
      MemorySegment parameterView = nativeView(buffers[1]);
      MemorySegment scratchView = nativeView(buffers[2]);
      MemorySegment outputView = dataView(buffers[3]);
      if (outputView.isReadOnly() || metadata.isReadOnly() || scratchView.isReadOnly())
        throw new IllegalArgumentException("writable staging required");
      if (overlaps(outputView, parameterView)
          || overlaps(outputView, scratchView)
          || overlaps(outputView, metadata)
          || overlaps(metadata, parameterView)
          || overlaps(metadata, scratchView))
        throw new IllegalArgumentException("ABI workspace alias");
      long total = outputView.byteSize() + scratchView.byteSize();
      boolean anyHeap = !outputView.isNative();
      inputs = new MemorySegment[inputCount];
      for (int i = 0; i < inputCount; i++) {
        MemorySegment input = inputView(batch.inputs().get(i));
        if (overlaps(outputView, input)
            || overlaps(metadata, input)
            || overlaps(scratchView, input))
          throw new IllegalArgumentException("ABI output/input alias");
        inputs[i] = input;
        anyHeap |= !input.isNative();
        if (!gathered(batch.capability(), i)) total += input.byteSize();
      }
      parameterSegment = parameterView;
      scratchSegment = scratchView;
      parameters = parameterView.address();
      scratchBytes = scratchView.byteSize();
      scratch = scratchBytes == 0 ? 0 : scratchView.address();
      output = outputView;
      totalBytes = total;
      heap = anyHeap;
      result = metadata.asSlice(184, 16);
    }

    boolean matches(AccelerationProvider.Batch candidate) {
      if (candidate != batch) return false;
      for (int i = 0; i < buffers.length; i++)
        if (buffers[i] != null
            && (buffers[i].position() != positions[i] || buffers[i].limit() != limits[i]))
          return false;
      for (int i = 0; i < inputs.length; i++) {
        var input = batch.inputs().get(i);
        if (input.span() != null
            && (input.span().foreignSegment() != inputs[i]
                || input.byteSize() != inputs[i].byteSize()
                || input.isNative() != inputs[i].isNative())) return false;
      }
      return true;
    }
  }

  private static AccelerationProvider.Result downcall(
      Handles handles, BatchView view, CriticalCallBudget criticalBudget) throws IOException {
    AccelerationProvider.Batch batch = view.batch;
    MemorySegment metadata = view.metadata;
    MemorySegment[] inputs = view.inputs;
    for (int i = 0; i < inputs.length; i++)
      writeInput(metadata, i, address(inputs[i]), inputs[i].byteSize(), batch.inputs().get(i));
    writeCall(
        metadata,
        batch,
        view.parameters,
        view.scratch,
        view.scratchBytes,
        address(view.output),
        view.output.byteSize());
    return invoke(
        handles,
        view.totalBytes,
        view.heap,
        metadata,
        view.result,
        inputs.length > 0 ? inputs[0] : null,
        inputs.length > 1 ? inputs[1] : null,
        inputs.length > 2 ? inputs[2] : null,
        view.output,
        view.parameterSegment,
        view.scratchSegment,
        batch.capability(),
        batch.output().elements(),
        view,
        criticalBudget);
  }

  /** One-time batches: views are built for this call only and nothing is retained. */
  private static AccelerationProvider.Result downcallOnce(
      Handles handles, AccelerationProvider.Batch batch, CriticalCallBudget criticalBudget)
      throws IOException {
    int inputCount = batch.inputs().size();
    if (inputCount > 3) throw new IllegalArgumentException("ABI input count");
    MemorySegment metadata = nativeView(batch.metadata());
    MemorySegment parameters = nativeView(batch.parameters());
    MemorySegment scratch = nativeView(batch.scratch());
    MemorySegment output = dataView(batch.output().bytes());
    if (output.isReadOnly() || metadata.isReadOnly() || scratch.isReadOnly())
      throw new IllegalArgumentException("writable staging required");
    if (overlaps(output, parameters)
        || overlaps(output, scratch)
        || overlaps(output, metadata)
        || overlaps(metadata, parameters)
        || overlaps(metadata, scratch)) throw new IllegalArgumentException("ABI workspace alias");
    long totalBytes = output.byteSize() + scratch.byteSize();
    boolean heap = !output.isNative();
    MemorySegment in0 = null, in1 = null, in2 = null;
    for (int i = 0; i < inputCount; i++) {
      var input = batch.inputs().get(i);
      MemorySegment view = inputView(input);
      if (overlaps(output, view) || overlaps(metadata, view) || overlaps(scratch, view))
        throw new IllegalArgumentException("ABI output/input alias");
      if (!gathered(batch.capability(), i)) totalBytes += view.byteSize();
      heap |= !view.isNative();
      writeInput(metadata, i, address(view), view.byteSize(), input);
      if (i == 0) in0 = view;
      else if (i == 1) in1 = view;
      else in2 = view;
    }
    writeCall(
        metadata,
        batch,
        parameters.address(),
        scratch.byteSize() == 0 ? 0 : scratch.address(),
        scratch.byteSize(),
        address(output),
        output.byteSize());
    return invoke(
        handles,
        totalBytes,
        heap,
        metadata,
        metadata.asSlice(184, 16),
        in0,
        in1,
        in2,
        output,
        parameters,
        scratch,
        batch.capability(),
        batch.output().elements(),
        null,
        criticalBudget);
  }

  // Call [0,64), input descriptors [64,160), output descriptor [160,184), result [184,200).
  // Pointers are written as 64-bit addresses (the native ABI requires 64-bit pointers); every
  // field the native side reads is written on every call, so no clearing pass is needed.
  private static void writeInput(
      MemorySegment metadata, int i, long address, long bytes, AccelerationProvider.Input input) {
    long at = 64 + i * 32L;
    metadata.set(JAVA_LONG, at, address);
    metadata.set(JAVA_LONG, at + 8, bytes);
    metadata.set(JAVA_LONG, at + 16, input.elements());
    metadata.set(JAVA_INT, at + 24, input.encoding().id);
    metadata.set(JAVA_INT, at + 28, 1);
  }

  private static void writeCall(
      MemorySegment metadata,
      AccelerationProvider.Batch batch,
      long parameters,
      long scratch,
      long scratchBytes,
      long output,
      long outputBytes) {
    long base = metadata.address();
    int inputCount = batch.inputs().size();
    metadata.set(JAVA_INT, 0, 64);
    metadata.set(JAVA_INT, 4, batch.capability().kernelId);
    metadata.set(JAVA_INT, 8, inputCount);
    metadata.set(JAVA_INT, 12, 1);
    metadata.set(JAVA_LONG, 16, inputCount == 0 ? 0 : base + 64);
    metadata.set(JAVA_LONG, 24, base + 160);
    metadata.set(JAVA_LONG, 32, parameters);
    metadata.set(JAVA_LONG, 40, 64);
    metadata.set(JAVA_LONG, 48, scratch);
    metadata.set(JAVA_LONG, 56, scratchBytes);
    metadata.set(JAVA_LONG, 160, output);
    metadata.set(JAVA_LONG, 168, outputBytes);
    metadata.set(JAVA_LONG, 176, batch.output().elements());
    metadata.set(JAVA_INT, 184, 16);
    metadata.set(JAVA_INT, 188, 0);
    metadata.set(JAVA_LONG, 192, Long.MIN_VALUE);
  }

  /**
   * The parameters, inputs, output and scratch are kept alive by the caller's scope for the whole
   * synchronous call; metadata (passed as the call argument) is held by the downcall. Small calls
   * take the direct critical entry when available, which also admits heap inputs and output; a heap
   * call that cannot take it declines. A cached view reuses its last result while the native result
   * repeats.
   */
  private static AccelerationProvider.Result invoke(
      Handles handles,
      long totalBytes,
      boolean heap,
      MemorySegment metadata,
      MemorySegment result,
      MemorySegment in0,
      MemorySegment in1,
      MemorySegment in2,
      MemorySegment output,
      MemorySegment parameters,
      MemorySegment scratch,
      Capability capability,
      long workElements,
      BatchView view,
      CriticalCallBudget criticalBudget)
      throws IOException {
    long workBytes =
        capability == Capability.OSQ_GATHER
            ? Math.addExact(
                totalBytes, Math.multiplyExact(workElements, parameters.get(JAVA_LONG, 8)))
            : totalBytes;
    // A short gather can still fault a cold mapped page; critical calls cannot block safepoints.
    boolean critical =
        criticalBudget.allows(capability, workBytes)
            && !mapped(metadata)
            && !mapped(result)
            && !mapped(in0)
            && !mapped(in1)
            && !mapped(in2)
            && !mapped(output)
            && !mapped(parameters)
            && !mapped(scratch);
    // Linker acquires every ADDRESS argument's scope. Nested descriptor pointers alone are
    // insufficient: a reader could close a mapped input while an ordinary downcall reads it.
    if (handles.pinned == null || heap && (!critical || handles.pinnedCritical == null))
      return new AccelerationProvider.Result(AccelerationProvider.Status.UNSUPPORTED, 0, 0);
    int status;
    boolean timed = critical && handles.pinnedCritical != null;
    long started = timed ? System.nanoTime() : 0;
    try {
      MethodHandle handle =
          critical && handles.pinnedCritical != null ? handles.pinnedCritical : handles.pinned;
      status =
          (int)
              handle.invokeExact(
                  metadata,
                  result,
                  argument(in0),
                  argument(in1),
                  argument(in2),
                  argument(output),
                  argument(parameters),
                  argument(scratch));
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable e) {
      throw new IOException("execute invocation", e);
    } finally {
      if (timed) criticalBudget.record(Math.max(0, System.nanoTime() - started));
    }
    if (status < 0 || status > 3 || result.get(JAVA_INT, 0) != 16)
      throw new IOException("unrecognized native status/metadata");
    int detail = result.get(JAVA_INT, 4);
    long elements = result.get(JAVA_LONG, 8);
    if (status == 1 && (detail != 0 || elements != Long.MIN_VALUE))
      throw new IOException("UNSUPPORTED modified result metadata");
    if (view == null) return new AccelerationProvider.Result(STATUSES[status], elements, detail);
    AccelerationProvider.Result last = view.last;
    if (last != null
        && last.status() == STATUSES[status]
        && last.outputElements() == elements
        && last.detail() == detail) return last;
    return view.last = new AccelerationProvider.Result(STATUSES[status], elements, detail);
  }
}
