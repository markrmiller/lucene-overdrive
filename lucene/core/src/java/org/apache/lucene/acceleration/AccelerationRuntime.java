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
package org.apache.lucene.acceleration;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceConfigurationError;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import org.apache.lucene.acceleration.memory.ManagedMemory;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.NativeAllocator;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ThreadInterruptedException;

/**
 * Frozen process registry. No pack discovery or library replacement occurs during a request.
 * Install/remove/upgrade requires a new JVM. Defaults keep the existing Java execution path.
 *
 * @lucene.experimental
 */
public final class AccelerationRuntime implements AutoCloseable {
  /** Exact source-bridge compatibility ID; any semantic bridge change requires a new ID. */
  public static final String BRIDGE_BUILD = "lucene-10.5.2-bridge-4";

  /** Startup outcome, separate from per-operation coverage. */
  public enum State {
    JAVA_ONLY,
    FROZEN
  }

  /** Bounded fallback reason labels. */
  public enum Reason {
    NO_PACK,
    NO_MEASURED_WIN,
    OFF,
    DISABLED,
    EXPERIMENTAL_DISABLED,
    SMALL_BATCH,
    UNSUPPORTED,
    BUDGET_DENIED,
    QUARANTINED,
    NATIVE_ERROR,
    UNECONOMICAL,
    SHUTDOWN
  }

  /** Snapshot of one exact capability. */
  public record Counter(
      long javaCalls,
      long nativeCalls,
      long items,
      long stagedBytes,
      long errors,
      boolean quarantined,
      Map<Reason, Long> fallbacks,
      long nativeBytes,
      long nativeNanos,
      long bridgeNanos,
      long scratchHighWaterBytes,
      Map<Integer, Long> batchSizes,
      long shadowSamples,
      long shadowBudgetDenials,
      long shadowMismatches) {}

  /** Bounded, privacy-safe mismatch record. No input bytes or document/query identifiers. */
  public record ShadowMismatch(String capability, String provider, long items, long outputBytes) {}

  /** Current owned scopes and separately admitted diagnostic execution. */
  public record ExecutionSnapshot(
      long activeScopes,
      long shadowBytes,
      long shadowHighWaterBytes,
      boolean stopping,
      List<ShadowMismatch> mismatches,
      List<String> blockers) {}

  /** Complete immutable startup identity and current bounded diagnostics. */
  public record Inventory(
      String bridge,
      State state,
      List<String> packs,
      Map<Capability, String> selected,
      List<String> rejections,
      boolean ready,
      Map<Capability, Counter> counters,
      MemoryLedger.Snapshot memory,
      ExecutionSnapshot execution) {}

  private static final class Entry {
    final AccelerationProvider.Kernel kernel;
    final String identity;
    final AtomicBoolean healthy = new AtomicBoolean(true);
    final AccelerationProvider.KernelDescriptor descriptor;
    final List<QualificationPolicy.Profile> profiles;
    final long minimumQualifiedRows;

    Entry(
        AccelerationProvider.Kernel kernel,
        String identity,
        AccelerationProvider.KernelDescriptor descriptor,
        List<QualificationPolicy.Profile> profiles) {
      this.kernel = kernel;
      this.identity = identity;
      this.descriptor = descriptor;
      this.profiles = profiles;
      this.minimumQualifiedRows =
          profiles.stream().mapToLong(QualificationPolicy.Profile::minimumRows).min().orElse(1);
    }
  }

  private static final class Metrics {
    final LongAdder java = new LongAdder(),
        errors = new LongAdder(),
        shadowSamples = new LongAdder(),
        shadowBudgetDenials = new LongAdder(),
        shadowMismatches = new LongAdder();

    /** Counters every native call updates, per thread rather than atomic. */
    final CallCounters calls = new CallCounters();

    final EnumMap<Reason, LongAdder> reasons = new EnumMap<>(Reason.class);

    Metrics() {
      for (Reason reason : Reason.values()) reasons.put(reason, new LongAdder());
    }
  }

  private final AccelerationConfig config;
  private final boolean qualificationOnly;
  private final boolean packPresent;
  private final MemoryLedger ledger;
  private final Map<Capability, Entry> entries;
  private final EnumMap<Capability, Metrics> metrics = new EnumMap<>(Capability.class);
  private final List<PackLoader.Loaded>
      loaded; // pins the verified parent-first loaders for JVM life
  private final List<String> rejections;
  private final AtomicLong scopes = new AtomicLong();
  private final ThreadLocal<TaskContext> tasks = new ThreadLocal<>();
  private final ThreadLocal<TaskBinding> bindings = new ThreadLocal<>();
  private final ThreadLocal<CallFrame> frames = new ThreadLocal<>();
  private final Set<CallFrame> liveFrames = ConcurrentHashMap.newKeySet();

  /** Live call frames are capped; past it, callers use per-call scopes. */
  static final int MAXIMUM_FRAMES = 256;

  private final Set<Capability> startupQuarantines = java.util.EnumSet.noneOf(Capability.class);

  private final AtomicBoolean ready = new AtomicBoolean(true);
  private final AtomicBoolean stopping = new AtomicBoolean();
  private final Object drainLock = new Object();

  private record ScopeRecord(long id, RequestMemoryScope scope, TaskContext context) {}

  private final ConcurrentHashMap<RequestMemoryScope, ScopeRecord> activeScopes =
      new ConcurrentHashMap<>();
  private final AtomicLong shadowBytes = new AtomicLong(), shadowHighWater = new AtomicLong();
  private final List<ShadowMismatch> shadowMismatches = new ArrayList<>();
  private volatile NativeBridge bridge;
  private final State state;
  private NativeCallJournal journal;

  /** Creates a startup registry. Applications must do this before opening readers. */
  public AccelerationRuntime(AccelerationConfig config) throws IOException {
    this(config, AccelerationProvider.class.getClassLoader());
  }

  /** Explicit startup loader; its API definition must be the host's one parent definition. */
  public AccelerationRuntime(AccelerationConfig config, ClassLoader startupLoader)
      throws IOException {
    this(config, startupLoader, false);
  }

  /**
   * Loads candidate kernels for explicit correctness and performance qualification. Candidates
   * remain inaccessible to application dispatch except through a measured owner-scoped boundary.
   * This runtime must not be installed as an application's production registry.
   */
  public static AccelerationRuntime forQualification(AccelerationConfig config) throws IOException {
    return forQualification(config, AccelerationProvider.class.getClassLoader());
  }

  /** Qualification with an explicit host API loader, subject to the normal verification checks. */
  public static AccelerationRuntime forQualification(
      AccelerationConfig config, ClassLoader startupLoader) throws IOException {
    return new AccelerationRuntime(config, startupLoader, true);
  }

  private AccelerationRuntime(
      AccelerationConfig config, ClassLoader startupLoader, boolean qualificationOnly)
      throws IOException {
    this(config, startupLoader, qualificationOnly, List.of());
  }

  private AccelerationRuntime(
      AccelerationConfig config,
      ClassLoader startupLoader,
      boolean qualificationOnly,
      List<String> initialDiagnostics)
      throws IOException {
    this.config = config;
    this.qualificationOnly = qualificationOnly;
    ledger =
        new MemoryLedger(
            new MemoryLedger.Limits(config.heapBytes(), config.nativeBytes(), config.mappedBytes()),
            config.requestBytes());
    if (config.memoryPolicy().ordinalBytes() > 0)
      ledger.setOrdinalLimit(config.memoryPolicy().ordinalBytes());
    for (Capability capability : Capability.values()) metrics.put(capability, new Metrics());
    List<String> diagnostics = new ArrayList<>(initialDiagnostics);
    ClasspathPacks discovered = ClasspathPacks.empty();
    if (config.mode() != AccelerationConfig.Mode.OFF
        && config.packPath() == null
        && config.experimental()
        && Runtime.version().feature() >= 25
        && config.nativeBytes() > 0
        && config.requestBytes() > 0) {
      try {
        discovered = ClasspathPacks.discover(startupLoader);
      } catch (IOException | IllegalArgumentException e) {
        diagnostic(diagnostics, "CLASSPATH_DISCOVER", e);
      }
    }
    ClasspathPacks classpath = discovered;
    try {
      packPresent = config.packPath() != null || !classpath.roots().isEmpty();
      List<java.nio.file.Path> directories = new ArrayList<>();
      QualificationPolicy qualification = QualificationPolicy.empty();
      boolean externalPolicy =
          System.getProperty("lucene.acceleration.native.qualificationPath") != null;
      if (!qualificationOnly
          && config.mode() != AccelerationConfig.Mode.OFF
          && packPresent
          && externalPolicy) {
        try {
          qualification = QualificationPolicy.load();
        } catch (IOException | IllegalArgumentException e) {
          diagnostic(diagnostics, "QUALIFICATION", e);
        }
      }
      for (var directory : classpath.roots()) {
        try {
          var manifest = PackLoader.verify(directory);
          var bundled = QualificationPolicy.read(directory.resolve("qualification.tsv"));
          if (bundled.profiles.stream().anyMatch(p -> !p.digest().equals(manifest.digest())))
            throw new IOException("bundled qualification belongs to a different pack");
          if (!qualificationOnly && !externalPolicy) qualification = qualification.append(bundled);
          directories.add(directory);
        } catch (IOException | IllegalArgumentException e) {
          diagnostic(diagnostics, "CLASSPATH_VERIFY", e);
        }
      }
      final QualificationPolicy policy = qualification;
      if (config.memoryPolicy().processEnvelopeBytes() > 0) {
        try {
          long known =
              Math.addExact(
                  Runtime.getRuntime().maxMemory(),
                  Math.addExact(config.nativeBytes(), config.memoryPolicy().safetyHeadroomBytes()));
          if (known > config.memoryPolicy().processEnvelopeBytes())
            diagnostics.add(
                "PROCESS_HEADROOM:configured envelope below heap/native/safety capacity");
        } catch (
            @SuppressWarnings("unused")
            ArithmeticException e) {
          diagnostics.add("PROCESS_HEADROOM:capacity sum overflows");
        }
      }
      List<PackLoader.Loaded> providers = new ArrayList<>();
      EnumMap<Capability, Entry> selected = new EnumMap<>(Capability.class);
      // OFF never touches the pack filesystem, providers, FFM implementation or restricted methods.
      if (config.mode() != AccelerationConfig.Mode.OFF && packPresent) {
        if (!config.experimental()) diagnostics.add("EXPERIMENTAL_DISABLED");
        else if (Runtime.version().feature() < 25) diagnostics.add("UNSUPPORTED_JDK");
        else if (config.nativeBytes() == 0 || config.requestBytes() == 0)
          diagnostics.add("NO_NATIVE_BUDGET");
        else {
          List<PackLoader.Verified> verified = new ArrayList<>();
          try {
            if (config.packPath() != null)
              directories.addAll(PackLoader.discover(config.packPath()));
            for (var directory : directories) {
              try {
                var manifest = PackLoader.verify(directory);
                if (qualificationOnly
                    || policy.profiles.stream().anyMatch(p -> p.digest().equals(manifest.digest())))
                  verified.add(manifest);
                else diagnostics.add("NO_MEASURED_WIN");
              } catch (IOException | IllegalArgumentException e) {
                diagnostic(diagnostics, "VERIFY", e);
              }
            }
          } catch (IOException e) {
            diagnostic(diagnostics, "DISCOVER", e);
          }
          if (!verified.isEmpty()) {
            try {
              journal = new NativeCallJournal();
            } catch (IOException | RuntimeException error) {
              diagnostic(diagnostics, "JOURNAL", error);
              verified.clear();
            }
          }
          if (!verified.isEmpty()) {
            try {
              bridge = createBridge();
            } catch (IOException e) {
              diagnostic(diagnostics, "BRIDGE", e);
            }
            for (var manifest : bridge == null ? List.<PackLoader.Verified>of() : verified) {
              String startupIdentity = BRIDGE_BUILD + ":" + manifest.digest() + ":pack-startup";
              if (journal.quarantined(NativeCallJournal.key(startupIdentity))) {
                startupQuarantines.addAll(manifest.capabilities());
                diagnostics.add("PERSISTENT_PACK_QUARANTINE:" + manifest.digest());
                continue;
              }
              int startupSlot = -1;
              try {
                startupSlot = journal.beginStartup(startupIdentity, manifest.digest());
                providers.add(PackLoader.instantiate(manifest, bridge, ledger, startupLoader));
              } catch (IOException
                  | ServiceConfigurationError
                  | LinkageError
                  | IllegalArgumentException e) {
                diagnostic(diagnostics, "PREPARE", e);
              } finally {
                if (startupSlot >= 0) journal.end(startupSlot);
              }
            }
          }
          // Resolve highest priority deterministically; equal-priority overlap stays Java.
          for (Capability capability : Capability.values()) {
            if (config.disabled().contains(capability)) continue;
            List<PackLoader.Loaded> candidates =
                providers.stream()
                    .filter(p -> p.provider().capabilities().contains(capability))
                    .filter(
                        p ->
                            qualificationOnly
                                || !policy
                                    .forKernel(
                                        p.manifest().digest(),
                                        p.provider().isa(capability),
                                        capability)
                                    .isEmpty())
                    .toList();
            String override = config.tuning().providerOverrides().get(capability);
            if (override != null)
              candidates =
                  candidates.stream().filter(p -> p.manifest().id().equals(override)).toList();
            int priority =
                candidates.stream()
                    .mapToInt(p -> p.manifest().priority())
                    .max()
                    .orElse(Integer.MIN_VALUE);
            candidates =
                candidates.stream().filter(p -> p.manifest().priority() == priority).toList();
            if (candidates.size() > 1) {
              diagnostics.add("AMBIGUOUS:" + capability.id);
              continue;
            }
            if (candidates.isEmpty()) continue;
            var provider = candidates.getFirst();
            String identity =
                provider.manifest().id()
                    + "@"
                    + provider.manifest().version()
                    + ":"
                    + provider.manifest().digest()
                    + ":"
                    + provider.provider().isa(capability)
                    + ":"
                    + capability.id
                    + ":"
                    + capability.arithmeticProfile;
            if (journal.quarantined(NativeCallJournal.key(identity))
                || journal.quarantined(
                    NativeCallJournal.key(identity.replace(":" + capability.id + ":", ":")))) {
              startupQuarantines.add(capability);
              diagnostics.add("PERSISTENT_QUARANTINE:" + capability.id);
              continue;
            }
            var kernel =
                journal.wrap(
                    provider.provider().bind(capability),
                    identity,
                    provider.manifest().digest(),
                    provider.provider().isa(capability),
                    capability);
            var description = provider.provider().kernelDescriptor(capability);
            if (!description.key().equals(AccelerationProvider.KernelKey.of(capability))) {
              diagnostics.add("DESCRIPTION:" + capability.id);
              continue;
            }
            try {
              if (!KernelFixtures.verify(capability, kernel, this::newScope)) {
                diagnostics.add("SELF_TEST:" + capability.id);
                continue;
              }
            } catch (IOException
                | IllegalArgumentException
                | RequestMemoryScope.AdmissionException e) {
              diagnostic(diagnostics, "SELF_TEST:" + capability.id, e);
              continue;
            }
            selected.put(
                capability,
                new Entry(
                    kernel,
                    identity,
                    description,
                    policy.forKernel(
                        provider.manifest().digest(),
                        provider.provider().isa(capability),
                        capability)));
          }
        }
      }
      if (!qualificationOnly
          && selected.isEmpty()
          && packPresent
          && config.mode() != AccelerationConfig.Mode.OFF
          && diagnostics.isEmpty()) diagnostics.add("NO_MEASURED_WIN");
      if (!selected.keySet().containsAll(config.required())) {
        if (journal != null) journal.close();
        throw new IOException(
            "required acceleration capabilities unavailable: "
                + config.required()
                + " "
                + diagnostics);
      }
      if (selected.isEmpty() && journal != null) {
        journal.close();
        journal = null;
      }
      entries = Map.copyOf(selected);
      loaded = List.copyOf(providers);
      cleanupClasspath(classpath, diagnostics);
      rejections = List.copyOf(diagnostics.subList(0, Math.min(64, diagnostics.size())));
      state = entries.isEmpty() ? State.JAVA_ONLY : State.FROZEN;
    } finally {
      cleanupClasspath(classpath, diagnostics);
    }
  }

  private static void cleanupClasspath(ClasspathPacks packs, List<String> diagnostics) {
    try {
      packs.close();
    } catch (IOException e) {
      // Provider snapshots already own their bytes. A failed temporary-file deletion cannot
      // discard a registry after process-lifetime native initialization has succeeded.
      diagnostic(diagnostics, "CLASSPATH_CLEANUP", e);
    }
  }

  public AccelerationConfig config() {
    return config;
  }

  /** Whether candidate kernels were loaded solely for explicit qualification. */
  public boolean isQualificationOnly() {
    return qualificationOnly;
  }

  /**
   * The verified candidate kernel for a prepared-buffer measurement. Bypasses dispatch admission
   * and runtime call metrics; the caller must own its buffers and validate and count its results.
   * Normal application runtimes cannot obtain a kernel through this entry point.
   */
  public AccelerationProvider.Kernel diagnosticKernel(Capability capability) throws IOException {
    checkpoint();
    if (!qualificationOnly)
      throw new IOException("candidate kernels require a qualification runtime");
    Entry entry = entries.get(capability);
    if (entry == null || !entry.healthy.get())
      throw new IOException("qualification kernel unavailable: " + capability.id);
    return entry.kernel;
  }

  private static void diagnostic(List<String> diagnostics, String stage, Throwable e) {
    if (diagnostics.size() >= 64) return;
    String message = String.valueOf(e.getMessage()).replace('\n', ' ').replace('\r', ' ');
    diagnostics.add(
        stage
            + ":"
            + e.getClass().getSimpleName()
            + ":"
            + message.substring(0, Math.min(160, message.length())));
  }

  private static NativeBridge createBridge() throws IOException {
    try {
      return (NativeBridge)
          Class.forName("org.apache.lucene.internal.acceleration.FfmBridge")
              .getConstructor()
              .newInstance();
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof Error error) throw error;
      throw new IOException("FFM bridge initialization", e.getCause());
    } catch (ReflectiveOperationException e) {
      throw new IOException("FFM bridge unavailable", e);
    }
  }

  private static final class Holder {
    static final AccelerationRuntime INSTANCE = initialize();

    private static AccelerationRuntime initialize() {
      return initializeSystem();
    }
  }

  // The implicit registry must not poison every reader/writer through a failed class initializer.
  // Explicit constructors and fromSystemProperties remain strict startup validation APIs.
  static AccelerationRuntime initializeSystem() {
    try {
      return new AccelerationRuntime(AccelerationConfig.fromSystemProperties());
    } catch (IOException | RuntimeException | LinkageError | ServiceConfigurationError error) {
      try {
        var fallback =
            new AccelerationConfig(
                AccelerationConfig.Mode.OFF,
                null,
                Set.of(),
                Set.of(),
                0,
                0,
                64,
                false,
                AccelerationConfig.StoragePolicy.CURRENT);
        AccelerationRuntime disabled =
            new AccelerationRuntime(
                fallback,
                AccelerationProvider.class.getClassLoader(),
                false,
                List.of("STARTUP_DISABLED:" + error.getClass().getSimpleName()));
        disabled.ready.set(false);
        return disabled;
      } catch (IOException impossible) {
        throw new AssertionError(impossible);
      }
    }
  }

  /** Initializes or returns the single process registry; configuration cannot be hot-reloaded. */
  public static AccelerationRuntime system() {
    return Holder.INSTANCE;
  }

  private static final class Configured {
    static final AccelerationRuntime INSTANCE = packConfigured() ? system() : null;
  }

  /**
   * Returns the process registry when a pack may be configured, or null without initializing it.
   * Stock insert points resolve this once, so with no pack they pay one constant branch.
   */
  public static AccelerationRuntime configured() {
    return Configured.INSTANCE;
  }

  /**
   * True when the startup properties, or a classpath pack resource on JDK 25+, may register
   * kernels. Reads only system properties and one class-loader resource; never initializes the
   * registry.
   */
  public static boolean packConfigured() {
    String prefix = "lucene.acceleration.";
    if (Runtime.version().feature() < 25
        || "off".equals(System.getProperty(prefix + "native.mode"))
        || "false".equals(System.getProperty(prefix + "experimental.enabled"))) return false;
    if (!System.getProperty(prefix + "native.packPath", "").isEmpty()) return true;
    ClassLoader loader = AccelerationProvider.class.getClassLoader();
    return loader != null && loader.getResource(ClasspathPacks.RESOURCE) != null;
  }

  /** Returns the independent storage policy without changing compute selection. */
  public AccelerationConfig.StoragePolicy storagePolicy() {
    return config.storagePolicy();
  }

  /** Managed capacity, including explicitly enrolled cache, ordinal and mapping owners. */
  public MemoryLedger ledger() {
    return ledger;
  }

  /** Frozen registration; this does not count a call or perform shape admission. */
  public boolean supports(Capability capability) {
    return entries.containsKey(capability);
  }

  /**
   * Cheap consumer-context preflight before optional materialization. False means keep the Java
   * path; true does not admit any shape, parameter, memory allocation or native call. Qualification
   * candidates need no deployment profile; production requires an exact matching profile context.
   */
  public boolean supports(Capability capability, String consumerContext) {
    if (!QualificationPolicy.validConsumerContext(consumerContext) || stopping.get()) return false;
    Entry entry = entries.get(capability);
    if (entry == null || !entry.healthy.get()) return false;
    if (qualificationOnly) return true;
    for (var profile : entry.profiles) {
      if (profile.context(consumerContext)) return true;
    }
    return false;
  }

  /**
   * Exact, side-effect-free preflight before optional input materialization. The shape describes
   * the proposed complete native batch, including all inputs, output and workspace. Parameter slots
   * 3 and 4 and the consumer context must match the same deployment profile. This neither reserves
   * storage nor grants execution: the completed batch is independently checked again.
   */
  public boolean supports(
      Capability capability,
      AccelerationProvider.Shape shape,
      long parameter3,
      long parameter4,
      String consumerContext) {
    java.util.Objects.requireNonNull(shape);
    boolean mapped = shape.layout() == AccelerationProvider.LayoutKind.MAPPED_BUFFER;
    // Policy describes mapped provenance; descriptors describe the physical direct buffer.
    if (mapped)
      shape =
          new AccelerationProvider.Shape(
              shape.elements(),
              shape.dimensions(),
              shape.bytes(),
              shape.encodings(),
              AccelerationProvider.LayoutKind.NATIVE_BUFFER,
              shape.alignment(),
              shape.nativeAddressable());
    return supports(capability, shape, parameter3, parameter4, consumerContext, mapped);
  }

  // Descriptor layouts describe physical buffers; mapped provenance is checked separately.
  boolean supports(
      Capability capability,
      AccelerationProvider.Shape shape,
      long parameter3,
      long parameter4,
      String consumerContext,
      boolean mapped) {
    java.util.Objects.requireNonNull(capability);
    java.util.Objects.requireNonNull(shape);
    if (!QualificationPolicy.validConsumerContext(consumerContext)
        || eligibilityReason(capability, shape.elements()) != null) return false;
    Entry entry = entries.get(capability);
    if (entry == null || !entry.descriptor.supports(shape)) return false;
    if (!qualificationOnly) {
      boolean matching = false;
      for (var profile : entry.profiles) {
        if (profile.shape(shape, mapped)
            && profile.context(consumerContext)
            && profile.parameters(parameter3, parameter4)) {
          matching = true;
          break;
        }
      }
      if (!matching) return false;
    }
    return entry.kernel.estimate(shape).supported();
  }

  /**
   * Caller-owned request identity and cancellation signal. Install the same context in each worker
   * to enforce one combined request limit across concurrent operation scopes. Cancellation signals
   * work; it never closes memory underneath an executing downcall.
   */
  public static final class TaskContext {
    private final MemoryLedger.Owner owner;
    private final long deadlineNanos;
    private final RequestMemoryScope.AdmissionPolicy admissionPolicy;
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final ConcurrentHashMap<RequestMemoryScope, Boolean> scopes = new ConcurrentHashMap<>();

    public TaskContext(MemoryLedger.Owner owner, long deadlineNanos) {
      this(owner, deadlineNanos, null);
    }

    /**
     * Host-owned admission applies to internal heap and native staging as well as explicit scopes.
     */
    public TaskContext(
        MemoryLedger.Owner owner,
        long deadlineNanos,
        RequestMemoryScope.AdmissionPolicy admissionPolicy) {
      this.owner = java.util.Objects.requireNonNull(owner);
      this.deadlineNanos = deadlineNanos;
      this.admissionPolicy = admissionPolicy;
    }

    /** Stops subsequent native work and publication after current synchronous calls settle. */
    public void cancel() {
      cancelled.set(true);
      for (RequestMemoryScope scope : scopes.keySet()) scope.cancel();
    }
  }

  /** A thread-confined context binding; each worker installs and closes its own binding. */
  public final class TaskBinding implements AutoCloseable {
    private final Thread thread = Thread.currentThread();
    private final TaskContext previous;
    private final TaskBinding previousBinding;
    private boolean closed;

    private TaskBinding(TaskContext context) {
      previous = tasks.get();
      previousBinding = bindings.get();
      tasks.set(java.util.Objects.requireNonNull(context));
      bindings.set(this);
    }

    /** Whether this thread can close this binding without crossing a caller-owned scope. */
    public boolean isCurrent() {
      return Thread.currentThread() == thread && !closed && bindings.get() == this;
    }

    @Override
    public void close() {
      if (closed) return;
      if (Thread.currentThread() != thread || bindings.get() != this)
        throw new IllegalStateException("task binding must close on its owner in reverse order");
      if (previous == null) tasks.remove();
      else tasks.set(previous);
      if (previousBinding == null) bindings.remove();
      else bindings.set(previousBinding);
      closed = true;
    }
  }

  /** Installs explicit host identity; no thread pool or implicit request lifetime is created. */
  public TaskBinding enterTask(TaskContext context) {
    if (config.memoryPolicy().coreBurstBytes() > 0)
      ledger.setCoreLimit(context.owner.coreId(), config.memoryPolicy().coreBurstBytes());
    return new TaskBinding(context);
  }

  /** Current caller-owned context, including its cancellation and deadline; null if unbound. */
  public TaskContext currentTaskContext() {
    return tasks.get();
  }

  /** Bound request identity, or null when no host task has been installed. */
  public MemoryLedger.Owner currentOwner() {
    TaskContext task = tasks.get();
    return task == null ? null : task.owner;
  }

  /** Enrolls Java staging before allocation, sharing the bound request's combined quota. */
  public MemoryLedger.Reservation reserveHeap(long bytes) throws IOException {
    checkpoint();
    if (stopping.get()) throw new IOException("acceleration runtime shutting down");
    MemoryLedger.Owner owner = currentOwner();
    if (owner == null)
      owner = ManagedMemory.owner("acceleration", MemoryLedger.Purpose.REQUEST_SCRATCH);
    if (config.memoryPolicy().coreBurstBytes() > 0)
      ledger.setCoreLimit(owner.coreId(), config.memoryPolicy().coreBurstBytes());
    TaskContext task = tasks.get();
    if (task == null || task.admissionPolicy == null)
      return ManagedMemory.reserve(
          ledger, owner, MemoryLedger.Kind.HEAP, bytes, MemoryLedger.Purpose.REQUEST_SCRATCH);
    if (!ledger.isEnrolled(MemoryLedger.Kind.HEAP)) return null;
    MemoryLedger.Admission admission =
        task.admissionPolicy.reserve(
            owner,
            MemoryLedger.Kind.HEAP,
            bytes,
            MemoryLedger.Purpose.REQUEST_SCRATCH,
            task.cancelled::get);
    if (!admission.granted())
      throw new RequestMemoryScope.AdmissionException(
          RequestMemoryScope.Failure.valueOf(admission.reason().name()));
    try {
      checkpoint();
    } catch (IOException cancelled) {
      admission.reservation().close();
      throw cancelled;
    }
    return admission.reservation();
  }

  /** Identifies only an expired bound task deadline while preserving IOException compatibility. */
  public static final class TaskDeadlineExceededException extends InterruptedIOException {
    private static final long serialVersionUID = 1L;

    private TaskDeadlineExceededException() {
      super("acceleration task deadline expired");
    }
  }

  /**
   * Checks interruption, shared request cancellation and the caller's monotonic deadline.
   *
   * <p>Without a bound task the caller is ordinary Lucene work, such as a merge or flush, so an
   * interrupt there consumes the flag and throws {@link ThreadInterruptedException} like the rest
   * of Lucene.
   */
  public void checkpoint() throws InterruptedIOException {
    TaskContext context = tasks.get();
    if (stopping.get())
      throw new InterruptedIOException("acceleration task cancelled or deadline expired");
    if (context == null) {
      if (Thread.interrupted()) throw new ThreadInterruptedException(new InterruptedException());
      return;
    }
    if (Thread.currentThread().isInterrupted() || context.cancelled.get())
      throw new InterruptedIOException("acceleration task cancelled or deadline expired");
    if (context.deadlineNanos != 0 && System.nanoTime() - context.deadlineNanos >= 0)
      throw new TaskDeadlineExceededException();
  }

  /**
   * Whether generic batch candidates are registered; measured boundaries use explicit ownership.
   */
  public boolean hasKernels() {
    return qualificationOnly && entries.keySet().stream().anyMatch(c -> c != Capability.TEST_COPY);
  }

  /** Selects before staging and counts stable Java reasons. */
  public boolean eligible(Capability capability, long items) {
    return eligible(capability, items, false);
  }

  private boolean eligible(Capability capability, long items, boolean measuredBoundary) {
    Reason reason = eligibilityReason(capability, items);
    if (reason == Reason.QUARANTINED && config.required().contains(capability))
      throw new IllegalStateException("required acceleration kernel quarantined: " + capability.id);
    if (reason != null) {
      recordJava(capability, reason);
      return false;
    }
    return true;
  }

  private Reason eligibilityReason(Capability capability, long items) {
    return stopping.get()
        ? Reason.SHUTDOWN
        : config.mode() == AccelerationConfig.Mode.OFF
            ? Reason.OFF
            : config.disabled().contains(capability)
                ? Reason.DISABLED
                : !config.experimental() && packPresent
                    ? Reason.EXPERIMENTAL_DISABLED
                    : startupQuarantines.contains(capability)
                        ? Reason.QUARANTINED
                        : !entries.containsKey(capability)
                            ? (!qualificationOnly && packPresent
                                ? Reason.NO_MEASURED_WIN
                                : Reason.NO_PACK)
                            : !entries.get(capability).healthy.get()
                                ? Reason.QUARANTINED
                                : items < entries.get(capability).minimumQualifiedRows
                                    ? Reason.SMALL_BATCH
                                    : scalarScoring(capability, items)
                                        ? Reason.SMALL_BATCH
                                        : items
                                                < config
                                                    .tuning()
                                                    .minimumBatches()
                                                    .getOrDefault(capability, config.minimumBatch())
                                            ? Reason.SMALL_BATCH
                                            : null;
  }

  private static boolean scalarScoring(Capability capability, long items) {
    return items <= 1
        && (capability == Capability.VECTOR_I8
            || capability == Capability.VECTOR_F32
            || capability == Capability.QUANTIZED_RESCORE
            || capability == Capability.MODERN_QUANTIZED_RESCORE);
  }

  /** Records Java selection outside a kernel path, including unchanged graph construction. */
  public void recordJava(Capability capability, Reason reason) {
    Metrics m = metrics.get(capability);
    m.java.increment();
    m.reasons.get(reason).increment();
  }

  /**
   * Admitted per-task scratch; applications may adapt the same contracts for their request tree.
   */
  public RequestMemoryScope newScope() throws IOException {
    checkpoint();
    return newScope(nativeAllocator(), false);
  }

  /**
   * Same-thread staging for a synchronous kernel call and its publication. Its storage is
   * thread-confined, so closing it avoids the VM-wide handshake that closing shared memory costs
   * (about 250 microseconds per call when measured); per-batch callers should prefer it to {@link
   * #newScope()}.
   */
  public RequestMemoryScope newCallScope() throws IOException {
    checkpoint();
    NativeBridge allocator = (NativeBridge) nativeAllocator();
    return newScope(allocator.synchronousAllocator(), true);
  }

  /**
   * This thread's reusable call frame, marked busy until closed, or null when a per-call scope must
   * be used instead: the frame is already in use on this thread, the runtime is stopping, the frame
   * cap is reached or the ledger denies the frame. The frame is admitted once, outside any task, so
   * cancelling a task never cancels it; callers still checkpoint their own task.
   */
  public CallFrame acquireFrame() throws IOException {
    checkpoint();
    if (stopping.get()) return null;
    CallFrame frame = frames.get();
    if (frame == null || frame.isRetired()) {
      frame = newFrame();
      if (frame == null) return null;
      frames.set(frame);
    }
    if (!frame.acquire()) return null;
    if (stopping.get()) {
      frame.close();
      return null;
    }
    return frame;
  }

  private CallFrame newFrame() throws IOException {
    for (CallFrame frame : liveFrames) if (!frame.thread().isAlive()) frame.retire();
    if (liveFrames.size() >= MAXIMUM_FRAMES) return null;
    Thread thread = Thread.currentThread();
    var owner =
        new MemoryLedger.Owner(
            "acceleration", "call-frame", 0, Long.toUnsignedString(thread.threadId()));
    RequestMemoryScope scope;
    try {
      scope = registerScope(new RequestMemoryScope(ledger, owner, nativeAllocator()), null);
    } catch (
        @SuppressWarnings("unused")
        IOException stopped) {
      return null;
    }
    try {
      CallFrame frame = new CallFrame(this, scope, thread);
      liveFrames.add(frame);
      if (stopping.get()) {
        frame.retire();
        return null;
      }
      return frame;
    } catch (
        @SuppressWarnings("unused")
        RequestMemoryScope.AdmissionException denied) {
      scope.close();
      return null;
    } catch (RuntimeException | Error failure) {
      scope.close();
      throw failure;
    }
  }

  boolean isStopping() {
    return stopping.get();
  }

  void forgetFrame(CallFrame frame) {
    liveFrames.remove(frame);
  }

  /** Owner-thread scratch lifetime for prepared candidate measurements, closed by its creator. */
  public RequestMemoryScope newQualificationScope() throws IOException {
    if (!qualificationOnly)
      throw new IOException("candidate scratch requires a qualification runtime");
    return newCallScope();
  }

  private RequestMemoryScope newScope(NativeAllocator allocator, boolean confined)
      throws IOException {
    TaskContext task = tasks.get();
    MemoryLedger.Owner owner =
        task == null
            ? new MemoryLedger.Owner(
                "acceleration", "standalone", 0, Long.toUnsignedString(scopes.incrementAndGet()))
            : task.owner;
    if (config.memoryPolicy().coreBurstBytes() > 0)
      ledger.setCoreLimit(owner.coreId(), config.memoryPolicy().coreBurstBytes());
    return registerScope(
        confined
            ? RequestMemoryScope.threadConfined(
                ledger, owner, allocator, task == null ? null : task.admissionPolicy)
            : task == null || task.admissionPolicy == null
                ? new RequestMemoryScope(ledger, owner, allocator)
                : new RequestMemoryScope(ledger, owner, allocator, task.admissionPolicy),
        task);
  }

  /** Physically independent chunk storage, with the same request owner and cancellation tree. */
  public RequestMemoryScope newScope(RequestMemoryScope parent) throws IOException {
    checkpoint();
    if (parent.ledger() != ledger) throw new IOException("parent scope belongs to another ledger");
    ScopeRecord record = activeScopes.get(parent);
    return registerScope(parent.childScope(), record == null ? tasks.get() : record.context());
  }

  private RequestMemoryScope registerScope(RequestMemoryScope scope, TaskContext task)
      throws IOException {
    synchronized (drainLock) {
      if (stopping.get()) {
        scope.close();
        throw new IOException("acceleration runtime shutting down");
      }
      activeScopes.put(scope, new ScopeRecord(scopes.incrementAndGet(), scope, task));
      if (task != null) task.scopes.put(scope, Boolean.TRUE);
      scope.onDrained(
          () -> {
            activeScopes.remove(scope);
            if (task != null) task.scopes.remove(scope);
            synchronized (drainLock) {
              drainLock.notifyAll();
            }
          });
      if (task != null && task.cancelled.get()) scope.cancel();
    }
    return scope;
  }

  /** Java-owned storage allocator; obtaining it does not grant native compute access. */
  public NativeAllocator nativeAllocator() throws IOException {
    checkpoint();
    if (stopping.get()) throw new IOException("acceleration runtime shutting down");
    NativeBridge allocator = bridge;
    if (allocator == null) {
      if (Runtime.version().feature() < 25) throw new IOException("off-heap scope requires JDK25");
      synchronized (this) {
        if (bridge == null) bridge = createBridge();
        allocator = bridge;
      }
    }
    return allocator;
  }

  /** Pure full-shape admission, including a provider's bounded workspace and measured cost hint. */
  public AccelerationProvider.Estimate admit(
      Capability capability, AccelerationProvider.Shape shape, long javaNanos) {
    return admit(
        capability,
        shape,
        javaNanos,
        shape.layout() == AccelerationProvider.LayoutKind.MAPPED_BUFFER);
  }

  private AccelerationProvider.Estimate admit(
      Capability capability,
      AccelerationProvider.Shape shape,
      long javaNanos,
      boolean measuredBoundary) {
    Entry entry = entries.get(capability);
    if (!eligible(capability, shape.elements(), measuredBoundary)
        || entry == null
        || !entry.descriptor.supports(shape)
        || !qualificationOnly && !matchesProfile(entry, shape, measuredBoundary, null, null))
      return new AccelerationProvider.Estimate(false, 0, 0);
    var estimate = entry.kernel.estimate(shape);
    long scratch = Math.max(estimate.scratchBytes(), entry.descriptor.scratch().bytes(shape));
    long staging = Math.max(estimate.stagingBytes(), entry.descriptor.staging().bytes(shape));
    long cost =
        Math.addExact(
            estimate.estimatedNanos(),
            Math.multiplyExact(config.tuning().stagingNanosPerByte(), shape.bytes()));
    if (javaNanos > 0 && Math.addExact(cost, config.tuning().minimumBenefitNanos()) >= javaNanos) {
      recordJava(capability, Reason.UNECONOMICAL);
      return new AccelerationProvider.Estimate(false, scratch, staging, cost);
    }
    return new AccelerationProvider.Estimate(estimate.supported(), scratch, staging, cost);
  }

  public AccelerationProvider.Estimate admit(
      Capability capability, AccelerationProvider.Shape shape) {
    return admit(capability, shape, 0);
  }

  /** Calls one frozen kernel. False means discard staging and safely execute the Java oracle. */
  public boolean execute(AccelerationProvider.Batch batch, long items, long stagedBytes)
      throws IOException {
    return execute(batch, items, stagedBytes, null);
  }

  /** Executes with an explicit owner for independently budgeted secondary comparison frames. */
  public boolean execute(
      AccelerationProvider.Batch batch, long items, long stagedBytes, RequestMemoryScope ownerScope)
      throws IOException {
    return execute(batch, items, stagedBytes, ownerScope, false, "generic");
  }

  /**
   * Executes for an explicit reviewed consumer context. Legacy APIs use {@code generic}; a tagged
   * qualification profile authorizes only a matching context. The context is Java dispatch data and
   * is never written to provider parameters or native inputs.
   */
  public boolean execute(
      AccelerationProvider.Batch batch,
      long items,
      long stagedBytes,
      RequestMemoryScope ownerScope,
      String consumerContext)
      throws IOException {
    if (!QualificationPolicy.validConsumerContext(consumerContext))
      throw new IllegalArgumentException("invalid acceleration consumer context");
    return execute(batch, items, stagedBytes, ownerScope, false, consumerContext);
  }

  AccelerationProvider.Kernel measuredByteKernel(
      int dimensions, int rows, VectorSimilarityFunction similarity) {
    return measuredByteKernel(dimensions, rows, similarity, "generic");
  }

  AccelerationProvider.Kernel measuredByteKernel(
      int dimensions, int rows, VectorSimilarityFunction similarity, String consumerContext) {
    Entry entry = entries.get(Capability.VECTOR_I8);
    if (!QualificationPolicy.validConsumerContext(consumerContext)
        || qualificationOnly
        || stopping.get()
        || entry == null
        || !entry.healthy.get()
        || rows
            < config
                .tuning()
                .minimumBatches()
                .getOrDefault(Capability.VECTOR_I8, config.minimumBatch())
        || !MeasuredByteProfile.shape(dimensions, rows, similarity)) return null;
    for (int i = 0; i < entry.profiles.size(); i++) {
      var profile = entry.profiles.get(i);
      if (profile.dimensions() == dimensions
          && profile.minimumRows() <= rows
          && profile.maximumRows() >= rows
          && profile.layout().equals("mapped")
          && profile.parameter3() != null
          && profile.parameters(similarity.ordinal(), 0)
          && profile.context(consumerContext)) return entry.kernel;
    }
    return null;
  }

  int measuredByteCapacity(int dimensions, VectorSimilarityFunction similarity) {
    return measuredByteCapacity(dimensions, similarity, "generic");
  }

  int measuredByteCapacity(
      int dimensions, VectorSimilarityFunction similarity, String consumerContext) {
    Entry entry = entries.get(Capability.VECTOR_I8);
    if (!QualificationPolicy.validConsumerContext(consumerContext)
        || qualificationOnly
        || stopping.get()
        || entry == null
        || !entry.healthy.get()) return 0;
    int maximum = 0;
    for (int i = 0; i < entry.profiles.size(); i++) {
      var profile = entry.profiles.get(i);
      if (profile.dimensions() == dimensions
          && profile.layout().equals("mapped")
          && profile.parameter3() != null
          && profile.parameters(similarity.ordinal(), 0)
          && profile.context(consumerContext))
        maximum =
            Math.max(
                maximum, (int) Math.min(MeasuredByteProfile.MAXIMUM_ROWS, profile.maximumRows()));
    }
    return maximum;
  }

  private static boolean matchesProfile(
      Entry entry,
      AccelerationProvider.Shape shape,
      boolean mapped,
      AccelerationProvider.Batch batch,
      String consumerContext) {
    for (int i = 0; i < entry.profiles.size(); i++) {
      var profile = entry.profiles.get(i);
      if (profile.shape(shape, mapped)
          && (consumerContext == null || profile.context(consumerContext))
          && (batch == null || profile.batch(batch))) return true;
    }
    return false;
  }

  boolean executeMappedByte(AccelerationProvider.Batch batch, RequestMemoryScope owner)
      throws IOException {
    return executeMappedByte(batch, owner, "generic");
  }

  boolean executeMappedByte(
      AccelerationProvider.Batch batch, RequestMemoryScope owner, String consumerContext)
      throws IOException {
    var p = batch.parameters().duplicate().order(ByteOrder.LITTLE_ENDIAN);
    int count = Math.toIntExact(p.getLong(0));
    if (batch.capability() != Capability.VECTOR_I8
        || p.getLong(8) != p.getLong(16)
        || p.getLong(24) != VectorSimilarityFunction.DOT_PRODUCT.ordinal()
        || !MeasuredByteProfile.shape(
            Math.toIntExact(p.getLong(8)), count, VectorSimilarityFunction.DOT_PRODUCT)
        || batch.inputs().size() != 2
        || !batch.inputs().get(1).bytes().isDirect()
        || owner.ledger() != ledger) throw new IOException("unmeasured mapped byte shape");
    return execute(batch, count, 0, owner, true, consumerContext);
  }

  private boolean execute(
      AccelerationProvider.Batch batch,
      long items,
      long stagedBytes,
      RequestMemoryScope ownerScope,
      boolean measuredBoundary,
      String consumerContext)
      throws IOException {
    long bridgeStart = System.nanoTime();
    checkpoint();
    Entry entry = entries.get(batch.capability());
    if (entry == null || !entry.healthy.get()) {
      recordJava(batch.capability(), Reason.QUARANTINED);
      if (config.required().contains(batch.capability()))
        throw new IOException("required kernel unavailable");
      return false;
    }
    Metrics m = metrics.get(batch.capability());
    long inputBytes = 0;
    boolean nativeAddressable = batch.output().bytes().isDirect();
    int alignment = bufferAlignment(batch.output().bytes());
    for (var input : batch.inputs()) {
      if (input.span() != null && batch.capability() != Capability.MASKED_FACETS) {
        recordJava(batch.capability(), Reason.UNSUPPORTED);
        return false;
      }
      inputBytes = Math.addExact(inputBytes, input.byteSize());
      nativeAddressable &= input.isNative();
      alignment = Math.min(alignment, input.bytes() == null ? 1 : bufferAlignment(input.bytes()));
    }
    boolean mapped = measuredBoundary || bridge != null && bridge.isMapped(batch);
    Set<AccelerationProvider.Encoding> encodings = encodings(batch.inputs());
    var shape =
        new AccelerationProvider.Shape(
            items,
            Math.toIntExact(
                batch.parameters().duplicate().order(ByteOrder.LITTLE_ENDIAN).getLong(8)),
            Math.addExact(
                inputBytes,
                Math.addExact(
                    batch.output().bytes().remaining(),
                    Math.addExact(
                        batch.scratch().remaining(),
                        Math.addExact(
                            batch.parameters().remaining(), batch.metadata().remaining())))),
            encodings,
            nativeAddressable
                ? AccelerationProvider.LayoutKind.NATIVE_BUFFER
                : AccelerationProvider.LayoutKind.HEAP_BUFFER,
            alignment,
            nativeAddressable);
    var estimate = admit(batch.capability(), shape, 0, mapped);
    if (!estimate.supported()
        || estimate.scratchBytes() > batch.scratch().remaining()
        || estimate.stagingBytes() > stagedBytes) {
      recordJava(batch.capability(), Reason.UNSUPPORTED);
      return false;
    }
    if (!qualificationOnly && !matchesProfile(entry, shape, mapped, batch, consumerContext)) {
      recordJava(batch.capability(), Reason.NO_MEASURED_WIN);
      return false;
    }
    JavaKernelOracle.validate(batch);
    if (!JavaKernelOracle.supportsFloatDomain(batch)) {
      recordJava(batch.capability(), Reason.UNSUPPORTED);
      return false;
    }
    long[] cell = m.calls.cell();
    CallCounters.max(cell, CallCounters.SCRATCH_HIGH_WATER, batch.scratch().remaining());
    AccelerationProvider.Result result;
    checkpoint();
    long nativeStart = System.nanoTime();
    CallCounters.add(cell, CallCounters.CALLS, 1);
    CallCounters.add(cell, CallCounters.ITEMS, items);
    CallCounters.add(cell, CallCounters.STAGED, stagedBytes);
    CallCounters.add(cell, CallCounters.BYTES, inputBytes);
    int bucket = items <= 1 ? 0 : Math.min(31, 64 - Long.numberOfLeadingZeros(items - 1));
    CallCounters.add(cell, CallCounters.BATCH_SIZES + bucket, 1);
    try {
      result = entry.kernel.execute(batch);
    } catch (IOException | RuntimeException e) {
      quarantine(batch.capability(), entry, m);
      throw e; // an undeclared failure is not a safe replay contract
    } finally {
      long end = System.nanoTime();
      CallCounters.add(cell, CallCounters.NATIVE_NANOS, end - nativeStart);
      CallCounters.add(cell, CallCounters.BRIDGE_NANOS, end - bridgeStart);
    }
    if (result == null || result.status() == null) {
      quarantine(batch.capability(), entry, m);
      throw new IOException("native result is missing status");
    }
    return switch (result.status()) {
      case OK -> {
        if (result.outputElements() != batch.output().elements() || result.detail() != 0) {
          quarantine(batch.capability(), entry, m);
          throw new IOException("native output metadata violates contract");
        }
        checkpoint();
        compareShadow(batch, entry, m, items, ownerScope);
        checkpoint();
        yield true;
      }
      case UNSUPPORTED -> {
        checkpoint();
        recordJava(batch.capability(), Reason.UNSUPPORTED);
        yield false;
      }
      case INVALID_INPUT -> {
        quarantine(batch.capability(), entry, m);
        throw new IllegalArgumentException("native rejected validated input: " + result.detail());
      }
      case INTERNAL_ERROR -> {
        quarantine(batch.capability(), entry, m);
        if (config.required().contains(batch.capability()))
          throw new IOException("required kernel failed");
        checkpoint();
        recordJava(batch.capability(), Reason.NATIVE_ERROR);
        yield false;
      }
    };
  }

  /**
   * Distinct input encodings, as an immutable {@code Set.of} that {@code Shape} keeps without
   * copying; batches have at most three inputs.
   */
  private static Set<AccelerationProvider.Encoding> encodings(
      List<AccelerationProvider.Input> inputs) {
    AccelerationProvider.Encoding a = null, b = null, c = null;
    for (var input : inputs) {
      var e = input.encoding();
      if (a == null || a == e) a = e;
      else if (b == null || b == e) b = e;
      else if (c == null || c == e) c = e;
      else return Set.copyOf(inputs.stream().map(AccelerationProvider.Input::encoding).toList());
    }
    return a == null
        ? Set.of(AccelerationProvider.Encoding.BYTES)
        : b == null ? Set.of(a) : c == null ? Set.of(a, b) : Set.of(a, b, c);
  }

  private static int bufferAlignment(ByteBuffer buffer) {
    if (!buffer.isDirect()) return 1;
    int remainder = buffer.alignmentOffset(buffer.position(), 64);
    return remainder == 0 ? 64 : Integer.lowestOneBit(remainder);
  }

  @SuppressWarnings(
      "try") // The admitted heap credits cover reference materialization until scope exit.
  private void compareShadow(
      AccelerationProvider.Batch primary,
      Entry entry,
      Metrics metrics,
      long items,
      RequestMemoryScope owner)
      throws IOException {
    var policy = config.shadowPolicy();
    // Sampling counts this thread's calls, so it needs no cross-thread sum on the call path.
    if (!policy.enabled() || metrics.calls.cell()[CallCounters.CALLS] % policy.sampleEvery() != 0)
      return;
    long bytes =
        Math.addExact(
            384,
            Math.addExact(primary.output().bytes().remaining(), primary.scratch().remaining()));
    for (var input : primary.inputs()) bytes = Math.addExact(bytes, input.byteSize());
    long referenceHeapBytes = JavaKernelOracle.referenceMemoryBytes(primary);
    bytes = Math.addExact(bytes, referenceHeapBytes);
    for (; ; ) {
      long current = shadowBytes.get();
      if (bytes > policy.maximumBytes() - current) {
        metrics.shadowBudgetDenials.increment();
        return;
      }
      if (shadowBytes.compareAndSet(current, current + bytes)) break;
    }
    shadowHighWater.accumulateAndGet(shadowBytes.get(), Math::max);
    try (RequestMemoryScope scope = owner == null ? newScope() : newScope(owner);
        var heapReservation = scope.reserveHeap(referenceHeapBytes)) {
      List<AccelerationProvider.Input> inputs = new ArrayList<>();
      for (var input : primary.inputs()) {
        ByteBuffer copy = scope.allocate(input.byteSize()).order(ByteOrder.LITTLE_ENDIAN);
        if (input.bytes() != null) copy.put(input.bytes().duplicate());
        else for (long i = 0; i < input.byteSize(); i++) copy.put(input.byteAt(i));
        copy.flip();
        inputs.add(new AccelerationProvider.Input(copy, input.elements(), input.encoding()));
      }
      ByteBuffer parameters = scope.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
      parameters.put(primary.parameters().duplicate()).flip();
      ByteBuffer output =
          scope.allocate(primary.output().bytes().remaining()).order(ByteOrder.LITTLE_ENDIAN);
      var secondary =
          new AccelerationProvider.Batch(
              primary.capability(),
              inputs,
              new AccelerationProvider.Output(output, primary.output().elements()),
              parameters,
              scope.allocate(primary.scratch().remaining()),
              scope.allocate(256));
      var result = JavaKernelOracle.execute(secondary);
      if (result.status() == AccelerationProvider.Status.UNSUPPORTED) return;
      metrics.shadowSamples.increment();
      boolean equal =
          result.status() == AccelerationProvider.Status.OK
              && result.outputElements() == primary.output().elements()
              && result.detail() == 0
              && output.duplicate().clear().equals(primary.output().bytes().duplicate().clear());
      if (!equal) {
        metrics.shadowMismatches.increment();
        synchronized (shadowMismatches) {
          if (shadowMismatches.size() < policy.mismatchLimit())
            shadowMismatches.add(
                new ShadowMismatch(
                    primary.capability().id, entry.identity, items, output.remaining()));
        }
        quarantine(primary.capability(), entry, metrics);
        throw new IOException("shadow comparison mismatch: " + primary.capability().id);
      }
      checkpoint();
    } catch (RequestMemoryScope.AdmissionException e) {
      checkpoint();
      if (e.reason() == RequestMemoryScope.Failure.CANCELLED
          || e.reason() == RequestMemoryScope.Failure.CLOSED
          || e.reason() == RequestMemoryScope.Failure.SHUTDOWN)
        throw new IOException("shadow owner stopped before publication", e);
      metrics.shadowBudgetDenials.increment();
    } finally {
      shadowBytes.addAndGet(-bytes);
    }
  }

  /**
   * Explicit retry of retained scratch close failures, including private synchronous call scopes.
   * Confined scratch must be recovered on its creating thread before that thread exits. A foreign
   * retry leaves its allocation charged and reports the failure; shared scratch can be recovered
   * from any thread. This does not cancel live work or retire healthy allocations.
   */
  public void retryFailedScratchReclamations() {
    RuntimeException failure = null;
    for (var record : activeScopes.values()) {
      try {
        record.scope().retryFailedReclamations();
      } catch (RuntimeException exception) {
        if (failure == null) failure = exception;
        else failure.addSuppressed(exception);
      }
    }
    if (failure != null) throw failure;
  }

  /**
   * Stops admission, cancels owned work and drains physical owners without force-freeing leases.
   */
  public void shutdown(long deadlineNanos) throws IOException {
    synchronized (drainLock) {
      stopping.set(true);
      ledger.shutdown();
    }
    IOException failure = null;
    try {
      ManagedMemory.retireAll(this);
    } catch (RuntimeException e) {
      failure = new IOException("cache retirement failed", e);
    }
    // The caller retains its ownership pin through a synchronous downcall and publication.
    // Cancelling never closes its buffers; its finally block performs physical retirement.
    for (var record : activeScopes.values()) record.scope().cancel();
    // Idle frames are freed here; a busy frame is freed when its owner closes it.
    for (CallFrame frame : liveFrames) {
      try {
        frame.retire();
      } catch (RuntimeException e) {
        if (failure == null) failure = new IOException("call frame retirement failed", e);
        else failure.addSuppressed(e);
      }
    }
    synchronized (drainLock) {
      while (!activeScopes.isEmpty()) {
        long left = deadlineNanos - System.nanoTime();
        if (left <= 0)
          throw new IOException(
              "shutdown deadline with pinned owners: "
                  + activeScopes.values().stream()
                      .limit(16)
                      .map(r -> Long.toString(r.id()))
                      .toList(),
              failure);
        try {
          drainLock.wait(Math.max(1, Math.min(1000, left / 1_000_000)));
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          var interrupted = new InterruptedIOException("shutdown interrupted with live owners");
          interrupted.initCause(e);
          throw interrupted;
        }
      }
    }
    try {
      long left = deadlineNanos - System.nanoTime();
      if (!ledger.awaitDrained(java.time.Duration.ofNanos(Math.max(0, left))))
        throw new IOException(
            "shutdown deadline with charged allocations: " + ledger.allocations(16), failure);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      var interrupted = new InterruptedIOException("shutdown interrupted with charged allocations");
      interrupted.initCause(e);
      throw interrupted;
    }
    if (journal != null) journal.close();
    if (failure != null) throw failure;
  }

  @Override
  public void close() throws IOException {
    shutdown(System.nanoTime() + 30_000_000_000L);
  }

  private void quarantine(Capability capability, Entry entry, Metrics metrics) {
    entry.healthy.set(false);
    if (journal != null) {
      try {
        journal.quarantine(NativeCallJournal.key(entry.identity));
      } catch (
          @SuppressWarnings("unused")
          IOException error) {
        ready.set(false);
      }
    }
    metrics.errors.increment();
    if (config.required().contains(capability)) ready.set(false);
  }

  /** Bounded per-selected-kernel critical-call observations; idle kernels have zero counters. */
  public Map<Capability, CriticalCallBudget.Snapshot> criticalCalls() {
    var result = new EnumMap<Capability, CriticalCallBudget.Snapshot>(Capability.class);
    entries.forEach((capability, entry) -> result.put(capability, entry.kernel.criticalCalls()));
    return Map.copyOf(result);
  }

  /** A consistent inventory of immutable selection and approximate monotonic runtime counters. */
  public Inventory inventory() {
    EnumMap<Capability, Counter> counters = new EnumMap<>(Capability.class);
    EnumMap<Capability, String> selected = new EnumMap<>(Capability.class);
    for (Capability capability : Capability.values()) {
      Metrics m = metrics.get(capability);
      EnumMap<Reason, Long> reasons = new EnumMap<>(Reason.class);
      for (var reason : m.reasons.entrySet()) reasons.put(reason.getKey(), reason.getValue().sum());
      Entry entry = entries.get(capability);
      long[] calls = m.calls.snapshot();
      Map<Integer, Long> batches = new java.util.HashMap<>();
      for (int i = 0; i < CallCounters.BUCKETS; i++)
        batches.put(i, calls[CallCounters.BATCH_SIZES + i]);
      if (entry != null) selected.put(capability, entry.identity);
      counters.put(
          capability,
          new Counter(
              m.java.sum(),
              calls[CallCounters.CALLS],
              calls[CallCounters.ITEMS],
              calls[CallCounters.STAGED],
              m.errors.sum(),
              startupQuarantines.contains(capability) || entry != null && !entry.healthy.get(),
              Map.copyOf(reasons),
              calls[CallCounters.BYTES],
              calls[CallCounters.NATIVE_NANOS],
              calls[CallCounters.BRIDGE_NANOS],
              calls[CallCounters.SCRATCH_HIGH_WATER],
              Map.copyOf(batches),
              m.shadowSamples.sum(),
              m.shadowBudgetDenials.sum(),
              m.shadowMismatches.sum()));
    }
    return new Inventory(
        BRIDGE_BUILD,
        state,
        loaded.stream()
            .map(
                p -> p.manifest().id() + "@" + p.manifest().version() + ":" + p.manifest().digest())
            .toList(),
        Map.copyOf(selected),
        rejections,
        ready.get(),
        Map.copyOf(counters),
        ledger.snapshot(),
        executionSnapshot());
  }

  public ExecutionSnapshot executionSnapshot() {
    List<ShadowMismatch> mismatches;
    synchronized (shadowMismatches) {
      mismatches = List.copyOf(shadowMismatches);
    }
    return new ExecutionSnapshot(
        activeScopes.size(),
        shadowBytes.get(),
        shadowHighWater.get(),
        stopping.get(),
        mismatches,
        stopping.get()
            ? activeScopes.values().stream().limit(16).map(r -> "scope:" + r.id()).toList()
            : List.of());
  }
}
