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
package org.apache.lucene.acceleration.memory;

import java.time.Duration;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

/**
 * Explicit admission for managed memory. This accounts enrolled capacity, not process RSS.
 * Admission and release use one short lock; allocation and reclamation always happen outside it.
 *
 * @lucene.experimental
 */
public final class MemoryLedger {
  /** Independent accounting dimensions; mapped capacity is virtual address space. */
  public enum Kind {
    HEAP,
    NATIVE,
    MAPPED
  }

  /** Allocation purpose, including optional partitions of the same node capacity. */
  public enum Purpose {
    REQUEST_SCRATCH,
    CACHE_VALUE,
    ORDINAL_BUILD,
    WARMED_VALUE,
    POOL_BLOCK
  }

  /** Mutually exclusive classifications of charged backing storage. */
  public enum State {
    BUILDING,
    PUBLISHED,
    ACTIVE,
    IDLE,
    RETIRED_PINNED,
    FAILED_RECLAMATION,
    RELEASED
  }

  /** Stable admission outcomes. */
  public enum Reason {
    GRANTED,
    NODE_CAPACITY,
    REQUEST_LIMIT,
    CORE_LIMIT,
    SHUTDOWN,
    FAIR_SHARE,
    QUEUE_FULL,
    CANCELLED,
    DEADLINE,
    PRESSURE
  }

  /** Admission classes distinguish foreground work from optional background warming. */
  public enum AdmissionClass {
    REQUEST,
    CACHE_BUILD,
    ORDINAL_BUILD,
    WARMING
  }

  /** All byte limits are explicit; native capacity is never derived from heap size. */
  public record Limits(long heapBytes, long nativeBytes, long mappedBytes) {
    public Limits {
      if (heapBytes < 0 || nativeBytes < 0 || mappedBytes < 0) {
        throw new IllegalArgumentException("negative memory limit");
      }
    }

    public long bytes(Kind kind) {
      return switch (kind) {
        case HEAP -> heapBytes;
        case NATIVE -> nativeBytes;
        case MAPPED -> mappedBytes;
      };
    }
  }

  /** Workers of a request must share its request ID; searcher generations share a core ID. */
  public record Owner(String ownerId, String coreId, long generation, String requestId) {
    public Owner {
      Objects.requireNonNull(ownerId);
      Objects.requireNonNull(coreId);
      Objects.requireNonNull(requestId);
    }
  }

  /** A denied admission carries no reservation. */
  public record Admission(Reason reason, Reservation reservation) {
    public boolean granted() {
      return reason == Reason.GRANTED;
    }
  }

  /** Consistent snapshot. Lifecycle buckets sum to charged capacity for each kind. */
  public record Snapshot(
      Limits installed,
      Limits target,
      Map<Kind, Long> charged,
      Map<Kind, Long> reserved,
      Map<Kind, Map<State, Long>> lifecycle,
      long failedReclamations,
      boolean shutdown) {}

  /** One immutable allocation identity, exposed only in on-demand diagnostics. */
  public record AllocationRecord(
      long id, Owner owner, Kind kind, Purpose purpose, long capacity, State state) {}

  // Package-private so tests can hold the admission monitor.
  final Object lock = new Object();
  private final long requestLimit;
  private final long[] charged = new long[Kind.values().length];
  private final long[] reserved = new long[Kind.values().length];
  // One bit per Kind ordinal, written under lock and read lock-free; enrollment never reverts.
  private volatile int enrolledKinds;
  private final Map<String, Long> requestBytes = new HashMap<>();
  private final Map<String, Long> coreBytes = new HashMap<>();
  private final EnumMap<Kind, Map<String, Long>> coreKindBytes = new EnumMap<>(Kind.class);
  private final Map<String, Long> coreLimits = new HashMap<>();
  private final Map<Long, Charge> allocations = new HashMap<>();
  private final AtomicLong allocationIds = new AtomicLong();
  private Limits installed;
  private Limits target;
  // Written under lock and read lock-free.
  private volatile boolean shutdown;
  private final Map<String, Integer> coreWeights = new HashMap<>();

  private record Waiter(String core, Kind kind) {}

  private final Map<Waiter, Integer> waiters = new HashMap<>();

  private static final class Demand {
    final Owner owner;
    final Kind kind;
    final long bytes;
    final Purpose purpose;

    Demand(Owner owner, Kind kind, long bytes, Purpose purpose) {
      this.owner = owner;
      this.kind = kind;
      this.bytes = bytes;
      this.purpose = purpose;
    }
  }

  private final java.util.ArrayDeque<Demand> demands = new java.util.ArrayDeque<>();
  private final EnumMap<Reason, Long> outcomes = new EnumMap<>(Reason.class);
  private long waitNanos;
  private long ordinalLimit = Long.MAX_VALUE;
  private long ordinalBytes;

  /**
   * Sets the ordinal build/retained partition of managed HEAP capacity. This is a sublimit of the
   * same node allowance, not another allocation pool. Shrinking below usage stops growth while
   * existing owners drain; setting zero after enrollment forbids further ordinal allocation.
   */
  public void setOrdinalLimit(long bytes) {
    requireCapacity(bytes);
    synchronized (lock) {
      ordinalLimit = bytes;
      lock.notifyAll();
    }
  }

  /** Current ordinal build reservations plus retained packed-heap capacity. */
  public long ordinalBytes() {
    synchronized (lock) {
      return ordinalBytes;
    }
  }

  /** Configured ordinal partition; {@link Long#MAX_VALUE} denotes the unenrolled default. */
  public long ordinalLimit() {
    synchronized (lock) {
      return ordinalLimit;
    }
  }

  /** Consistent, bounded-cardinality accounting telemetry, without exporting owner identities. */
  public record Telemetry(
      Map<Purpose, Long> byPurpose,
      Map<Reason, Long> admissions,
      int activeRequests,
      int activeCores,
      int queuedRequests,
      long waitNanos,
      long oldestPinnedNanos,
      long failedAllocations,
      Limits excess,
      long ordinalTargetBytes,
      long ordinalUsedBytes) {}

  /** Sets weighted soft fairness; idle cores do not strand capacity. */
  public void setCoreWeight(String coreId, int weight) {
    if (weight <= 0 || weight > 10000)
      throw new IllegalArgumentException("core weight outside 1..10000");
    synchronized (lock) {
      if (!coreWeights.containsKey(coreId) && coreWeights.size() >= 256)
        throw new IllegalArgumentException("core weight policy bound exceeded");
      coreWeights.put(Objects.requireNonNull(coreId), weight);
      lock.notifyAll();
    }
  }

  /** Creates a ledger with a combined heap/native request limit (mapped bytes also count). */
  public MemoryLedger(Limits limits, long requestLimit) {
    this.installed = this.target = Objects.requireNonNull(limits);
    if (requestLimit < 0) throw new IllegalArgumentException("negative request limit");
    this.requestLimit = requestLimit;
    for (Kind kind : Kind.values()) coreKindBytes.put(kind, new HashMap<>());
    enroll(limits);
  }

  /** Enrollment stays enabled when an installed budget is subsequently shrunk to zero. */
  public boolean isEnrolled(Kind kind) {
    return (enrolledKinds & (1 << Objects.requireNonNull(kind).ordinal())) != 0;
  }

  private void enroll(Limits limits) {
    int kinds = enrolledKinds;
    for (Kind kind : Kind.values()) if (limits.bytes(kind) != 0) kinds |= 1 << kind.ordinal();
    enrolledKinds = kinds;
  }

  /** Optional per-core burst ceiling; absent cores may use unused node capacity. */
  public void setCoreLimit(String coreId, long limit) {
    if (limit < 0) throw new IllegalArgumentException("negative core limit");
    synchronized (lock) {
      if (!coreLimits.containsKey(coreId) && coreLimits.size() >= 256)
        throw new IllegalArgumentException("core limit policy bound exceeded");
      coreLimits.put(Objects.requireNonNull(coreId), limit);
      lock.notifyAll();
    }
  }

  /** Reserves additional peak capacity before any allocator is called. */
  public Admission tryReserve(Owner owner, Kind kind, long bytes, Purpose purpose) {
    AdmissionClass admissionClass =
        switch (purpose) {
          case WARMED_VALUE, POOL_BLOCK -> AdmissionClass.WARMING;
          case CACHE_VALUE -> AdmissionClass.CACHE_BUILD;
          case ORDINAL_BUILD -> AdmissionClass.ORDINAL_BUILD;
          case REQUEST_SCRATCH -> AdmissionClass.REQUEST;
        };
    return tryReserve(owner, kind, bytes, purpose, admissionClass);
  }

  /** Optional warming yields the final fifth of capacity to foreground callers. */
  public Admission tryReserve(
      Owner owner, Kind kind, long bytes, Purpose purpose, AdmissionClass admissionClass) {
    Objects.requireNonNull(admissionClass);
    Objects.requireNonNull(owner);
    Objects.requireNonNull(kind);
    Objects.requireNonNull(purpose);
    requireCapacity(bytes);
    synchronized (lock) {
      Reason reason = check(owner, kind, purpose, bytes);
      if (reason == Reason.GRANTED && admissionClass == AdmissionClass.WARMING) {
        long capacity = Math.min(installed.bytes(kind), target.bytes(kind));
        long warmCeiling = capacity - capacity / 5;
        if (charged[kind.ordinal()] + reserved[kind.ordinal()] + bytes > warmCeiling)
          reason = Reason.PRESSURE;
      }
      outcomes.merge(reason, 1L, Long::sum);
      if (reason != Reason.GRANTED) return new Admission(reason, null);
      addReserved(owner, kind, purpose, bytes);
      return new Admission(Reason.GRANTED, new Reservation(owner, kind, purpose, bytes));
    }
  }

  /** Bounded optional waiting. Cancellation is checked outside the ledger admission lock. */
  public Admission awaitReserve(
      Owner owner,
      Kind kind,
      long bytes,
      Purpose purpose,
      Duration timeout,
      BooleanSupplier cancelled)
      throws InterruptedException {
    return awaitReserve(owner, kind, bytes, purpose, timeout, cancelled, () -> {});
  }

  /** Foreground waiting with one pressure reclamation outside the admission lock. */
  public Admission awaitReserve(
      Owner owner,
      Kind kind,
      long bytes,
      Purpose purpose,
      Duration timeout,
      BooleanSupplier cancelled,
      Runnable reclaim)
      throws InterruptedException {
    Objects.requireNonNull(timeout);
    Objects.requireNonNull(cancelled);
    Objects.requireNonNull(reclaim);
    long budget = timeout.toNanos();
    if (budget < 0) throw new IllegalArgumentException("negative wait timeout");
    if (cancelled.getAsBoolean()) return denied(Reason.CANCELLED);
    long start = System.nanoTime();
    Admission initial = tryReserve(owner, kind, bytes, purpose);
    if (initial.granted() && budget > 0 && System.nanoTime() - start >= budget) {
      initial.reservation().close();
      return denied(Reason.DEADLINE);
    }
    if (initial.granted()
        || initial.reason() == Reason.SHUTDOWN
        || initial.reason() == Reason.REQUEST_LIMIT
        || initial.reason() == Reason.CORE_LIMIT) return initial;
    if (budget == 0) return denied(Reason.DEADLINE);
    Waiter waiter = new Waiter(owner.coreId(), kind);
    Demand demand = new Demand(owner, kind, bytes, purpose);
    synchronized (lock) {
      // Growth may succeed immediately, but never queues while retaining an earlier working set.
      if (requestBytes.getOrDefault(owner.requestId(), 0L) != 0) return denied(Reason.PRESSURE);
      if (bytes > Math.min(installed.bytes(kind), target.bytes(kind))) return initial;
      int queued = waiters.values().stream().mapToInt(Integer::intValue).sum();
      if (queued >= 256) {
        outcomes.merge(Reason.QUEUE_FULL, 1L, Long::sum);
        return new Admission(Reason.QUEUE_FULL, null);
      }
      waiters.merge(waiter, 1, Integer::sum);
      demands.addLast(demand);
    }
    try {
      reclaim.run();
      while (true) {
        if (cancelled.getAsBoolean()) return denied(Reason.CANCELLED);
        synchronized (lock) {
          long elapsed = System.nanoTime() - start;
          if (elapsed >= budget) return denied(Reason.DEADLINE);
          Admission admission = tryReserve(owner, kind, bytes, purpose);
          if (admission.granted()
              || admission.reason() == Reason.SHUTDOWN
              || admission.reason() == Reason.REQUEST_LIMIT
              || admission.reason() == Reason.CORE_LIMIT) return admission;
          if (requestBytes.getOrDefault(owner.requestId(), 0L) != 0) return denied(Reason.PRESSURE);
          long pause = Math.min(budget - elapsed, 10_000_000L);
          lock.wait(pause / 1_000_000, (int) (pause % 1_000_000));
        }
      }
    } catch (InterruptedException interrupted) {
      denied(Reason.CANCELLED);
      throw interrupted;
    } finally {
      synchronized (lock) {
        demands.remove(demand);
        int remaining = waiters.get(waiter) - 1;
        if (remaining == 0) waiters.remove(waiter);
        else waiters.put(waiter, remaining);
        waitNanos += System.nanoTime() - start;
        lock.notifyAll();
      }
    }
  }

  private Admission denied(Reason reason) {
    synchronized (lock) {
      outcomes.merge(reason, 1L, Long::sum);
    }
    return new Admission(reason, null);
  }

  /** Runs a caller-owned pressure eviction outside every ledger lock, then retries admission. */
  public Admission tryReserveWithReclaim(
      Owner owner, Kind kind, long bytes, Purpose purpose, Runnable reclaim) {
    Admission first = tryReserve(owner, kind, bytes, purpose);
    if (first.reason() != Reason.NODE_CAPACITY && first.reason() != Reason.FAIR_SHARE) return first;
    Objects.requireNonNull(reclaim).run();
    return tryReserve(owner, kind, bytes, purpose);
  }

  /** Waits for shutdown drainage without reclaiming any live owner or worker. */
  public boolean awaitDrained(Duration timeout) throws InterruptedException {
    long budget = timeout.toNanos(), start = System.nanoTime();
    if (budget < 0) throw new IllegalArgumentException("negative drain timeout");
    synchronized (lock) {
      while (true) {
        boolean empty = true;
        for (int i = 0; i < charged.length; i++) empty &= charged[i] == 0 && reserved[i] == 0;
        if (empty) return true;
        long remaining = budget - (System.nanoTime() - start);
        if (remaining <= 0) return false;
        lock.wait(remaining / 1_000_000, (int) (remaining % 1_000_000));
      }
    }
  }

  /** Pressure, failure, purpose, and waiter metrics from one ledger instant. */
  public Telemetry telemetry() {
    synchronized (lock) {
      EnumMap<Purpose, Long> purposes = new EnumMap<>(Purpose.class);
      for (Purpose purpose : Purpose.values()) purposes.put(purpose, 0L);
      long oldest = 0, failed = 0, now = System.nanoTime();
      for (Charge allocation : allocations.values()) {
        purposes.merge(allocation.purpose, allocation.capacity, Long::sum);
        if (allocation.state == State.RETIRED_PINNED)
          oldest = Math.max(oldest, now - allocation.classifiedNanos);
        if (allocation.state == State.FAILED_RECLAMATION) failed++;
      }
      return new Telemetry(
          Map.copyOf(purposes),
          Map.copyOf(outcomes),
          requestBytes.size(),
          coreBytes.size(),
          waiters.values().stream().mapToInt(Integer::intValue).sum(),
          waitNanos,
          oldest,
          failed,
          new Limits(
              Math.max(0, charged[0] + reserved[0] - target.heapBytes()),
              Math.max(0, charged[1] + reserved[1] - target.nativeBytes()),
              Math.max(0, charged[2] + reserved[2] - target.mappedBytes())),
          ordinalLimit,
          ordinalBytes);
    }
  }

  /** Freezes incompatible growth immediately; installs lower limits when charges drain. */
  public void setTargetLimits(Limits limits) {
    synchronized (lock) {
      target = Objects.requireNonNull(limits);
      enroll(limits);
      drainLimits();
    }
  }

  /** Prohibits new reservations; existing owners drain without forced reclamation. */
  public void shutdown() {
    synchronized (lock) {
      shutdown = true;
      lock.notifyAll();
    }
  }

  /** Whether {@link #shutdown()} has been called; reads without the ledger lock. */
  public boolean isShutdown() {
    return shutdown;
  }

  /** Captures a consistent view without calling external code. */
  public Snapshot snapshot() {
    synchronized (lock) {
      EnumMap<Kind, Long> c = new EnumMap<>(Kind.class);
      EnumMap<Kind, Long> r = new EnumMap<>(Kind.class);
      EnumMap<Kind, Map<State, Long>> lifecycle = new EnumMap<>(Kind.class);
      long failures = 0;
      for (Kind kind : Kind.values()) {
        c.put(kind, charged[kind.ordinal()]);
        r.put(kind, reserved[kind.ordinal()]);
        EnumMap<State, Long> states = new EnumMap<>(State.class);
        for (State state : State.values()) states.put(state, 0L);
        for (Charge charge : allocations.values()) {
          if (charge.kind == kind) states.merge(charge.state, charge.capacity, Long::sum);
        }
        failures += states.get(State.FAILED_RECLAMATION);
        lifecycle.put(kind, Map.copyOf(states));
      }
      return new Snapshot(
          installed,
          target,
          Map.copyOf(c),
          Map.copyOf(r),
          Map.copyOf(lifecycle),
          failures,
          shutdown);
    }
  }

  /** Bounded per-core admitted capacity; includes both reservations and committed storage. */
  public record CoreUsage(int weight, long limitBytes, Map<Kind, Long> admittedBytes, int queued) {}

  public Map<String, CoreUsage> coreUsage(int maximum) {
    if (maximum < 0 || maximum > 256) throw new IllegalArgumentException("core diagnostic bound");
    synchronized (lock) {
      Map<String, CoreUsage> result = new java.util.LinkedHashMap<>();
      for (String core : coreWeights.keySet()) addCoreUsage(result, core, maximum);
      for (String core : coreLimits.keySet()) addCoreUsage(result, core, maximum);
      for (String core : coreBytes.keySet()) addCoreUsage(result, core, maximum);
      for (Waiter waiter : waiters.keySet()) addCoreUsage(result, waiter.core(), maximum);
      return Map.copyOf(result);
    }
  }

  private void addCoreUsage(Map<String, CoreUsage> result, String core, int maximum) {
    if (result.size() >= maximum || result.containsKey(core)) return;
    Map<Kind, Long> bytes = new EnumMap<>(Kind.class);
    for (Kind kind : Kind.values()) bytes.put(kind, coreKindBytes.get(kind).getOrDefault(core, 0L));
    int queued = 0;
    for (var waiter : waiters.entrySet())
      if (waiter.getKey().core().equals(core)) queued += waiter.getValue();
    result.put(
        core,
        new CoreUsage(
            coreWeights.getOrDefault(core, 1),
            coreLimits.getOrDefault(core, Long.MAX_VALUE),
            Map.copyOf(bytes),
            queued));
  }

  /** Returns a bounded number of immutable allocation diagnostics. */
  public java.util.List<AllocationRecord> allocations(int maximum) {
    if (maximum < 0) throw new IllegalArgumentException("negative diagnostic limit");
    synchronized (lock) {
      return allocations.values().stream().limit(maximum).map(Charge::record).toList();
    }
  }

  private Reason check(Owner owner, Kind kind, Purpose purpose, long bytes) {
    if (shutdown) return Reason.SHUTDOWN;
    long currentRequest = requestBytes.getOrDefault(owner.requestId(), 0L);
    if (bytes > requestLimit || currentRequest > requestLimit - bytes) return Reason.REQUEST_LIMIT;
    long coreLimit = coreLimits.getOrDefault(owner.coreId(), Long.MAX_VALUE);
    long currentCore = coreBytes.getOrDefault(owner.coreId(), 0L);
    if (bytes > coreLimit || currentCore > coreLimit - bytes) return Reason.CORE_LIMIT;
    int index = kind.ordinal();
    long effective = Math.min(installed.bytes(kind), target.bytes(kind));
    if (bytes > effective
        || charged[index] > effective - bytes
        || reserved[index] > effective - bytes - charged[index]) return Reason.NODE_CAPACITY;
    if (isOrdinal(kind, purpose) && (bytes > ordinalLimit || ordinalBytes > ordinalLimit - bytes))
      return Reason.NODE_CAPACITY;
    return fairShare(owner, kind, bytes, effective, purpose);
  }

  private Reason fairShare(Owner owner, Kind kind, long bytes, long effective, Purpose purpose) {
    if (coreWeights.isEmpty() == false && waiters.isEmpty() == false) {
      Map<String, Long> byCore = coreKindBytes.get(kind);
      boolean ownerPresent =
          coreWeights.containsKey(owner.coreId()) || byCore.containsKey(owner.coreId());
      long totalWeight = 0;
      for (int weight : coreWeights.values()) totalWeight += weight;
      for (String core : byCore.keySet()) if (!coreWeights.containsKey(core)) totalWeight++;
      for (Waiter waiting : waiters.keySet()) {
        if (waiting.kind() != kind) continue;
        if (waiting.core().equals(owner.coreId())) ownerPresent = true;
        if (!coreWeights.containsKey(waiting.core()) && !byCore.containsKey(waiting.core()))
          totalWeight++;
      }
      if (!ownerPresent) totalWeight++;
      long share = share(effective, totalWeight, coreWeights.getOrDefault(owner.coreId(), 1));
      if (byCore.getOrDefault(owner.coreId(), 0L) > share - bytes) {
        for (Waiter waiting : waiters.keySet()) {
          if (waiting.kind() != kind || waiting.core().equals(owner.coreId())) continue;
          long otherShare =
              share(effective, totalWeight, coreWeights.getOrDefault(waiting.core(), 1));
          if (byCore.getOrDefault(waiting.core(), 0L) < otherShare
              && !oversizedTurn(owner, kind, bytes, effective, totalWeight, purpose))
            return Reason.FAIR_SHARE;
        }
      }
    }
    return Reason.GRANTED;
  }

  private boolean fitsHardLimits(Demand demand, long effective) {
    int index = demand.kind.ordinal();
    return demand.bytes <= effective - charged[index] - reserved[index]
        && demand.bytes <= requestLimit - requestBytes.getOrDefault(demand.owner.requestId(), 0L)
        && demand.bytes
            <= coreLimits.getOrDefault(demand.owner.coreId(), Long.MAX_VALUE)
                - coreBytes.getOrDefault(demand.owner.coreId(), 0L)
        && (!isOrdinal(demand.kind, demand.purpose) || demand.bytes <= ordinalLimit - ordinalBytes);
  }

  private boolean oversizedTurn(
      Owner owner, Kind kind, long bytes, long effective, long totalWeight, Purpose purpose) {
    // Soft shares cannot deadlock first allocations larger than every contender's share.
    // Protected requests that fit their share win first; otherwise oversized requests serialize
    // FIFO.
    Map<String, Long> byCore = coreKindBytes.get(kind);
    Demand oldest = null;
    for (Demand demand : demands) {
      if (demand.kind != kind || !fitsHardLimits(demand, effective)) continue;
      long used = byCore.getOrDefault(demand.owner.coreId(), 0L);
      long target =
          share(effective, totalWeight, coreWeights.getOrDefault(demand.owner.coreId(), 1));
      if (demand.bytes <= target - used) return false;
      if (oldest == null
          && used == 0
          && requestBytes.getOrDefault(demand.owner.requestId(), 0L) == 0) oldest = demand;
    }
    return oldest != null
        && oldest.owner.requestId().equals(owner.requestId())
        && oldest.owner.coreId().equals(owner.coreId())
        && oldest.bytes == bytes
        && oldest.purpose == purpose;
  }

  private static long share(long capacity, long totalWeight, int weight) {
    // Weight and configured-core bounds keep the remainder product within long range.
    return capacity / totalWeight * weight + capacity % totalWeight * weight / totalWeight;
  }

  private static boolean isOrdinal(Kind kind, Purpose purpose) {
    return kind == Kind.HEAP && purpose == Purpose.ORDINAL_BUILD;
  }

  private void addReserved(Owner owner, Kind kind, Purpose purpose, long bytes) {
    reserved[kind.ordinal()] += bytes;
    if (isOrdinal(kind, purpose)) ordinalBytes += bytes;
    adjust(requestBytes, owner.requestId(), bytes);
    adjust(coreBytes, owner.coreId(), bytes);
    adjust(coreKindBytes.get(kind), owner.coreId(), bytes);
  }

  private static void adjust(Map<String, Long> counters, String key, long delta) {
    long next = Math.addExact(counters.getOrDefault(key, 0L), delta);
    if (next == 0) counters.remove(key);
    else counters.put(key, next);
  }

  private void drainLimits() {
    long[] limits = new long[Kind.values().length];
    for (Kind kind : Kind.values()) {
      int i = kind.ordinal();
      limits[i] =
          target.bytes(kind) >= charged[i] + reserved[i]
              ? target.bytes(kind)
              : installed.bytes(kind);
    }
    installed = new Limits(limits[0], limits[1], limits[2]);
  }

  private static void requireCapacity(long bytes) {
    if (bytes < 0) throw new IllegalArgumentException("negative capacity");
  }

  /** Calculates element payload with checked multiplication. */
  public static long bytes(long count, int elementBytes) {
    requireCapacity(count);
    if (elementBytes <= 0) throw new IllegalArgumentException("invalid element size");
    return Math.multiplyExact(count, elementBytes);
  }

  /** Outstanding capacity credits. All methods are thread safe and close is idempotent. */
  public final class Reservation implements AutoCloseable {
    private final Owner owner;
    private final Kind kind;
    private final Purpose purpose;
    private long remaining;
    private boolean closed;

    private Reservation(Owner owner, Kind kind, Purpose purpose, long bytes) {
      this.owner = owner;
      this.kind = kind;
      this.purpose = purpose;
      this.remaining = bytes;
    }

    public long remaining() {
      synchronized (lock) {
        return remaining;
      }
    }

    public boolean tryGrow(long additionalBytes) {
      requireCapacity(additionalBytes);
      synchronized (lock) {
        if (closed) throw new IllegalStateException("closed reservation");
        if (check(owner, kind, purpose, additionalBytes) != Reason.GRANTED) return false;
        addReserved(owner, kind, purpose, additionalBytes);
        remaining += additionalBytes;
        return true;
      }
    }

    /** Converts reserved capacity atomically; allocation must not exceed this reservation. */
    public Charge commit(long capacity, State state) {
      requireCapacity(capacity);
      Objects.requireNonNull(state);
      if (state == State.RELEASED) throw new IllegalArgumentException("released initial state");
      synchronized (lock) {
        if (closed || capacity > remaining)
          throw new IllegalStateException("insufficient reservation");
        remaining -= capacity;
        reserved[kind.ordinal()] -= capacity;
        charged[kind.ordinal()] += capacity;
        Charge charge =
            new Charge(allocationIds.incrementAndGet(), owner, kind, purpose, capacity, state);
        allocations.put(charge.id, charge);
        return charge;
      }
    }

    @Override
    public void close() {
      synchronized (lock) {
        if (closed) return;
        closed = true;
        reserved[kind.ordinal()] -= remaining;
        if (isOrdinal(kind, purpose)) ordinalBytes -= remaining;
        adjust(requestBytes, owner.requestId(), -remaining);
        adjust(coreBytes, owner.coreId(), -remaining);
        adjust(coreKindBytes.get(kind), owner.coreId(), -remaining);
        remaining = 0;
        drainLimits();
        lock.notifyAll();
      }
    }
  }

  /** Charge held until successful physical close, or explicit managed-heap reference release. */
  public final class Charge {
    private final long id;
    private Owner owner;
    private final Kind kind;
    private final Purpose purpose;
    private final long capacity;
    private State state;
    private long classifiedNanos = System.nanoTime();

    private Charge(long id, Owner owner, Kind kind, Purpose purpose, long capacity, State state) {
      this.id = id;
      this.owner = owner;
      this.kind = kind;
      this.purpose = purpose;
      this.capacity = capacity;
      this.state = state;
    }

    private AllocationRecord record() {
      return new AllocationRecord(id, owner, kind, purpose, capacity, state);
    }

    MemoryLedger ledger() {
      return MemoryLedger.this;
    }

    public AllocationRecord snapshot() {
      synchronized (lock) {
        return record();
      }
    }

    /** A late retirement classification cannot resurrect a physically released charge. */
    public void reclassify(State next) {
      Objects.requireNonNull(next);
      if (next == State.RELEASED)
        throw new IllegalArgumentException("use releaseAfterDeallocation");
      synchronized (lock) {
        if (state != State.RELEASED && state != State.FAILED_RECLAMATION && state != next) {
          state = next;
          classifiedNanos = System.nanoTime();
        }
      }
    }

    /** Transfers existing capacity without any refund or additional node charge. */
    public boolean transferOwner(Owner next, State nextState) {
      Objects.requireNonNull(next);
      Objects.requireNonNull(nextState);
      if (nextState == State.RELEASED) throw new IllegalArgumentException("released destination");
      synchronized (lock) {
        if (state == State.RELEASED || state == State.FAILED_RECLAMATION) {
          throw new IllegalStateException("allocation cannot be transferred: " + state);
        }
        long requestAdditional = owner.requestId().equals(next.requestId()) ? 0 : capacity;
        long coreAdditional = owner.coreId().equals(next.coreId()) ? 0 : capacity;
        if (requestBytes.getOrDefault(next.requestId(), 0L) > requestLimit - requestAdditional
            || coreBytes.getOrDefault(next.coreId(), 0L)
                > coreLimits.getOrDefault(next.coreId(), Long.MAX_VALUE) - coreAdditional)
          return false;
        if (coreAdditional > 0
            && fairShare(
                    next,
                    kind,
                    coreAdditional,
                    Math.min(installed.bytes(kind), target.bytes(kind)),
                    purpose)
                != Reason.GRANTED) return false;
        adjust(requestBytes, owner.requestId(), -capacity);
        adjust(coreBytes, owner.coreId(), -capacity);
        adjust(coreKindBytes.get(kind), owner.coreId(), -capacity);
        adjust(requestBytes, next.requestId(), capacity);
        adjust(coreBytes, next.coreId(), capacity);
        adjust(coreKindBytes.get(kind), next.coreId(), capacity);
        owner = next;
        state = nextState;
        classifiedNanos = System.nanoTime();
        return true;
      }
    }

    /** Invoke only after physical release succeeds; duplicates do not refund twice. */
    public void releaseAfterDeallocation() {
      synchronized (lock) {
        if (state == State.RELEASED) return;
        charged[kind.ordinal()] -= capacity;
        if (isOrdinal(kind, purpose)) ordinalBytes -= capacity;
        adjust(requestBytes, owner.requestId(), -capacity);
        adjust(coreBytes, owner.coreId(), -capacity);
        adjust(coreKindBytes.get(kind), owner.coreId(), -capacity);
        state = State.RELEASED;
        allocations.remove(id);
        drainLimits();
        lock.notifyAll();
      }
    }
  }
}
