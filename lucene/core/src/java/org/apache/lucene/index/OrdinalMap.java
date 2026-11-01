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

package org.apache.lucene.index;

import java.io.IOException;
import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BooleanSupplier;
import org.apache.lucene.acceleration.AccelerationRuntime;
import org.apache.lucene.acceleration.memory.GenerationCache;
import org.apache.lucene.acceleration.memory.ManagedMemory;
import org.apache.lucene.acceleration.memory.MemoryLedger;
import org.apache.lucene.acceleration.memory.RequestMemoryScope;
import org.apache.lucene.acceleration.memory.SharedValueLease;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.Accountables;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;
import org.apache.lucene.util.InPlaceMergeSorter;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.PriorityQueue;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.packed.PackedInts;
import org.apache.lucene.util.packed.PackedLongValues;

/**
 * Maps per-segment ordinals to/from global ordinal space, using a compact packed-ints
 * representation.
 *
 * <p><b>NOTE</b>: this is a costly operation, as it must merge sort all terms, and may require
 * non-trivial RAM once done. It's better to operate in segment-private ordinal space instead when
 * possible.
 *
 * @lucene.internal
 */
public class OrdinalMap implements Accountable, AutoCloseable {
  private static final class Shared {
    static final ConcurrentHashMap<IndexReader.CacheKey, GenerationCache<String, OrdinalMap>>
        CACHE = new ConcurrentHashMap<>();
  }

  /** Opt-in reader-generation singleflight. Field keys must describe identical ordinal inputs. */
  public static SharedValueLease<OrdinalMap> buildShared(
      IndexReader.CacheHelper owner,
      String field,
      TermsEnum[] subs,
      long[] weights,
      float acceptableOverheadRatio,
      BooleanSupplier cancelled)
      throws IOException, InterruptedException {
    GenerationCache<String, OrdinalMap> cache =
        Shared.CACHE.computeIfAbsent(
            owner.getKey(),
            key -> {
              GenerationCache<String, OrdinalMap> created = new GenerationCache<>(0);
              ManagedMemory.Registration registration =
                  ManagedMemory.registerOwner(AccelerationRuntime.system(), created::close);
              owner.addClosedListener(
                  closedKey -> {
                    registration.close();
                    GenerationCache<String, OrdinalMap> removed = Shared.CACHE.remove(closedKey);
                    if (removed != null) removed.close();
                  });
              return created;
            });
    return cache.acquire(
        field,
        0,
        stop -> {
          Managed map =
              buildManaged(
                  owner.getKey(),
                  subs,
                  weights,
                  acceptableOverheadRatio,
                  stop,
                  AccelerationRuntime.system().ledger(),
                  false);
          MemoryLedger.Charge charge = map.memoryCharge;
          if (charge == null) {
            map.close();
            throw new IllegalStateException(
                "shared ordinal ownership requires managed heap enrollment");
          }
          return new SharedValueLease.Entry<>(map, map::close, charge);
        },
        cancelled);
  }

  /** Merge-loop entries admitted per ledger round trip; overshoot is at most one chunk. */
  static final int BUDGET_CHUNK_ENTRIES = 1024;

  /** Reserved builder bytes per merge-loop entry. */
  static final long BUDGET_ENTRY_BYTES = 32;

  private static final class BuildBudget implements AutoCloseable {
    private final MemoryLedger.Reservation reservation;
    private final BooleanSupplier cancelled;
    private final MemoryLedger.Owner durable;
    private final AccelerationRuntime runtime;
    private final BytesRefBuilder processing = new BytesRefBuilder();
    private boolean processed;
    private int credits;

    BuildBudget(int segments, BooleanSupplier cancelled, MemoryLedger ledger, boolean checkTask)
        throws IOException {
      this.cancelled = cancelled;
      AccelerationRuntime system = AccelerationRuntime.system();
      runtime = checkTask && system.ledger() == ledger ? system : null;
      if (runtime != null) runtime.checkpoint();
      MemoryLedger.Owner initial =
          ManagedMemory.owner("ordinal-map", MemoryLedger.Purpose.ORDINAL_BUILD);
      MemoryLedger.Owner bound = system.ledger() == ledger ? system.currentOwner() : null;
      MemoryLedger.Owner builder = bound == null ? initial : bound;
      durable =
          new MemoryLedger.Owner(
              initial.ownerId(), builder.coreId(), builder.generation(), initial.requestId());
      reservation =
          ManagedMemory.reserve(
              ledger,
              builder,
              MemoryLedger.Kind.HEAP,
              Math.addExact(131072, MemoryLedger.bytes(segments, 65536)),
              MemoryLedger.Purpose.ORDINAL_BUILD);
    }

    void add() throws IOException {
      if (--credits < 0) refill();
    }

    /** Charges a global ord's two entries once, when the first sub moves past its term. */
    void advancePast(BytesRef term) throws IOException {
      if (processed && processing.get().bytesEquals(term)) return;
      processed = true;
      processing.copyBytes(term);
      add();
      add();
    }

    private void refill() throws IOException {
      if (runtime != null) runtime.checkpoint();
      if (cancelled.getAsBoolean()) throw new IOException("cancelled ordinal build");
      if (reservation != null
          && reservation.tryGrow(BUDGET_CHUNK_ENTRIES * BUDGET_ENTRY_BYTES) == false)
        throw new RequestMemoryScope.AdmissionException(RequestMemoryScope.Failure.NODE_CAPACITY);
      credits = BUDGET_CHUNK_ENTRIES - 1;
    }

    void retain(Managed map) throws IOException {
      if (runtime != null) runtime.checkpoint();
      if (cancelled.getAsBoolean()) throw new IOException("cancelled ordinal publication");
      if (reservation != null) {
        map.memoryCharge = reservation.commit(map.ramBytesUsed(), MemoryLedger.State.PUBLISHED);
        if (map.memoryCharge.transferOwner(durable, MemoryLedger.State.PUBLISHED) == false) {
          map.memoryCharge.releaseAfterDeallocation();
          throw new RequestMemoryScope.AdmissionException(RequestMemoryScope.Failure.CORE_LIMIT);
        }
        map.memoryCleanup = ManagedMemory.retainHeap(map, map.memoryCharge);
      }
    }

    @Override
    public void close() {
      if (reservation != null) reservation.close();
    }
  }

  /** Charges a managed build's merge-loop entries as the stock constructor advances each sub. */
  private static final class BudgetedTermsEnum extends FilterLeafReader.FilterTermsEnum {
    private final BuildBudget budget;
    private BytesRef term;
    private long nextOrd;

    BudgetedTermsEnum(TermsEnum in, BuildBudget budget) {
      super(in);
      this.budget = budget;
    }

    @Override
    public BytesRef next() throws IOException {
      // The constructor advances every sub positioned on a term once that term is merged.
      if (term != null) budget.advancePast(term);
      term = in.next();
      if (term != null) {
        // One entry per segment ord, including ords the constructor fills for skipped ones.
        for (long ord = in.ord(); nextOrd <= ord; nextOrd++) budget.add();
      }
      return term;
    }
  }

  /**
   * A map from an explicit managed build ({@link #buildShared} or {@link
   * #build(IndexReader.CacheKey, TermsEnum[], long[], float, MemoryLedger)}). It reads through a
   * stock map that close drops, so only these maps are closeable and check for close.
   */
  private static final class Managed extends OrdinalMap {
    private volatile OrdinalMap mappings;
    private Cleaner.Cleanable memoryCleanup;
    private MemoryLedger.Charge memoryCharge;

    Managed(OrdinalMap mappings) {
      super(mappings.owner, mappings.valueCount, mappings.ramBytesUsed);
      this.mappings = mappings;
    }

    private OrdinalMap open() {
      OrdinalMap open = mappings;
      if (open == null) throw new IllegalStateException("closed ordinal map");
      return open;
    }

    /** Drops references after the final shared lease; borrowed mappings then reject reads. */
    @Override
    public synchronized void close() {
      if (mappings == null) return;
      mappings = null;
      if (memoryCleanup != null) memoryCleanup.clean();
    }

    @Override
    public LongValues getGlobalOrds(int segmentIndex) {
      int mapped = open().segmentMap.oldToNew(segmentIndex);
      // Keep the owning map alive even when a caller retains only this borrowed mapping.
      return new LongValues() {
        @Override
        public long get(long ord) {
          return open().segmentToGlobalOrds[mapped].get(ord);
        }
      };
    }

    @Override
    public long getFirstSegmentOrd(long globalOrd) {
      return open().getFirstSegmentOrd(globalOrd);
    }

    @Override
    public int getFirstSegmentNumber(long globalOrd) {
      return open().getFirstSegmentNumber(globalOrd);
    }

    @Override
    public Collection<Accountable> getChildResources() {
      return open().getChildResources();
    }
  }

  /** Maps from the default builds hold no managed resources, so closing them does nothing. */
  @Override
  public void close() {}

  // TODO: we could also have a utility method to merge Terms[] and use size() as a weight when we
  // need it
  // TODO: use more efficient packed ints structures?

  private static class TermsEnumPriorityQueue extends PriorityQueue<TermsEnumIndex> {

    TermsEnumPriorityQueue(int size) {
      super(size);
    }

    @Override
    protected boolean lessThan(TermsEnumIndex a, TermsEnumIndex b) {
      return a.compareTermTo(b) < 0;
    }
  }

  private static class SegmentMap implements Accountable {
    private static final long BASE_RAM_BYTES_USED =
        RamUsageEstimator.shallowSizeOfInstance(SegmentMap.class);

    /** Build a map from an index into a sorted view of `weights` to an index into `weights`. */
    private static int[] map(final long[] weights) {
      final int[] newToOld = new int[weights.length];
      for (int i = 0; i < weights.length; ++i) {
        newToOld[i] = i;
      }
      new InPlaceMergeSorter() {
        @Override
        protected void swap(int i, int j) {
          final int tmp = newToOld[i];
          newToOld[i] = newToOld[j];
          newToOld[j] = tmp;
        }

        @Override
        protected int compare(int i, int j) {
          // j first since we actually want higher weights first
          return Long.compare(weights[newToOld[j]], weights[newToOld[i]]);
        }
      }.sort(0, weights.length);
      return newToOld;
    }

    /** Inverse the map. */
    private static int[] inverse(int[] map) {
      final int[] inverse = new int[map.length];
      for (int i = 0; i < map.length; ++i) {
        inverse[map[i]] = i;
      }
      return inverse;
    }

    private final int[] newToOld, oldToNew;

    SegmentMap(long[] weights) {
      newToOld = map(weights);
      oldToNew = inverse(newToOld);
      assert Arrays.equals(newToOld, inverse(oldToNew));
    }

    int newToOld(int segment) {
      return newToOld[segment];
    }

    int oldToNew(int segment) {
      return oldToNew[segment];
    }

    @Override
    public long ramBytesUsed() {
      return BASE_RAM_BYTES_USED
          + RamUsageEstimator.sizeOf(newToOld)
          + RamUsageEstimator.sizeOf(oldToNew);
    }
  }

  /**
   * Create an ordinal map that uses the number of unique values of each {@link SortedDocValues}
   * instance as a weight.
   *
   * @see #build(IndexReader.CacheKey, TermsEnum[], long[], float)
   */
  public static OrdinalMap build(
      IndexReader.CacheKey owner, SortedDocValues[] values, float acceptableOverheadRatio)
      throws IOException {
    final TermsEnum[] subs = new TermsEnum[values.length];
    final long[] weights = new long[values.length];
    for (int i = 0; i < values.length; ++i) {
      subs[i] = values[i].termsEnum();
      weights[i] = values[i].getValueCount();
    }
    return build(owner, subs, weights, acceptableOverheadRatio);
  }

  /**
   * Create an ordinal map that uses the number of unique values of each {@link SortedSetDocValues}
   * instance as a weight.
   *
   * @see #build(IndexReader.CacheKey, TermsEnum[], long[], float)
   */
  public static OrdinalMap build(
      IndexReader.CacheKey owner, SortedSetDocValues[] values, float acceptableOverheadRatio)
      throws IOException {
    final TermsEnum[] subs = new TermsEnum[values.length];
    final long[] weights = new long[values.length];
    for (int i = 0; i < values.length; ++i) {
      subs[i] = values[i].termsEnum();
      weights[i] = values[i].getValueCount();
    }
    return build(owner, subs, weights, acceptableOverheadRatio);
  }

  /**
   * Creates an ordinal map that allows mapping ords to/from a merged space from <code>subs</code>.
   *
   * @param owner a cache key
   * @param subs TermsEnums that support {@link TermsEnum#ord()}. They need not be dense (e.g. can
   *     be FilteredTermsEnums}.
   * @param weights a weight for each sub. This is ideally correlated with the number of unique
   *     terms that each sub introduces compared to the other subs
   * @throws IOException if an I/O error occurred.
   */
  public static OrdinalMap build(
      IndexReader.CacheKey owner, TermsEnum[] subs, long[] weights, float acceptableOverheadRatio)
      throws IOException {
    if (subs.length != weights.length) {
      throw new IllegalArgumentException("subs and weights must have the same length");
    }

    // enums are not sorted, so let's sort to save memory
    final SegmentMap segmentMap = new SegmentMap(weights);
    return new OrdinalMap(owner, subs, segmentMap, acceptableOverheadRatio);
  }

  /** Explicitly enrolls builder peaks and retained packed storage in an embedded ledger. */
  public static OrdinalMap build(
      IndexReader.CacheKey owner,
      TermsEnum[] subs,
      long[] weights,
      float acceptableOverheadRatio,
      MemoryLedger ledger)
      throws IOException {
    return buildManaged(owner, subs, weights, acceptableOverheadRatio, () -> false, ledger);
  }

  static Managed buildManaged(
      IndexReader.CacheKey owner,
      TermsEnum[] subs,
      long[] weights,
      float acceptableOverheadRatio,
      BooleanSupplier cancelled,
      MemoryLedger ledger)
      throws IOException {
    if (subs.length != weights.length)
      throw new IllegalArgumentException("mismatched ordinal inputs");
    return buildManaged(owner, subs, weights, acceptableOverheadRatio, cancelled, ledger, true);
  }

  private static Managed buildManaged(
      IndexReader.CacheKey owner,
      TermsEnum[] subs,
      long[] weights,
      float acceptableOverheadRatio,
      BooleanSupplier cancelled,
      MemoryLedger ledger,
      boolean checkTask)
      throws IOException {
    if (subs.length != weights.length)
      throw new IllegalArgumentException("mismatched ordinal inputs");
    // Shared builds use the aggregate waiter signal: one cancelled caller cannot stop other
    // waiters.
    try (BuildBudget budget = new BuildBudget(subs.length, cancelled, ledger, checkTask)) {
      final SegmentMap segmentMap = new SegmentMap(weights);
      TermsEnum[] budgeted = new TermsEnum[subs.length];
      for (int i = 0; i < subs.length; i++) budgeted[i] = new BudgetedTermsEnum(subs[i], budget);
      Managed map =
          new Managed(new OrdinalMap(owner, budgeted, segmentMap, acceptableOverheadRatio));
      budget.retain(map);
      return map;
    }
  }

  private static final long BASE_RAM_BYTES_USED =
      RamUsageEstimator.shallowSizeOfInstance(OrdinalMap.class);

  /** Cache key of whoever asked for this awful thing */
  public final IndexReader.CacheKey owner;

  // number of global ordinals
  final long valueCount;
  // globalOrd -> (globalOrd - segmentOrd) where segmentOrd is the ordinal in the first segment
  // that contains this term
  final LongValues globalOrdDeltas;
  // globalOrd -> first segment container
  final LongValues firstSegments;
  // for every segment, segmentOrd -> globalOrd
  final LongValues[] segmentToGlobalOrds;
  // the map from/to segment ids
  final SegmentMap segmentMap;
  // ram usage
  final long ramBytesUsed;

  /** Only for {@link Managed}, which reads through its own stock map. */
  private OrdinalMap(IndexReader.CacheKey owner, long valueCount, long ramBytesUsed) {
    this.owner = owner;
    this.valueCount = valueCount;
    this.ramBytesUsed = ramBytesUsed;
    globalOrdDeltas = null;
    firstSegments = null;
    segmentToGlobalOrds = null;
    segmentMap = null;
  }

  /**
   * Here is how the OrdinalMap encodes the mapping from global ords to local segment ords. Assume
   * we have the following global mapping for a doc values field: <br>
   * bar -&gt; 0, cat -&gt; 1, dog -&gt; 2, foo -&gt; 3 <br>
   * And our index is split into 2 segments with the following local mappings for that same doc
   * values field: <br>
   * Segment 0: bar -&gt; 0, foo -&gt; 1 <br>
   * Segment 1: cat -&gt; 0, dog -&gt; 1 <br>
   * We will then encode delta between the local and global mapping in a packed 2d array keyed by
   * (segmentIndex, segmentOrd). So the following 2d array will be created by OrdinalMap: <br>
   * [[0, 2], [1, 1]]
   *
   * <p>The general algorithm for creating an OrdinalMap (skipping over some implementation details
   * and optimizations) is as follows:
   *
   * <p>[1] Create and populate a PQ with ({@link TermsEnum}, index) tuples where index is the
   * position of the termEnum in an array of termEnum's sorted by descending size. The PQ itself
   * will be ordered by {@link TermsEnum#term()}
   *
   * <p>[2] We will iterate through every term in the index now. In order to do so, we will start
   * with the first term at the top of the PQ . We keep track of a global ord, and track the
   * difference between the global ord and {@link TermsEnum#ord()} in ordDeltas, which maps: <br>
   * (segmentIndex, {@link TermsEnum#ord()}) -> globalTermOrdinal - {@link TermsEnum#ord()} <br>
   * We then call {@link TermsEnum#next()} then update the PQ to iterate (remember the PQ maintains
   * and order based on {@link TermsEnum#term()} which changes on the next() calls). If the current
   * term exists in some other segment, the top of the queue will contain that segment. If not, the
   * top of the queue will contain a segment with the next term in the index and the global ord will
   * also be incremented.
   *
   * <p>[3] We use some information gathered in the previous step to perform optimizations on memory
   * usage and building time in the following steps, for more detail on those, look at the code.
   *
   * <p>[4] We will then populate segmentToGlobalOrds, which maps (segmentIndex, segmentOrd) -&gt;
   * globalOrd. Using the information we tracked in ordDeltas, we can construct this information
   * relatively easily.
   *
   * @param owner For caching purposes
   * @param subs A TermsEnum[], where each index corresponds to a segment
   * @param segmentMap Provides two maps, newToOld which lists segments in descending 'weight' order
   *     (see {@link SegmentMap} for more details) and a oldToNew map which maps each original
   *     segment index to their position in newToOld
   * @param acceptableOverheadRatio Acceptable overhead memory usage for some packed data structures
   * @throws IOException throws IOException
   */
  OrdinalMap(
      IndexReader.CacheKey owner,
      TermsEnum[] subs,
      SegmentMap segmentMap,
      float acceptableOverheadRatio)
      throws IOException {
    // create the ordinal mappings by pulling a termsenum over each sub's
    // unique terms, and walking a multitermsenum over those
    this.owner = owner;
    this.segmentMap = segmentMap;
    // even though we accept an overhead ratio, we keep these ones with COMPACT
    // since they are only used to resolve values given a global ord, which is
    // slow anyway
    PackedLongValues.Builder globalOrdDeltas =
        PackedLongValues.monotonicBuilder(PackedInts.COMPACT);
    PackedLongValues.Builder firstSegments = PackedLongValues.packedBuilder(PackedInts.COMPACT);
    long firstSegmentBits = 0L;
    final PackedLongValues.Builder[] ordDeltas = new PackedLongValues.Builder[subs.length];
    for (int i = 0; i < ordDeltas.length; i++) {
      ordDeltas[i] = PackedLongValues.monotonicBuilder(acceptableOverheadRatio);
    }
    long[] ordDeltaBits = new long[subs.length];
    long[] segmentOrds = new long[subs.length];

    // Just merge-sorts by term:
    TermsEnumPriorityQueue queue = new TermsEnumPriorityQueue(subs.length);

    for (int i = 0; i < subs.length; i++) {
      TermsEnumIndex sub = new TermsEnumIndex(subs[segmentMap.newToOld(i)], i);
      if (sub.next() != null) {
        queue.add(sub);
      }
    }

    TermsEnumIndex.TermState topState = new TermsEnumIndex.TermState();

    long globalOrd = 0;
    while (queue.size() != 0) {
      TermsEnumIndex top = queue.top();
      topState.copyFrom(top);

      int firstSegmentIndex = Integer.MAX_VALUE;
      long globalOrdDelta = Long.MAX_VALUE;

      // Advance past this term, recording the per-segment ord deltas:
      while (true) {
        long segmentOrd = top.termsEnum.ord();
        long delta = globalOrd - segmentOrd;
        int segmentIndex = top.subIndex;
        // We compute the least segment where the term occurs. In case the
        // first segment contains most (or better all) values, this will
        // help save significant memory
        if (segmentIndex < firstSegmentIndex) {
          firstSegmentIndex = segmentIndex;
          globalOrdDelta = delta;
        }
        ordDeltaBits[segmentIndex] |= delta;

        // for each per-segment ord, map it back to the global term; the while loop is needed
        // in case the incoming TermsEnums don't have compact ordinals (some ordinal values
        // are skipped), which can happen e.g. with a FilteredTermsEnum:
        assert segmentOrds[segmentIndex] <= segmentOrd;

        // TODO: we could specialize this case (the while loop is not needed when the ords
        // are compact)
        do {
          ordDeltas[segmentIndex].add(delta);
          segmentOrds[segmentIndex]++;
        } while (segmentOrds[segmentIndex] <= segmentOrd);

        if (top.next() == null) {
          queue.pop();
          if (queue.size() == 0) {
            break;
          }
          top = queue.top();
        } else {
          top = queue.updateTop();
        }
        if (top.termEquals(topState) == false) {
          break;
        }
      }

      // for each unique term, just mark the first segment index/delta where it occurs
      firstSegments.add(firstSegmentIndex);
      firstSegmentBits |= firstSegmentIndex;
      globalOrdDeltas.add(globalOrdDelta);
      globalOrd++;
    }

    long ramBytesUsed = BASE_RAM_BYTES_USED + segmentMap.ramBytesUsed();
    this.valueCount = globalOrd;

    // If the first segment contains all of the global ords, then we can apply a small optimization
    // and hardcode the first segment indices and global ord deltas as all zeroes.
    if (ordDeltaBits.length > 0 && ordDeltaBits[0] == 0L && firstSegmentBits == 0L) {
      this.firstSegments = LongValues.ZEROES;
      this.globalOrdDeltas = LongValues.ZEROES;
    } else {
      PackedLongValues packedFirstSegments = firstSegments.build();
      PackedLongValues packedGlobalOrdDeltas = globalOrdDeltas.build();
      this.firstSegments = packedFirstSegments;
      this.globalOrdDeltas = packedGlobalOrdDeltas;
      ramBytesUsed += packedFirstSegments.ramBytesUsed() + packedGlobalOrdDeltas.ramBytesUsed();
    }

    // ordDeltas is typically the bottleneck, so let's see what we can do to make it faster
    segmentToGlobalOrds = new LongValues[subs.length];
    ramBytesUsed += RamUsageEstimator.shallowSizeOf(segmentToGlobalOrds);
    for (int i = 0; i < ordDeltas.length; ++i) {
      final PackedLongValues deltas = ordDeltas[i].build();
      if (ordDeltaBits[i] == 0L) {
        // segment ords perfectly match global ordinals
        // likely in case of low cardinalities and large segments
        segmentToGlobalOrds[i] = LongValues.IDENTITY;
      } else {
        final int bitsRequired =
            ordDeltaBits[i] < 0 ? 64 : PackedInts.bitsRequired(ordDeltaBits[i]);
        final long monotonicBits = deltas.ramBytesUsed() * 8;
        final long packedBits = bitsRequired * deltas.size();
        if (deltas.size() <= Integer.MAX_VALUE
            && packedBits <= monotonicBits * (1 + acceptableOverheadRatio)) {
          // monotonic compression mostly adds overhead, let's keep the mapping in plain packed ints
          final int size = (int) deltas.size();
          final PackedInts.Mutable newDeltas =
              PackedInts.getMutable(size, bitsRequired, acceptableOverheadRatio);
          final PackedLongValues.Iterator it = deltas.iterator();
          for (int ord = 0; ord < size; ++ord) {
            newDeltas.set(ord, it.next());
          }
          assert it.hasNext() == false;
          segmentToGlobalOrds[i] =
              new LongValues() {
                @Override
                public long get(long ord) {
                  return ord + newDeltas.get((int) ord);
                }
              };
          ramBytesUsed += newDeltas.ramBytesUsed();
        } else {
          segmentToGlobalOrds[i] =
              new LongValues() {
                @Override
                public long get(long ord) {
                  return ord + deltas.get(ord);
                }
              };
          ramBytesUsed += deltas.ramBytesUsed();
        }
        ramBytesUsed += RamUsageEstimator.shallowSizeOf(segmentToGlobalOrds[i]);
      }
    }

    this.ramBytesUsed = ramBytesUsed;
  }

  /**
   * Given a segment number, return a {@link LongValues} instance that maps segment ordinals to
   * global ordinals.
   */
  public LongValues getGlobalOrds(int segmentIndex) {
    return segmentToGlobalOrds[segmentMap.oldToNew(segmentIndex)];
  }

  /**
   * Given global ordinal, returns the ordinal of the first segment which contains this ordinal (the
   * corresponding to the segment return {@link #getFirstSegmentNumber}).
   */
  public long getFirstSegmentOrd(long globalOrd) {
    return globalOrd - globalOrdDeltas.get(globalOrd);
  }

  /** Given a global ordinal, returns the index of the first segment that contains this term. */
  public int getFirstSegmentNumber(long globalOrd) {
    return segmentMap.newToOld((int) firstSegments.get(globalOrd));
  }

  /** Returns the total number of unique terms in global ord space. */
  public long getValueCount() {
    return valueCount;
  }

  @Override
  public long ramBytesUsed() {
    return ramBytesUsed;
  }

  @Override
  public Collection<Accountable> getChildResources() {
    List<Accountable> resources = new ArrayList<>();
    resources.add(Accountables.namedAccountable("segment map", segmentMap));
    // TODO: would be nice to return the ordinal and segment maps too, but it's not straightforward
    //  because of optimizations.
    return resources;
  }
}
