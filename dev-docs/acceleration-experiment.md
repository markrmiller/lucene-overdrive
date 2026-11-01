<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# Lucene 10.5.2 acceleration experiment

Native validation uses JDK21 compilation and JDK25 execution. Select them independently
with `-Pcompiler.java.home=/usr/lib/jvm/java-21-openjdk-amd64` and
`-Pruntime.java.home=/usr/lib/jvm/java-25-openjdk-amd64`. Without the compiler override,
`runtime.java.home` keeps its existing behavior of selecting both compilation and execution.


This branch implements the Lucene side of
`/shared/design/solr-lucene-acceleration-design`, against release commit
`145d89cd30c38d25e39a0a2232cf00eee208b9cf`. It adds an optional pack runtime,
bounded memory ownership, source bridges and a separately built native reference
pack. Existing Lucene codecs and index formats are unchanged. These APIs and the
reference pack are experimental; correctness tests do not establish a speedup or
production qualification.

## Classpath installation

Solr and standalone Lucene use the same Nitrous artifact. The host distribution must
already contain this bridge; Solr's adapters belong in `solr-ref`, which consumes the
published `lucene-overdrive` Maven dependency. Adding Nitrous to an unmodified host
does not add missing dispatch hooks.

Nitrous `pack/bundle.py PACK QUALIFICATION NEW_JAR` produces a single JAR with
`META-INF/lucene-acceleration/pack.zip`. Put it on the host API's startup classpath
and restart. The runtime discovers the passive resource, checks the local JAR's
ownership and path permissions, extracts bounded contents into a private temporary
directory, verifies the existing pack contract, and reads its bundled qualification
data before preparing providers. Provider classes stay in their verified loader;
the outer JAR contains no duplicate Lucene or Solr classes. Discovery performs no
downloads. FFM still requires JDK25 and launcher native-access permission.

Default capacity is at most 64 MiB and one 128th of maximum Java heap; per-call
capacity is at most 8 MiB and one quarter of native capacity. Capacity limits do not
allocate their contents at startup. No pack means Java. Installing a pack does not
change storage policy, enroll unmeasured kernels, or widen qualified shapes.
`native.mode=off`, zero budgets, and `experimental.enabled=false` remain explicit
disable controls. All property names carry the `lucene.acceleration.` prefix.

An explicit `native.packPath` replaces classpath discovery. An explicit
`native.qualificationPath` replaces bundled profiles, including when it is empty
or invalid; it cannot silently restore bundled enrollment after an operator has
disabled it. Classpath packs carry policy matching their own manifest digest.
Removing the Nitrous JAR and restarting restores Java without reindexing.

Hosts that selectively prepare native input columns under current storage use the exact
`supports(capability, shape, parameter3, parameter4, consumerContext)` preflight.
It matches one complete deployment profile before materializing storage, without
reserving capacity, calling a native kernel, or updating call/fallback counters.
It is not an execution grant: the finished batch is checked again by `execute`.
The simpler capability/context support check is only a registration hint.


## Prepared-kernel qualification follow-up

The original byte-vector measurements below cover an explicitly owned
`OwnedByteBatchScorer`, raw mapped byte vectors, DOT_PRODUCT, 64 or 384 dimensions,
64 or 512 contiguous ordinals, and reusable output. Lucene's exact search scores in
batches of 64 (`VectorScorer.DEFAULT_BULK_BATCH_SIZE`); HNSW hops score at most 32
scattered rows; their mapped quantized gather route is separately qualified below. Create it for a leaf/field/query,
use `bulkScore`, then close it on its creating thread before releasing its reader
or request binding. It snapshots the query, admits bounded scratch once and
reuses it until close. Mapping pins, full validation, cancellation, runtime
counters, maximum reduction and output publication are included in the measured
path. Empty/singleton/small/scattered batches, other dimensions or metrics, and
unsupported mappings use the active Java scorer. Scalar raw-vector scoring remains Java. Native failures never replay or publish partial native output.

The current ownership/startup contract is `lucene-10.5.2-bridge-4`; packs for the
older bridge require a rebuild and fresh qualification before application use.

Enrollment now uses trusted deployment data selected with
`lucene.acceleration.native.qualificationPath`, rather than a CPU model string or a
compiled-in pack hash. The bounded, read-only TSV file has twelve required fields per row:
bridge build, manifest SHA256, negotiated ISA, exact capability ID, dimensions,
minimum rows, maximum rows, parameter slots 3 and 4, layout, maximum admitted
bytes, and evidence label. The byte bound includes all input, output, scratch,
parameter, and metadata spans, including the complete borrowed mapped field
region for gather calls; staged-byte accounting remains separate. Parameter slots contain an exact decimal long or `-`
when unused; layouts are `mapped`, `native`, or `buffer`. `native` and `buffer`
exclude mapped storage; mapped calls require an explicit `mapped` row. The bridge
classifies the actual input and workspace segments before dispatch. Each row binds the
immutable bridge/arithmetic profile and the exact reproducible pack; changing a
binary requires new measurements and a new row. Native capability/OS feature
negotiation still occurs before dispatch. An empty or invalid policy admits no
production kernels. The system minimum batch defaults to one, and the qualified
rows provide each operation's profitability threshold. Explicit global or
per-capability minimum-batch settings remain respected. Small and unqualified
shapes stay Java before native staging whenever their row count is below every
qualified profile. Qualification data must describe measured full consumer
boundaries, including journal and ownership overhead; prepared-kernel wins alone
are insufficient. Never copy rows from a different digest or machine feature
level.

An optional thirteenth field selects an exact consumer context. Omitted context
means `generic`. An explicit context is accepted only for
`docvalues.range.filter.words.exact.v1`, with values `generic`,
`solr.point.uncached.full-live.v1`, or `solr.frange.full-live.v1`. Empty, malformed,
unknown and non-RANGE explicit contexts reject the policy, including an explicit
`generic` field on another capability. Contexts are never wildcards. SYSTEM
execution matches both context and the complete physical/parameter profile;
context preflight alone admits no allocation or native call. The context is Java
dispatch data and changes neither native inputs nor the pack ABI. QUALIFICATION
comparisons bypass deployment-profile matching. The allowlist generator emits
field thirteen only when reviewed JSON explicitly supplies `consumerContext`;
it never invents a context or widens the measured shape.

The normal scalar-quantized scorer now consults this registry. Its mapped gather
scorer survives `forGraphTraversal()` and reaches existing HNSW bulk scoring;
unqualified shapes and scalar calls use Java. Raw float reductions and MaxSim
retain Java unless a separately measured exact profile is explicitly enrolled.
Missing packs retain Java. Implicit process startup fails softly into a Java-only
registry with a bounded rejection label when properties or required capabilities
are invalid; explicit config parsing and runtime constructors remain strict.

Enabled native execution creates a private fixed-size mmap crash journal under
`~/.cache/lucene-acceleration` (override with
`lucene.acceleration.native.statePath`). At most 256 synchronous calls record
kernel identity, capability, and output element count. Pack startup receives its
own breadcrumb before provider construction, library loading, and negotiation. The hot path performs no filesystem
syscalls or forced writes. Dead PID/start-time owners with active records produce
persistent per-pack/kernel quarantine on restart; clean calls clear their slots.
A startup crash quarantines the whole pack. Kernel identity includes its
capability ID, so exact-integer siblings remain independent. Quarantine files
retain bounded bridge, digest, ISA, capability and count metadata for inspection.
The journal is a process-crash breadcrumb, not a power-loss durability guarantee.
Known failures force the quarantine record. Startup refuses native execution if
the journal cannot be created securely, if records are malformed, or if its
bounded directory exceeds 1024 files. Operators may remove a quarantine record
only after investigating the failed binary, or install a newly qualified binary.
No document IDs, input bytes, query text, or reader paths are recorded.

Bridge4 also accepts a bounded borrowed span for typed heap arrays. A borrowed
input retains the existing three-argument ByteBuffer constructor and adds an
explicit `Input.borrowed` factory. The FFM adapter accepts an actual read-only
MemorySegment with exactly matching size, storage kind and live scope; raw
addresses and arbitrary opaque objects are rejected. Only MASKED_FACETS accepts
these inputs initially. Long word arrays and short/int ordinal columns pass as
critical downcall arguments without constructing a ByteBuffer or copying the
primary input. Structural Java validation stays O(1); the native kernel checks
ordinal bounds before output publication. Shadow comparison alone may copy
within its independent diagnostic budget. Heap calls above the short-work bound
decline before native entry.

Pack loading takes a private snapshot and re-verifies it against the frozen
manifest digest before constructing any provider or loading native code. Replacing
an artifact at the original path cannot change executed bytes after verification.
Compatible API/ABI minor additions and passive metadata are accepted. Unknown
kernels must explicitly declare `optional:true`; unknown required operations and
`requiredFields`, changed arithmetic contracts, and unsupported major versions
remain hard rejections. The native handshake still negotiates the host's minor
version and known capability contracts.

The bridge requires pinned native entries for execution: every input, output,
parameter, scratch, and metadata segment is a downcall argument, so the linker
holds every memory scope through both large ordinary calls and small critical
calls. Exact float reductions and MaxSim always use ordinary downcalls; their
byte length does not bound the expensive math work. Gather eligibility also
bounds rows times dimension rather than counting only ordinal/output bytes.
Every mapped segment also forces an ordinary downcall: even a short gather can
fault a cold page and block a critical call. Mixed heap/mapped batches decline
before native entry. The mapped OSQ scorer copies its small query into the
existing native call frame, or into its scoped native workspace for large batches,
so its mapped vector rows can use the ordinary pinned entry safely.
Old packs lacking these symbols decline safely. Reader-close races cannot
free native memory under a running call. Reader lifetime is still the caller's
responsibility outside a synchronous call.

`AccelerationRuntime.forQualification(...)` loads all candidate packs with the
existing verification, known-answer, permission and budget checks. It is solely
for explicit diagnostics. Public raw kernel/scratch access remains restricted to
that runtime. Existing native fixtures and legacy benchmarks opt into
qualification explicitly; their success does not qualify application dispatch.

The new measurement seams separate three costs:

* `native/c/benchmark.c` in the nitrous repository: unchanged C primitives with prepared inputs and reusable
  output, and a separate full C ABI phase on those same buffers. No timed staging,
  allocation, arena lifetime, checksum or oracle work. See
  native measurement instructions (nitrous `native/c/benchmark.md`).
* `PreparedAccelerationBenchmark` and `PreparedBitsetBenchmark`: prepared Java
  work versus prepared diagnostic downcalls. Native calls still include FFM
  adaptation and ABI validation, so this is not a pure C-kernel timing.
* `ReaderByteScoringBenchmark`: real indexed byte-vector ordinal lookup, staging,
  execution, maximum reduction and publication versus the existing Java bulk
  scorer. `PreparedByteScoring` copies the query once and reuses bounded,
  owner-thread scratch and caller-owned output. Reader ownership stays with the
  caller; the frame closes once after its query/trial lifetime. There are no
  per-vector allocations or arena closures. Bounded descriptor views remain
  per-batch costs and must be included in the result.

Compare matching dimensions, batch size, metric, ISA and input/output semantics.
Use separate processes for reference (`native.maximumIsa=baseline`) and optimized
selection, since a library's ISA mask is immutable. The benchmarks print actual
selected identities and reject a mislabeled reference/optimized ISA. The Java
vector provider remains active. Preserve raw samples and allocation rates; require
repeatable full-path gains with exact parity before adding any winning dispatch
shape. Only run reader-boundary measurements for prepared-kernel candidates that
win. If a realistic boundary does not win, remove that candidate from the
application path rather than tuning bridge overhead indefinitely.

On 2026-10-07, admitted measurements separated pure C, prepared FFM, staged
reader/bitset boundaries, direct mapped input, and final restricted dispatch. All
comparisons use two independent process/fork rounds, with warm reused buffers and
exact parity checks. The JMH control uses the active 256-bit Java vector provider.

| Operation | Java, us | Native, us | Decision |
|---|---:|---:|---|
| Prepared DOT, 64 dimensions x 512 rows: pure C | 4.65 | 9.11 | Kernel loses; reject |
| Prepared DOT, 384 dimensions x 512 rows: pure C | 20.11 | 17.47 | Investigate full boundary |
| Staged reader DOT, 384 x 512, contiguous | 28.97 | 34.40 | Reject staged path |
| Staged reader DOT, 384 x 512, scattered | 30.54 | 34.29 | Reject scattered path |
| Reusable staged AND, 65536 words | 17.06 | 38.69 | Reject bitset path |
| Direct mapped diagnostic DOT, 384 x 64 | 3.65 | 2.64 | Diagnostic win; remains Java dispatch |
| Direct mapped diagnostic DOT, 384 x 512 | 28.75 | 19.12 | Qualify restricted dispatch |
| Final dispatch DOT, 384 x 512 | 28.94 | 19.55 | Enable this shape only |

Final dispatch fork means were Java 28.60/29.29 us versus native 19.39/19.71 us.
Mean speedup was 1.48x, with approximately 36867 versus 1498 Java bytes allocated
per whole batch. Those gains include range/contiguity checks, reader pinning,
runtime descriptor validation, cancellation, counters, downcall, maximum and
publication. They exclude index creation and once-per-owner preparation/close.
The 64-row diagnostic shape was not measured through the final dispatcher and is
therefore not enabled. Prepared small word/64-dimensional kernels already lose;
large AND loses even at prepared FFM and at its realistic copy boundary. Those
kernels remain qualification-only. Raw results, commands, immutable measured JAR
images and source/pack manifests are retained in the published evidence run.
The initial timed dispatch admission attempt timed out without launching; its
queued attempt subsequently ran under the same exclusive host runner.

## Second native-kernel pass

A second pass on 2026-10-07 rewrote the slow native kernels, cut the fixed per-call
cost and widened mapped dispatch. Every comparison below is `PreparedKernelBenchmark`
or `ReaderByteScoringBenchmark` against the real Lucene Java path, two interleaved
single-fork rounds, cache-hot, on the same gm12-wsl host. Times are microseconds.

Root causes found by profiling and disassembly:

* Byte-vector scoring computed all four metric accumulators one row at a time. It now
  widens with `cvtepi8_epi16` + `madd_epi16`, scores 4 rows per pass while reusing
  the widened query, and vectorizes the exact float transform (`div_ps`, `sqrt_pd`,
  matching scalar SSE bits).
* `read_le` zeroed an 8-byte local and memcpy'd a narrower value into it, so every
  4-byte load became a store-forwarding stall (histogram and sorted intersection were
  10-50x slower than Java). Loads are now width-typed and always inlined.
* GCC 9 generic tuning split every unaligned 256-bit load/store into 128-bit halves.
  The build now passes `-mno-avx256-split-unaligned-load/-store`.
* Each FFM call built fresh segment views, a result slice and a result record
  (about 112 bytes per call; on this WSL2 host fresh heap pages also faulted). The
  bridge now reuses a checked view for a batch presented again; metadata is still
  rewritten through the liveness-checked segment on every call. A batch presented
  once keeps the previous path. Kernel-call allocation fell to under 1 byte.
* The default Java postings path paid a runtime lookup, checkpoint, eligibility check
  and counters on every block refill. `prefixSum` is now gated by a static
  capability check, restoring stock behavior when no prefix kernel is loaded.

New kernels: 27 (`DOCSET_INTERSECT_COUNT`, fused AND + popcount) and 28
(`DOCVALUES_RANGE_FILTER`, range test fused with a selection bitset, skipping empty
selection words). Kernel 16 is single-pass across all sets. Kernels 11, 12, 21 and
25 gained AVX2 paths: SIMD ordinal bounds checking; branchless aggregate whose
running sum and overflow flag follow the scalar wrapped prefix sequence exactly;
a prefix sum whose carried chain is one add per block; and vector ForUtil groups
with all residual lanes assembled in one word. All remain qualification-only.

| Kernel (pack v6) | Java 256 | Native 256 | Java 4096 | Native 4096 | Java 65536 | Native 65536 |
|---|---:|---:|---:|---:|---:|---:|
| RANGE_FILTER (28) | 0.204 | 0.099 | 4.863 | 0.366 | 240.2 | 8.57 |
| RANGE (8) | 0.191 | 0.102 | 4.711 | 0.707 | 238.8 | 10.5 |
| NIBBLE_UNPACK (22) | 0.311 | 0.066 | 5.244 | 0.137 | 85.07 | 3.43 |
| NIBBLE_PACK (10) | 0.076 | 0.070 | 1.184 | 0.181 | 17.45 | 2.17 |
| AGGREGATE (12) | 0.379 | 0.348 | 7.191 | 4.229 | 303.5 | 68.3 |
| CARDINALITY (6) | 0.106 | 0.107 | 1.617 | 0.671 | 25.61 | 9.65 |
| INTERSECT_COUNT (27) | 0.110 | 0.127 | 1.672 | 0.780 | 26.80 | 11.4 |
| MULTI_AND, 4 sets (16) | 0.131 | 0.117 | 2.188 | 1.482 | 41.89 | 25.6 |
| PREFIX_SUM (21) | 0.070 | 0.098 | 1.420 | 0.677 | 18.56 | 10.4 |
| DECODE (7) | 0.049 | 0.104 | 0.786 | 0.698 | 14.22 | 10.8 |
| U7_RESCORE rows (13) | 10.50 | 4.805 | 178.6 | 79.32 | 2099 | 1055 |
| HISTOGRAM (11) | 0.115 | 0.303 | 1.512 | 3.211 | 41.61 | 60.4 |
| INTERSECT_SORTED (15, scalar) | 0.529 | 0.736 | 13.90 | 11.90 | 468.8 | 471.8 |
| FOR_DECODE, one block (25) | 0.046 | 0.182 | | | | |

Remaining losses are structural. A native call still costs roughly 60-70 ns, so
operations Java finishes in under ~100 ns (DECODE, PREFIX_SUM and the counting
kernels at 256, one 256-value FOR block) cannot win individually. The histogram is
bound by scattered read-modify-writes that Java performs identically; branchless and
multi-table variants measured slower than the simple loop. Plain two-input AND over
large word arrays is memory-bandwidth bound and reaches parity; only fused forms
(27, 28, 16) win. Large facet and doc-values wins require kernels that decode packed
mapped data directly rather than heap arrays.

Widened mapped dispatch (`ReaderByteScoringBenchmark`, MAPPED_CONTIGUOUS, DOT_PRODUCT,
through `MEASURED_DISPATCH`, including pinning, validation, counters and publication):

| Shape | Java, pack v5 run | Native, pack v5 | Java, pack v6 run | Native, pack v6 |
|---|---:|---:|---:|---:|
| 384 dimensions x 512 rows | 28.68 | 5.58 | 29.95 | 5.51 |
| 64 x 512 | 12.41 | 2.62 | 14.12 | 2.62 |
| 384 x 64 | 3.66 | 1.24 | 3.81 | 1.33 |
| 64 x 64 | 1.79 | 0.90 | 1.31 | 0.97 |

All four shapes win (2.0-5.4x on the v5 run), so all are enrolled. The v6 run used
an interim bridge that cached a view for every batch; mapped dispatch creates a new
batch per call, so that cost about 80 ns per call. The committed bridge keeps the
original path for batches presented once. A clean re-measurement of the final bridge
was blocked by concurrent host load and is pending in the evidence run.

## Heap critical calls and removed hooks

Small calls (all buffers within 256 KiB) now use `Linker.Option.critical(true)` on
two new pack exports, `ap_execute_direct_v1` and `ap_execute_direct_reference_v1`.
They take each input's and the output's data pointer as arguments and substitute them
into private descriptor copies before the unchanged validation. A heap segment may be
passed only as a direct critical argument, never as a pointer inside a struct, so the
original two-argument entries cannot accept heap memory. Heap `ByteBuffer` inputs and
outputs now reach the kernel without staging copies; the manifest declares the
`HEAP_BUFFER` layout. A heap batch over 256 KiB, or one on a pack or JDK without the
direct entry, declines to Java before any native call. Metadata, parameters and
scratch remain native. Alias checks compare heap segments by array, not address.
Inputs stay byte-addressed, so `int[]`, `long[]` and `float[]` callers still stage.

Hooks that measured slower than the Java they replaced are removed: the postings
split/ForUtil decoder (it copied input byte by byte), the doc-values bulk-decode and
range wrappers (each decoded every value through `LongValues` into staging, the
work the Java path already does, before the kernel ran), the
`DocValuesStats` aggregate/statistics path, and the modern quantized scorer that
copied every row through `vectorValue` before scoring. The kernels themselves stay
for qualification and benchmarks. The postings prefix hook stays behind its static
gate.

## Reader-boundary prototypes: BM25 blocks and quantized-vector gather

Two prototypes were measured from a real mapped index rather than from decoded arrays.
Both are reachable only through an explicit qualification runtime, and ordinary runtimes
still select only the I8 kernel. Each passes native, sanitizer, guard-page and
differential fixtures, plus bitwise parity against Lucene on mapped indexes
(`TestBm25Block`, `TestOsqGather`). Results are means from JMH runs on gm12-wsl
(Ryzen 9 5900X, JDK 25) and are not portable claims.

**BM25 block scoring (kernel 29, `Bm25Block`).** For each block it reads norms in place
from the mapped norms slice and computes `w - w / (1 + freq * cache[norm])`, bit-identical
to `BM25Similarity`. The index has 100k documents. The dense term has 196 blocks of 512
documents and the sparse term has 12.

| Scope, term | Java µs | Native µs |
| --- | --- | --- |
| Norms and score, dense | 54.7 | 69.0 |
| Norms and score, sparse | 3.32 | 4.29 |
| Full term, dense | 144.9 | 166.7 |
| Full term, sparse | 6.21 | 7.88 |

The first kernel cost 128 µs. It wrote eight scalar cache loads to the stack and then read
them back as one 256-bit vector, which defeats store forwarding. Building the vector in
registers made it 3.5 times faster (1.52 to 0.43 ns per document in an isolated C
measurement). Two `vpgatherdd` were slower than either version. Java's JIT already runs
this loop at about 0.55 ns per document, so even with zero call cost the best case is about
1.27 times on scoring alone. Postings decoding dominates a full term, where the best case
is about 8 percent. **Verdict: not worth a hook.**

**Lucene104 OSQ gather (kernel 30, `AcceleratedFlatVectorsScorer.wrapQuantized`).** This
scores quantized rows in place from the mapped field region, by ordinal, with corrections
read from each row. The index has 50k vectors of 384 dimensions, and each operation scores
random batches of 32 (an HNSW expansion) or 64 (an exact-search block). The path went from
losing badly to winning in five steps:

1. **About 255 µs per batch for every encoding.** Profiling showed the time in
   `RequestMemoryScope.close`. `newScope()` uses a shared arena, and closing a shared arena
   forces a VM-wide thread handshake. The thread-confined `newCallScope()` (now public) has
   none. That brought the cost to about 4.6 µs.
2. **Per-thread call frames (`CallFrame`, `AccelerationRuntime.acquireFrame`).** Each frame is
   native storage admitted once per thread and reused by every synchronous call: 16 KiB of
   data, a parameter block, scratch and metadata.
   - Frames are not reentrant: a nested acquire returns null and the caller falls back to a
     call scope.
   - Frames sit outside task cancellation and are capped at 256.
   - A frame is freed when a new frame is created after its thread has exited.
   - Shutdown retires idle frames; a busy frame is retired when its owner closes it.
   - The scorer reuses its batch per frame and count, so the bridge's checked native view is
     reused too.
3. **Leaner `execute`.** The `HashSet` per call is gone, there is one end timestamp instead of
   two, and 7-bit and nibble queries are validated eight bytes at a time. The per-call
   counters (`CallCounters`) now live in one cell per thread instead of taking eight atomic
   updates. Shadow sampling counts each thread's calls.
4. **AVX2 kernel for the bit-plane encodings.** The C plane loop used the portable SWAR
   popcount. The new kernel uses `vpshufb` nibble-table popcounts and weights the four query
   planes as 8·c3 + 4·c2 + 2·c1 + c0 in bytes (at most 120), so each 32 or 16 bytes need only
   one SAD. The native differential now covers multi-chunk planes and all-ones inputs.
5. **Final roster.** Two interleaved process rounds on pack
   `3c44fa636719659835f4a237088679134fa502f5e19e05a2041647c4913c75a4`, in µs per batch:

| Encoding | Java (32) | Native (32) | Java (64) | Native (64) |
| --- | --- | --- | --- | --- |
| UNSIGNED_BYTE | 3.23 | 2.17 (1.49x) | 6.73 | 3.91 (1.72x) |
| SEVEN_BIT | 3.39 | 2.40 (1.41x) | 6.64 | 4.28 (1.55x) |
| PACKED_NIBBLE | 2.56 | 2.02 (1.27x) | 5.54 | 3.94 (1.41x) |
| SINGLE_BIT_QUERY_NIBBLE | 1.52 | 1.35 (1.12x) | 2.97 | 2.47 (1.20x) |
| DIBIT_QUERY_NIBBLE | 2.16 | 1.65 (1.31x) | 4.31 | 3.29 (1.31x) |

The native call is now about 65 percent of the batch. The kernel takes roughly half, and C
validation takes 13 to 18 percent, mostly the first cache miss on each randomly placed row.
That miss would otherwise land in the kernel, and validation must finish before any output
is written. Runtime admission, validation and counters are about 20 percent. **Verdict: a
1.1 to 1.7 times win at HNSW and exact-search batch sizes on this host.** The scorer stays
off by default. It is reachable only through an explicit qualification runtime until it has
a measured profile of its own, like `MeasuredByteProfile` for the I8 kernel.

The staged byte/float `BulkScorer` and the gated postings prefix hook also use
`newCallScope()` now. The quantized-vector writer keeps `newScope()`, because it runs once
per field flush and only reserves heap.

**BM25, rerun on the final pack** (µs): norms and score, dense, 54.8 Java vs 64.2 native;
sparse, 3.29 vs 4.44. Full term, dense, 148.8 vs 166.0. For the sparse full term, one Java
fork was bimodal at 12.7 to 19.7 µs; the other fork and every earlier run give 6.2 µs against
8.1 µs native. BM25 still loses in all four shapes, and the verdict stands.

The dispatch recheck on the same pack holds: I8 DOT_PRODUCT is 1.77x at 64 dimensions × 64
rows, 3.24x at 384 × 64, 5.41x at 64 × 512 and 5.60x at 384 × 512.

## Source coverage

| Design area | Lucene implementation and test seam |
| --- | --- |
| Doc01 ownership/governor | `MemoryLedger`, typed `MemoryStores`, request/worker/child scopes, checked growth and physical-close accounting |
| Doc01 scheduling/pressure | Weighted soft fairness, bounded waits, cancellation, foreground reserve, shrink-to-target and reclaim outside admission locks |
| Doc01 representations | Sorted integers, compressed deltas, dense heap and opt-in dense native `DocSetView` with operation/iterator leases and admitted algebra outputs |
| Doc01 actual owners | `LRUQueryCache` enrolled heap/native values, `OrdinalMap` build peak and retained packed heap, mapped arena capacity until physical unmap |
| Doc01 sharing | `GenerationCache` singleflight, generation invalidation, independently cancelled followers and reader-close eligibility |
| Doc02 startup/SPI | Frozen registry, exact keys/profiles, descriptors, bounded workspace/cost estimates, provider overrides, priority/ambiguity and known-answer qualification |
| Doc02 execution | Private output, checked schemas, synchronous caller-thread FFM, task cancellation, capacity-only fallback, quarantine and required readiness |
| Doc02 diagnostics | Per-capability provider/ISA, batch buckets, bytes, elapsed time, scratch peaks, fallback reasons, scope/drain blockers and bounded shadow mismatch records |
| Doc03 query vectors | Signed-byte batches, legacy 1–7-bit rescoring, all five modern LVQ encodings and matched-profile float batches |
| Doc03 indexing | Exact optimized scalar quantization, packing/unpacking/bitplane transforms, ordered centroid/statistics; flush/merge standard codec bytes preserved |
| Doc03 numeric/set/facet operations | Dense/sorted/multi-set algebra, cardinality, range, dense/sparse histogram, fused numeric aggregation and ordered floating statistics |
| Doc03 actual consumers | Vector scorers/writers and postings prefix sums; the bulk numeric, range, ForUtil, postings-split, `DocValuesStatsCollector` and modern quantized row-staging hooks were removed after measuring slower than Java |
| Doc03 conditional prototypes | Bounded decoded-matrix MaxSim preserving v1 maxima, strict complete-block LZ4 with unchanged Java dictionary/partial paths |
| Doc04 artifact operations | Optional pinned release builder, source/debug/export/SBOM evidence, passive dependency audit, offline verification and immutable install/rollback commands |
| Doc05 verification | Java/native/oracle fixtures, real codec/owner integration tests, sanitizer/differential/guard-page tests and fresh-JVM cross-mode index matrix |
| Doc05 measurement | Four total-boundary JMH variants for word algebra and byte batches; four Java-compute document-set storage candidates |

The ordinary Java vectorization provider remains the default and oracle.
The wrapper publishes arrays, match bits, scores and input cursor advances only
on successful completion. Candidate order, duplicate ordinals, empty-batch maximum,
query mutation and supplier isolation retain existing behavior. Partitioned bulk
calls use one request identity and independently drained frames; the caller's
output remains private until every frame completes.

Graph construction and HNSW traversal explicitly obtain the original Java scorer
through `RandomVectorScorer.forGraphTraversal()`. Scorer wrappers cannot silently
change graph traversal when bulk scoring is used there. Native scalar quantization
and packing are separate exact codec operations; enabling them changes no format.

Float qualification reproduces the active Java reduction/FMA profile. The scalar
Java profile admits the full bounded region. Panama's larger SIMD reductions and
mapped bulk reductions differ; those shapes select the original Java path. No
loose tolerance substitutes for raw-bit parity. Native AVX2 currently accelerates
word algebra and signed-byte/legacy quantized integer accumulation; per-capability
inventory reports scalar tiers for the remaining kernels.

SPI records contain no FFM types. Normal builds need no pack, C compiler or native
library. Provider artifacts cannot bundle another host API. Passive verification
checks exact bridge/API/ABI profiles, digests, permissions, class ownership,
platform minima and bounded ELF dependencies before loading provider code. Search
paths may contain unrelated profiler/CUDA directories, but cannot shadow approved
system dependencies. CPU/OS probes verify every configured CPU before ISA use; restricted or incomplete probes select baseline. Each library publishes one immutable mask, so later runtimes cannot change earlier dispatch identities.

Shadow execution is disabled by default. Its separate budget includes native clone
storage and a conservative checked Java-oracle materialization bound, including
jagged-array headers and empty multiset rows. Explicit managed HEAP admission shares
the request quota. Only primary results can publish; mismatches quarantine the
capability, fail the operation and retain bounded metadata without input contents.

## Memory and task ownership

`MemoryLedger` admits committed capacity plus peak reservations across HEAP,
NATIVE and MAPPED dimensions. Owner/core/generation/request and purpose records
track building, published, retired and pinned states. Growth retains old and new
capacity concurrently. Failed physical frees remain charged with retry handles.
An enrolled dimension stays enrolled when its target shrinks to zero.

Typed stores expose bounded accessors and frozen views without raw writable
buffers. `DocSetLease` covers complete operations and iterators; explicit close or
exhaustion releases it. Cache eviction retires ownership while active readers stay
charged. Abandoned Lucene cache iterators conservatively remain pinned until reader
core close. `OrdinalMap` preserves packed heap storage: attaching peak/retained
accounting and sharing is the design's first phase; flattening packed maps into
native longs would violate its conditional later-phase requirement. Mapped virtual
capacity is separate from resident/native capacity and stays charged until unmap.

Hosts bind an explicit `TaskContext` on every worker. Child scopes inherit owner,
quota and cancellation. Cancellation prevents allocation/entry/publication while
an in-flight synchronous call retains its buffers. Shutdown cancels and waits for
caller-owned scopes and external leases to drain; it never closes active call
buffers. Deadlines report remaining owners and charges rather than claiming a
successful drain. Default zero HEAP/MAPPED limits leave ordinary allocations
unenrolled; positive policies enroll the migrated owners. These managed counts
model owned references/capacity, not total JVM heap or process RSS.

## Build and run

This checkout uses Gradle 9.8. To compile the ordinary non-preview FFM bridge
and run the experiment, use build JDK 21 and runtime JDK 25:

```sh
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
RUNTIME_JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 \
bash gradlew :lucene:core:jar -Pgradle.ge=false

# in the nitrous checkout beside this one
pack/assemble.sh ../lucene-overdrive \
  ../lucene-overdrive/lucene/core/build/libs/lucene-core-10.6.0-SNAPSHOT.jar /absolute/fresh/reference-pack
```

The native kernels, provider and pack assembly live in the separate nitrous repository
(http://10.0.30.167:3300/markmiller/nitrous). Lucene builds and runs without it; a pack only
implements the interfaces defined here. Pass a pack to the native parity tests with
`-Ptests.acceleration.pack=/absolute/pack`.

The explicit pack build (Cargo drives the native build) runs optimized C fixtures and ASan/UBSan validation,
packages one provider and one target library, records compiler/source provenance,
license/notice and SBOM, and atomically publishes to a fresh destination. It cannot
overwrite an existing pack. The pack implements 33 operation schemas with portable C11, strict
floating-point ordering, a scalar reference export and verified AVX2 dispatch. Linux x86-64
has been exercised here; AArch64 packaging support is not platform qualification.
See nitrous `native/c/README.md` for parameter domains and ABI details.

Default configuration is Java only: no pack path, zero native budgets and
experimental loading disabled. All properties below use `lucene.acceleration.`:
Compute-related properties apply to explicit qualification runtimes. Ordinary
runtimes keep Java compute until measured full-path shapes are implemented.

| Suffix | Default | Meaning |
| --- | --- | --- |
| `native.mode` | `auto` | `off`, `auto`, or `required` |
| `native.packPath` | absent | Absolute single-pack directory or directory of packs |
| `experimental.enabled` | `false` | Permit a candidate pack in a qualification runtime |
| `memory.nativeBytes` | `0` | Managed native committed plus reserved byte limit |
| `memory.requestBytes` | `0` | Combined admission limit for one request owner |
| `memory.managedHeapBytes`, `memory.mappedBytes` | `0` | Explicit enrolled HEAP/MAPPED targets |
| `memory.ordinalBytes` | `0` | Optional ordinal partition within the enrolled HEAP allowance |
| `memory.coreBurstBytes` | `0` | Optional per-core hard burst ceiling |
| `memory.safetyHeadroomBytes`, `memory.processEnvelopeBytes` | `0` | Advisory process envelope diagnostics |
| `shadow.enabled`, `shadow.sampleEvery` | `false`, `1000` | Budgeted secondary sampling |
| `shadow.maximumBytes`, `shadow.mismatchLimit` | `0`, `16` | Secondary capacity and bounded record count |
| `native.provider.<capability ID>` | absent | Qualified provider ID override |
| `native.minimumBatch.<capability ID>` | absent | Per-capability threshold override |
| `native.minimumBenefitNanos`, `native.stagingNanosPerByte` | `0` | Optional conservative admission cost hints |
| `native.maximumIsa` | `auto` | `baseline` safely selects the scalar reference export |
| `native.minimumBatch` | `64` | Diagnostic bulk admission threshold; scalar scoring stays Java |
| `native.requiredCapabilities` | empty | Comma-separated exact capability IDs; required mode needs a nonempty set |
| `native.disabledCapabilities` | empty | Comma-separated exact capability IDs |
| `storage.policy` | `current` | `current` or `selectiveOffHeap` |

Native access must be enabled on the host FFM module: for a classpath launch use
`--enable-native-access=ALL-UNNAMED`; for a module-path launch use
`--enable-native-access=org.apache.lucene.core`. Qualification auto mode rejects a pack when
permission, runtime, ABI, checksums, qualification or budgets are unavailable.
Required mode fails startup if its declared set cannot be selected; ordinary
required runtimes reject unmeasured candidates even with a valid pack. `off` does
not inspect the pack filesystem or load provider/FFM code. Off-heap Java storage
requires its explicit budgets and JDK 25, but does not require native compute
access or a pack. Removing a pack and restarting needs no reindexing.

Compile nitrous `qualification/Qualification.java` with `javac --release 21`
against the built core JAR. Run it in fresh JVMs using `write`, `append`, `read`
or `operations`, followed by an absolute index path. It checks index portability,
raw score bits, ordinal mappings, primitive results, managed-memory draining and
actual native coverage when kernels are selected. It prints the actual Java
provider and FMA policy as well as startup and terminal inventories. Use
`--add-modules=jdk.incubator.vector` to include the existing Java vector provider
in comparisons. The `operations` probe explicitly creates a qualification runtime;
the index probes exercise ordinary Java selection. For example, qualify a pack with:

```sh
java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -Dlucene.acceleration.native.packPath=/absolute/fresh/reference-pack \
  -Dlucene.acceleration.experimental.enabled=true \
  -Dlucene.acceleration.memory.nativeBytes=67108864 \
  -Dlucene.acceleration.memory.requestBytes=8388608 \
  -Dlucene.acceleration.native.minimumBatch=1 \
  -cp lucene/core/build/libs/lucene-core-10.6.0-SNAPSHOT.jar:PROBE_CLASSES \
  Qualification operations /absolute/index
```

nitrous `qualification/qualification.sh CORE_JAR PACK_DIRECTORY FRESH_OUTPUT`
runs the restart and index-portability matrix and retains one raw log per fresh
JVM. Set `JAVA_HOME` to JDK 25; optionally set `JAVA21_HOME` to exercise the older
runtime's Java fallback. The output directory must be fresh.

## Validation and integration boundary

Core tests cover pack verification and ambiguity, known-answer qualification,
required and auto failures, quarantine, malformed results, exact scorer behavior,
supplier ownership, decoder publication, concurrent admission, cancellation,
retirement and failed reclamation. Native tests include seeded differential
fixtures, finite bounds, guard-page tails, concurrent calls and sanitizers. Test
fault libraries are compiled separately; production libraries contain no runtime
fault switch. `tests.acceleration.pack` supplies a built pack to native JUnit
cases; absent that property, native-only cases are skipped explicitly.

Solr configuration translation, transaction-log ownership, Solr cache/facet/ordinal
call sites, distributed merge semantics, readiness and deployment coordination
remain in the Solr repository. The neutral SPI, memory helpers, MaxSim and facet
operations supplied here can serve that adapter without importing Solr into Lucene.

`boundary-benchmark.sh` retains independent fork samples and seeded variant order
for current Java, fused Java layout/allocation, native reference and native ISA
execution. Both native measurements include dispatch, validation, staging and
publication. `DocSetStorageBenchmark` compares sorted/compressed/dense/native
backing with Java computation. Record uncertainty and negative results; these
cache-hot primitives cannot establish complete-query/service gains. Promotion,
owner-approved thresholds, long-lived canaries, cold mmap/concurrent service
workloads and qualification of additional hardware remain separate evidence gates.
The APIs, source integrations and experimental harness are testable now; defaults
remain Java/current-storage and no promotion is implied.
Use prepared-kernel and reader-boundary qualification (nitrous `qualification/prepared-benchmark.md`)
for the follow-up that separates prepared computation from staging and tests bounded reuse.

The profiling followup identified shared-arena close handshakes as 98.39% of small
vector-call wall samples and 92.16% of large vector-call samples. Synchronous
`BlockOperations` calls now use confined arenas; ordinary worker and cache storage
remain shared. Confined allocation, worker pins, child scopes and physical cleanup
stay on the creating thread, while cancellation may arrive from other threads.
Shadow comparison children inherit confinement and preserve their independent
budgets and parent pins. `retryFailedScratchReclamations()` reaches retained private
call scopes; a failed confined close must be recovered on its owner thread before
that thread exits. A foreign retry cannot refund or force-free its storage, and a
lost owner leaves the charge visible and prevents a successful shutdown drain.

Two interleaved forks per shape using the same AVX2 pack show the ownership change
reducing the small vector batch from 313.912 to 4.682 microseconds and the largest
from 345.731 to 36.619 microseconds. Current Java scores are 0.638 and 22.303
microseconds respectively. All seven measured native total-call shapes still lose
to Java. After the change, the largest vector profile attributes 46.59% of wall
samples to native vector computation and 34.84% to staging; the largest docset
attributes 43.39% to staging and 12.86% to output copying. These are inclusive
sample shares, not isolated kernel timings. No default or promotion threshold has
changed. Raw measurements, exact benchmark images, profiles and source identities
are retained in the separate profiling followup evidence bundle.
