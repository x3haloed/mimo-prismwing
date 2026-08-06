# PW-0108 — Metal-I/O routed-layer acquisition bound

- Status: completed
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: `81638393be2b7b0ebeddcfdf73e2e05e5a0d738d`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0101 layer-4 oracle
  `9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d`;
  PW-0106 artifact
  `fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21`
- Hardware/runtime: Apple M1 shared 16 GiB host; internal SSD; macOS 26.6;
  macOS 26.5 SDK; PW-0106 page-stable artifact
- Related records: PW-0105, PW-0106, PW-0107

## Hypothesis and causal mechanism

PW-0107 proves that ordinary no-copy command aggregation leaves a cold median
96.001 ms in two synchronous waits around only 8.320 ms of GPU execution. The
next hypothesis is narrower than “async is faster”: Metal's dedicated I/O
queue can acquire the 201,375,744 selected source bytes into bounded reusable
Metal buffers quickly enough that an independent compute queue could overlap
acquisition of expert or tile `n+1` with execution of `n`.

Apple documents that Metal fast resource loading uses dedicated I/O command
queues to load file ranges directly into Metal resources and synchronizes them
with compute through shared events. The installed SDK exposes concurrent and
serial `MTLIOCommandQueue` modes, bounded commands in flight, `loadBuffer`,
status/error reporting, and shared-event wait/signal operations on this host.
The M1 supports the required fast-resource-loading feature family. Sources:
[Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading),
[Metal I/O command buffers](https://developer.apple.com/documentation/metal/mtliocommandbuffer),
and [Apple's Metal 3 resource-loading session](https://developer.apple.com/videos/play/wwdc2022/10104/).

## Construction contract

Capability and preservation: use the identical authenticated PW-0106 artifact
and its 48 selected gate/up/down weight and scale records. Do not change tensor
bytes, shapes, scales, routes, arithmetic, repair, reduction, or scatter. Phase
A is an acquisition/integrity bound, not an inference endpoint or TPS result.

Causal path: create a real `MTLIOFileHandle` for the artifact, a real
`MTLIOCommandQueue`, and page-aligned shared Metal destination buffers. Encode
the manifest-authorized file ranges with `loadBuffer`; commit, wait, fail closed
on any non-complete status or error, and compare every loaded record byte for
byte with its authenticated mapped source after timing. Record creation,
encoding, commit, wait, verification, release, process activity, and Gate 8
separately.

Topology: add one Rust Metal-I/O authority next to the existing Metal compute
runtime. Do not add a generic scheduler, background service, compatibility
layer, or alternate artifact schema. Raw Objective-C selectors are acceptable
where the pinned `metal` crate lacks Metal-I/O wrappers, but selector
availability, returned object identity, status values, ownership, and errors
must be checked explicitly.

Embodiment depth: compare one serial I/O command buffer against two and three
concurrent command buffers over disjoint, manifest-derived record partitions.
The total destination capacity may not exceed the 201,719,808-byte layer
artifact plus page-alignment overhead. Buffers and file handles must live until
all commands complete and be released at each trial boundary. Phase A may not
execute projections, add compression, use CPU `pread`/copy as a fallback,
retain a route cache, or construct a full-bank artifact.

## Verification and measurement protocol

Before the real transfer, add deterministic tests for manifest-range
partitioning, nonoverlap, destination offsets, byte totals, alignment, status
decoding, unavailable selectors, and non-complete/error failure. Add a small
real-device probe that loads nonzero file offsets into distinct output slots
and proves exact bytes with both serial and concurrent queues.

Run at least three cold and three genuine warm trials for each one-, two-, and
three-command-buffer configuration. Reverse configuration order each
repetition. Cold preparation invalidates the artifact before each trial; warm
preparation explicitly prefaults once and does not invalidate. Hashing and
byte comparison occur after the timed transfer interval and are reported
separately. Record actual physical reads, page-ins, signed faults, CPU time,
queue configuration, commands in flight, encoded records/bytes, complete
status, transfer wall, integrity wall, buffer capacity, RSS, memory pressure,
swap, throttling, release, and protected-service health.

The PW-0107 candidate cold median is 115.446542 ms, so its 2x continuation
threshold is 57.723271 ms. Existing identical CPU staging and scatter consume
about 10 ms, and the 8.320 ms GPU interval can be overlapped but not erased.
Therefore a Metal-I/O acquisition median above 47.7 ms cannot reach the gate
even under ideal GPU overlap. This is the predeclared physical continuation
bound, not an observed-data adjustment.

## Gates

- **Availability:** the real Apple M1 device creates the I/O file handle and
  queue without fallback; every command completes with no Metal error.
- **Integrity/accounting:** every one of 48 records is loaded exactly once,
  ranges do not overlap, encoded bytes equal 201,375,744, buffer capacity stays
  bounded, and every destination byte matches its authenticated artifact
  source. Serial and concurrent configurations produce identical hashes.
- **Physical continuation:** at least one concurrent configuration has a cold
  median transfer wall at or below 47.7 ms, no cold trial exceeds the PW-0107
  C3 wall for the corresponding repetition, and warm transfer does not regress
  relative to serial Metal I/O. Only then may Phase B build the shared-event
  I/O/compute arena pipeline.
- **Safety:** every Gate 8 boundary passes; no swap growth, new throttled pages,
  protected-service loss, failed release, or unexplained resident retention.

If availability, integrity, or safety fails, reject this embodiment. If the
47.7 ms physical bound fails, reject Metal I/O overlap as a path to the frozen
2x layer gate on this SSD and artifact; do not build Phase B merely because the
API works. A faster external device would be a new named hardware condition,
bounded by the $500 cap. If the bound passes, freeze a separate Phase B
contract before combining I/O and compute.

## Result

The release implementation at
`ea83a9b4d9a46379ad4d6b4a9932934b03a0ab8f` created real Metal-I/O file
handles, serial/concurrent queues, command buffers, and shared destination
buffers on Apple M1. Its two-command exact-offset probe loaded distinct nonzero
file ranges into distinct output slots and reproduced exact bytes in 0.872 ms.
The 54-test Rust suite and strict Clippy pass. No CPU-read fallback exists.

All 18 trials load every one of 48 manifest records exactly once. Encoded
tensor bytes close at 201,375,744 into a 201,719,808-byte bounded destination;
all cold trials record exactly 201,719,808 physical read bytes, all warm trials
record zero reads and page-ins, and every trial produces the identical
record-stream hash
`f31aed3aa8b65ae938968d48a61134cc0b6013a51f9798bf31c9460166094927`.
All command statuses are complete with no Metal errors.

Timed acquisition distributions are:

| State / command buffers | Trial walls (ms) | Median (ms) | Speedup vs serial |
| --- | --- | ---: | ---: |
| Cold / 1 | 71.912, 72.875, 74.445 | 72.875 | control |
| Cold / 2 | 58.893, 59.094, 59.553 | 59.094 | 1.233x |
| Cold / 3 | 57.725, 58.034, 58.262 | 58.034 | 1.256x |
| Warm / 1 | 30.179, 30.522, 30.263 | 30.263 | control |
| Warm / 2 | 17.496, 19.045, 18.790 | 18.790 | 1.611x |
| Warm / 3 | 14.942, 14.782, 14.754 | 14.782 | 2.047x |

Concurrent Metal I/O is real and useful, but the best cold median exceeds the
predeclared 47.7 ms physical continuation bound by 10.334 ms. It remains below
the corresponding PW-0107 complete-layer wall in every trial and has no warm
regression, but those secondary gates cannot compensate for the failed
mathematical bound. About 10 ms of unchanged CPU staging/scatter remains after
perfect I/O/GPU overlap, so a 58.034 ms acquisition floor cannot reach the
57.723 ms complete-layer 2x gate. Destination initialization costs another
roughly 15 ms per fresh allocation, although reusable Phase-B arenas could
amortize it; post-timing full integrity scanning costs roughly 577--591 ms and
is explicitly diagnostic rather than acquisition wall.

Gate 8 passes with 78% minimum free memory, 419,610,624-byte peak RSS,
10,503,936-byte final physical footprint, zero swap growth, zero new throttled
pages, and stable protected services. The raw report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0108/trials-001/report.json`
hashes to
`6f7d816b4f39c00b967642bdf300e7baea8563a5fca593ab5d0943b5df047d68`.
The clean analyzer at
`c48ded6beb46b95d093e8a87580dc9f4de1ff90e` emitted `analysis-001.json`, hash
`5281fd36c06e2a2e5767918bbb63f0fe33cbec4a1478b4281806d6fdf56ac43d`.
The updated throughput model hashes to
`a3b7c5578183af2727905ce5bdd058fa68a91350b715c6a8a00c0d469bf1fd33`.

## Decision

Reject the internal-SSD Metal-I/O overlap embodiment before Phase B. Do not
build shared-event arenas merely because the API and concurrent speedup work:
the measured acquisition floor cannot clear the frozen complete-layer gate.
Retain the exact loader and three-buffer result as a hardware/storage control.
A faster named storage configuration is a new conditional branch under the
$500 cap; reducing exact executable bytes before I/O is a distinct embodiment
and changes this bound. No full-bank artifact, endpoint walk, or L3 arithmetic
promotion is authorized.
