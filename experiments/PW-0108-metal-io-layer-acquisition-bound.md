# PW-0108 — Metal-I/O routed-layer acquisition bound

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: pending
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

Not yet executed.
