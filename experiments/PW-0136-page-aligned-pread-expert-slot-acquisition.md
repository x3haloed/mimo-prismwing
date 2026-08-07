# PW-0136 — Page-aligned `pread` expert-slot acquisition bound

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-07
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0106 artifact
  `fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21`;
  PW-0106 artifact manifest
  `40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8`
- Hardware/runtime: Apple M1 shared 16 GiB; internal SSD; native Rust,
  Darwin `pread`, page-aligned shared allocations, and Metal no-copy buffers
- Exactness: L1 acquisition/layout only; no arithmetic or model change
- Related records: PW-0105 through PW-0108, PW-0111, PW-0129, PW-0135

## New premise

PW-0106 tested fault-driven cold mmap/no-copy and PW-0108 tested Metal I/O into
shared destinations. Neither tested the explicit-read embodiment independently
selected by TurboFieldfare and Swiftlet: one fixed-stride expert miss becomes
one `pread` into a bounded, reusable, page-aligned allocation already wrapped
by Metal.

TurboFieldfare reports that this distinction is causal on its smaller Gemma
expert payloads: cold mmap took 9.88 ms per expert versus 2.79 ms for `pread`,
and its streaming simulator reached about 0.50 versus 3.97 tok/s. That result
does not transfer numerically to MiMo. Prismwing's selected source-FP8 layer is
201,719,808 bytes—about 25,214,976 bytes per expert—and has no resident shared
expert with which to hide misses. Test the real Prismwing bytes before adopting
the scheduler.

## Compression-depth contract

Capability invariant: acquire every byte of the authenticated PW-0106 layer-4
artifact into eight expert-scoped Metal-visible slots without changing bytes,
expert identity, order, padding, or later computational meaning.

Embodiment boundary: replace demand paging or Metal I/O with native Darwin
file-descriptor reads, bounded host threads, `posix_memalign`, and
`newBufferWithBytesNoCopy`. The experiment may specialize for macOS/ARM64 and
the fixed artifact schema. It may not change weights, precision, arithmetic,
routes, artifact contents, storage hardware, or build a decode scheduler.

Project constraints: allocate exactly eight reusable slots once, remain below
1 GiB candidate capacity, use only the internal SSD, enforce normative Gate 8,
and keep installation verification outside transfer timing. No full bank,
cache-policy experiment, endpoint walk, or hardware purchase is authorized.

## Frozen mechanism and measurements

Open and fully verify the existing artifact once. Derive, rather than assume,
eight equal contiguous expert extents from the manifest: each extent must
contain exactly gate/up/down weights and scales for one selected expert, cover
the file without overlap or gaps, and be page aligned. Allocate eight
page-aligned slots once, wrap each through Metal without a second allocation,
and prove the Metal buffer exposes the same base address and complete length.

For thread limits 1, 2, 4, and 8, issue exactly one full `pread` per expert.
Workers may claim distinct experts dynamically but may never share a slot or
publish an expert as valid before its complete read returns. The timed transfer
interval begins immediately before worker launch and ends after every worker
joins. Allocation, buffer wrapping, artifact verification, page invalidation,
integrity hashing, and Gate 8 observation remain outside that interval and are
recorded separately.

Run three cold and three warm trials for every thread limit. Rotate variant
order across repetitions. Before every cold trial, apply the same successful
`MS_INVALIDATE` plus `MADV_DONTNEED` artifact invalidation used by PW-0106 and
record physical reads/page-ins. Before warm trials, prefault the artifact once
outside timing and perform no invalidation. Preserve every trial.

After every transfer, hash the eight complete slots in artifact order and
require the PW-0106 artifact SHA-256. Record transfer wall, per-expert read
walls, short-read retries, bytes requested/returned, worker count, physical
reads, page-ins/faults, CPU time, allocation capacity/alignment, buffer pointer
identity, integrity time, RSS, physical footprint, memory pressure, swap,
throttling, and protected services.

## Gates

- **Authority:** artifact and manifest identities match PW-0106; eight derived
  expert extents cover exactly 201,719,808 bytes and preserve selected order.
- **Integrity:** every trial performs eight disjoint complete reads totaling
  201,719,808 bytes; no short read reaches publication; the concatenated slot
  hash equals the authenticated artifact; all eight Metal buffers retain exact
  pointer identity and length.
- **Cold validity:** the selected candidate's three cold trials each record at
  least 95% of artifact bytes as physical reads. Otherwise the result is not a
  cold acquisition claim even if its wall is low.
- **Physical continuation:** select thread count by cold median only. Continue
  to a separately frozen slot-owned I/O/Metal pipeline if its cold median is at
  most 47.7 ms, none of its cold trials exceeds 57.723 ms, and its warm median
  does not regress against serial warm `pread`. This is PW-0108's unchanged
  mathematical bound, not a new endpoint threshold.
- **Safety:** all Gate 8 boundaries pass; candidate capacity stays below 1 GiB;
  no swap growth, new throttled pages, protected-service loss, failed final
  release, or unexplained resident retention occurs.

A pass authorizes only a new experiment combining protected slot ownership,
coarse hit/miss execution, and the retained one-barrier layer mechanism. That
successor must never refill a GPU-owned slot and must defer cache shrink to a
token boundary. It should test pending-MoE orchestration, not per-expert
completion launches or monolithic fusion.

A failure rejects parallel `pread` for the unchanged internal-SSD source-FP8
payload. Preserve it as a control for a future approximately 53%-sized INT4
artifact if that numerical branch independently passes fidelity. Report zero
accepted tokens, `A=0`, no endpoint timing, and no TPS.
