# PW-0136 — Page-aligned `pread` expert-slot acquisition bound

- Status: completed
- Disposition: rejected
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

## Result

Every one of 24 trials performs exactly eight complete expert reads totaling
201,719,808 bytes, reconstructs the authenticated artifact hash, and preserves
all eight Metal buffer pointer identities and lengths. Every cold trial records
exactly 201,719,808 physical read bytes; every warm trial records zero. There
are no short-read retries.

Cold medians for 1/2/4/8 workers are `59.094`, `58.125`, `58.205`, and
`58.515` ms. The selected two-worker result misses the 47.7 ms continuation
bound by `10.425` ms, and all three of its trials (`58.094`, `58.125`, and
`58.151` ms) exceed the 57.723 ms trial ceiling. More workers do not raise
cold storage throughput. The result nearly reproduces PW-0108's 58.034 ms
three-command Metal-I/O median, identifying the unchanged internal SSD plus
201.7 MB payload as the common floor rather than either API's submission
policy.

Warm medians are `30.126`, `18.437`, `13.640`, and `13.632` ms; bounded
parallel copies help when the file is already resident, but that state cannot
satisfy the cold gate. The decision is
`reject_parallel_pread_for_internal_ssd_source_fp8`. Do not build the
source-FP8 slot-owned scheduler. Retain the eight-slot implementation as the
direct control for a future fidelity-qualified INT4 artifact, where the byte
premise changes materially.

Gate 8 passes across 29 snapshots at 79% minimum free memory, 417,251,328-byte
maximum peak RSS, 206,637,248-byte maximum physical footprint,
206,538,560-byte final footprint, zero swap growth or new throttled pages, and
stable protected-service PID sets.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0136/run-001.json`, SHA-256
`e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0136/analysis-001/manifest.json`,
SHA-256
`7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab`.
No endpoint TPS or measured throughput-model constant changes.
