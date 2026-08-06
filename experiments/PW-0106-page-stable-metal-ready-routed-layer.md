# PW-0106 — Page-stable Metal-ready routed layer

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: pending
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0101 layer-4 oracle
  `9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d`
- Hardware/runtime: Apple M1 shared 16 GiB host; verified internal-SSD
  checkpoint; source-FP8 Metal path from PW-0099 through PW-0105
- Related records: PW-0039, PW-0040, PW-0042, PW-0049, PW-0097 through
  PW-0101, PW-0105

## Shared construction contract

Capability and preservation: execute PW-0101's exact real layer-4 routed-row
fixture with the same source FP8 weights and scales, input, route IDs and
weights, dynamic activation quantization, Metal projection kernel, BF16 staging,
sparse repair, SwiGLU, weighted reduction, and final residual. A runtime layout
may change storage and installation but not tensor bytes, arithmetic, ordering,
or the existing L3 candidate's declared numerical behavior. Exact source ranges
and the PW-0101 oracle remain authoritative; the rejected layer-final result
must be reproduced rather than hidden or relaxed.

Causal path and risk frontier: PW-0105 attributes 37.802 seconds of a
40.561-second routed token to repeated source tensor/scale validation, page
acquisition, and a global `MS_INVALIDATE`/`MADV_DONTNEED` walk over all 17
checkpoint mappings after every expert. Source-to-`MTLBuffer` copying accounts
for only 0.773 seconds and synchronous waits for 0.816 seconds. The least-proven
boundary is therefore a prevalidated, stable, layer-scoped mapping through
copied and no-copy Metal bindings, not the projection arithmetic.

Topology: introduce one versioned L1 runtime-artifact authority and extend the
existing Rust/Metal executor with explicit installation modes. Do not create a
parallel model scheduler or kernel. The artifact builder, verifier, and
executor must share one manifest schema; every tensor record binds its exact
source tensor name, shape, dtype, source shard and byte range, source SHA-256,
artifact offset and length, artifact SHA-256, and padding. Unknown revisions,
routes, sizes, alignments, hashes, overlaps, or trailing bytes fail closed.

Embodiment depth: the first artifact contains only the eight routed experts in
the frozen layer-4 fixture. For each expert, gate, up, and down weight and scale
payloads are stored losslessly in sequential page-aligned regions. The manifest
and records are immutable, create-new files external to Git. An artifact-wide
mapping remains live for one routed-layer transaction and is released at its
boundary. This experiment may compare `new_buffer_with_data` against
`new_buffer_with_bytes_no_copy` views of that same mapping. It may not yet add
Metal I/O, fused projections, argument buffers, indirect command buffers,
asynchronous arenas, a route cache, new codecs, or changed arithmetic.

Project constraints: the M1 host has 16 GiB shared memory. Run only one Metal
trial at a time, retain Gate 8 phase-level RSS, memory-pressure, swap,
throttling, buffer-release, and protected-service-health stops, and release the
mapping before final safety readback. The selected artifact is expected to be
about 202 MB including alignment. A complete 47-layer, 256-expert source-FP8
bank would be about 303 GB (282.6 GiB) before manifest overhead, not 159 GiB;
do not build it until this gate justifies that storage and construction time.

## Artifact and device-probe contract

At construction time, derive the selected experts from the authenticated
PW-0101 oracle rather than a handwritten list. Use the host VM page size as the
minimum alignment. Require every Metal-bound weight and scale offset and every
no-copy mapping base/length to satisfy the documented alignment contract. Fill
padding deterministically with zero and hash both payloads and the complete
artifact. Reopen and verify the completed artifact in a fresh mapping before
execution; construction timing is installation evidence, never inference
timing.

Before timing the real layer, prove on tiny deterministic files that the
builder/verifier detects one-byte payload mutation, padding mutation, truncated
data, wrong source identity, overlapping offsets, bad alignment, and unknown
schema. Then perform a real-device probe that binds a page-aligned mapped
region through Metal without copying, dispatches a deterministic read kernel,
and verifies exact output. Failure of a read-only mapping may justify a
precisely named private writable mapping control, but that would not count as
zero-copy file-backed execution.

## Interleaved measurement contract

Measure these three variants on identical layer-4 input and route order:

1. **C0 copied/global-release control:** current verified safetensors views,
   fresh copied source buffers, and the existing per-expert global checkpoint
   invalidation. This preserves PW-0105's physical baseline.
2. **C1 artifact/copied:** one verified page-stable artifact mapping retained
   for the complete routed layer, fresh copied source buffers per projection,
   and one layer-boundary artifact release.
3. **C2 artifact/no-copy:** the identical retained artifact mapping exposed to
   Metal through page-aligned no-copy buffers, with the same projection count,
   waits, CPU staging, arithmetic, and one layer-boundary release.

Run at least three cold trials per variant in interleaved order, with the order
rotated across repetitions. Before each cold trial, invalidate only that
variant's relevant file pages; record whether the invalidation request succeeds
and physical disk-read/page-in counters. Also run at least three warm trials
without invalidation. Do not silently discard any trial. Record complete layer
wall, tensor authority validation, mapping/open, page acquisition, source
installation, allocation, encoding, commit, wait, GPU interval, readback, CPU
staging/repair/SwiGLU/scatter, release, bytes moved, RSS, physical footprint,
memory pressure, swap, and service health. Report medians and every raw trial;
component intervals are diagnostic and layer wall is the performance authority.

## Gates

- **Artifact validity:** every selected source tensor byte and manifest identity
  matches the verified checkpoint; complete artifact and per-record hashes pass
  after a fresh reopen. Mutation tests fail closed.
- **Device validity:** the real Apple M1 no-copy probe returns exact bytes and
  releases without a safety failure. If unavailable through the pinned Metal
  bindings or mapping contract, record a rejection rather than relabeling a
  copy as no-copy.
- **Correctness:** all variants preserve selected expert IDs/order, route
  weights, expert gate/up/SwiGLU/down diagnostics, routed residual, final
  residual, sparse-repair counts, and source-byte ledger relative to PW-0101.
  The known source-exact failure remains visible.
- **Performance:** continue the Metal-ready artifact branch only if C1 or C2
  achieves at least 2.0x median cold complete routed-layer speedup over C0, and
  no cold candidate trial regresses. Attribute C1 versus C0 to stable layout
  and page lifecycle; attribute C2 versus C1 to no-copy binding. Warm results
  cannot satisfy the cold gate.
- **Safety:** every Gate 8 snapshot passes; no swap growth, new throttled pages,
  protected-service loss, failed final release, or unexplained resident-memory
  retention. Any safety stop aborts the remaining trials.

Passing this component gate promotes only construction of a broader routed
layer transaction experiment. It does not promote a runtime default, accepted
TPS, full-bank artifact, Metal I/O path, fused layer kernel, or the rejected L3
arithmetic. If C1 fails 2x, kill full artifact expansion. If C1 passes but C2
adds less than 10% cold layer improvement, retain stable layout/lifecycle and
deprioritize no-copy until a fused transaction removes the remaining small
buffer boundaries.

## Result

Not yet executed.
