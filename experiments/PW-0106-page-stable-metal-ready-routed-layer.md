# PW-0106 — Page-stable Metal-ready routed layer

- Status: complete
- Disposition: conditional
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract `caf142f60ada9f689cd4fab28d28f05407340050`;
  artifact/runtime foundation `e4366463b72705d5e459eef787b287574041cb57`;
  canonical benchmark `ec1d2f2b42532a3e870218d0b47021a8525a45b6`;
  clean trees
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

The L1 builder emitted 48 independently page-aligned gate/up/down weight and
scale records for the eight authenticated layer-4 experts. The immutable
201,719,808-byte artifact hashes to
`fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21`;
its manifest hashes to
`40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8`.
Every record was rebound after a fresh reopen to its verified checkpoint tensor
name, shard hash, absolute source range, dtype, shape, byte count, and tensor
hash. Construction took 1,491.426 ms and fresh verification 1,155.739 ms.
Those are installation measurements, not layer latency. The build report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0106/artifact-001/build-report.json`
hashes to
`ea41fc741f85455b43d9c1b289540e6a5682d128e21b1c8aeffc60429b938b09`.

The real Apple M1 probe successfully wrapped the page-aligned read-only file
mapping through `new_buffer_with_bytes_no_copy`, dispatched a byte-read kernel,
and returned 4,096 exact bytes in 0.797 ms. All 18 canonical trials then
reproduced one identical set of expert gate/up/SwiGLU/down diagnostics, routed
bytes, final-residual bytes, route IDs and weights, and sparse-repair counts.
The routed result remains the known rejected L3 result rather than gaining a
weaker gate.

Cold complete routed-layer medians are:

| Variant | Median wall | Physical reads | Speedup vs C0 |
| --- | ---: | ---: | ---: |
| C0 copied/global release | 785.196 ms | 202,653,696 bytes every trial | 1.00x |
| C1 artifact/copied | 301.831 ms | 193,314,816--201,719,808 bytes | 2.601x |
| C2 artifact/no-copy | 123.053 ms | 201,719,808 bytes every trial | 6.381x |

C1 alone clears the 2x continuation gate by eliminating 354.996 ms of repeated
tensor scanning and 375.217 ms of expert-scoped global release; cold source
page acquisition moves into its 252.588 ms copied-buffer installation. C2
removes that copy, spending only 0.425 ms creating source buffers and moving
the cold page acquisition into 95.784 ms of synchronous waits, of which only
8.291 ms is GPU execution. C2 is 2.453x faster than C1. None of the nine cold
candidate trials regresses against any cold control.

After one explicit 237.654 ms prefault outside timing, the genuine warm
candidate medians are 68.411 ms copied and 41.082 ms no-copy, with zero
physical reads and zero page-ins in all six candidate trials. C0 remains cold
by construction because its per-expert lifecycle always invalidates the
checkpoint; its rows are retained as the physical-policy control, not relabeled
as a warm-cache baseline.

Two earlier attempts stopped before producing a report when Darwin's
`ru_minflt` observation fell across a multithreaded boundary; the reproduced
transition was 36,840 to 36,833. The protocol now preserves minor/major fault
deltas as signed observations while disk bytes, page-ins, and CPU-time counters
still fail closed on regression. The failure ledger hashes to
`89cc1c13c7b289d5a13cd20242abfbf2ad5568ff9104f98c356cadfe697dff02`.
The first complete report then exposed that its nominal warm rows followed a
candidate invalidation and still read the full artifact. That report is
preserved under `trials-001`, hash
`33bc9febde92e4bcb25987a2e260ac7ea9cceee07e41c65a294adc17eff0fdbf`,
but only its cold rows are physically interpretable.

The first corrected report under `trials-002` used a non-object commit string
expanded from an abbreviated hash and is therefore noncanonical despite
otherwise valid measurements. The canonical rerun at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0106/trials-003/report.json`
hashes to
`fb0a1cf0e9dba0d3941a5d9786e4867fe04ea21dcd81469d986928fdaada9232`.
Its clean canonical analysis hashes to
`635e26fb8060c216e917423c6052a3cb42865bc81a27cf2bc7b4322ce2b7edfc`.
Gate 8 passes throughout with 78% minimum free memory, 557,498,368-byte peak
RSS, 117,248,128-byte final physical footprint, zero swap growth, zero new
throttled pages, and stable protected services.

## Decision

Promote the page-stable no-copy representation into a broader routed-layer
transaction experiment. PW-0105's proposed mechanism is causally confirmed:
stable prevalidation/lifecycle supplies a real 2.601x cold gain, and the M1's
no-copy mapping supplies another 2.453x after that larger defect is removed.
The next candidate should keep all eight experts' gate/up/down commands and
intermediate activation stages inside one layer-scoped Metal transaction,
overlap page acquisition with GPU execution through bounded arenas, and return
only the routed residual.

Do not promote this component as endpoint TPS or build the complete 303 GB
artifact yet. A direct 47x extrapolation of the 123.053 ms layer is about 5.78
seconds of routed work per token before non-MoE work, and cold waits still hide
roughly 87.6 ms beyond GPU execution per layer. The experiment also preserves
PW-0101's arithmetic failure. Transaction fusion and the separate numerical
branch must each pass their own gates before another full walk.
