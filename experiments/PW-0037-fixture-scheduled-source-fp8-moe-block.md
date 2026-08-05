# PW-0037 — Fixture-scheduled source-FP8 heterogeneous MoE block

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `4fcdf63`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0016 selected-range manifest
  SHA-256 `62b15f1c6aabbbacb9a5c730af30b8f78ded0516666024dd3dbfefbe74549f22`;
  PW-0016 route fixture and exact source-FP8 tensors
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal;
  selected exact tensor artifacts on external platter
- Related records: PW-0016, PW-0017, PW-0034, PW-0036

## Hypothesis and mechanism

PW-0036 proves efficient source-FP8 batch-eight execution for one expert. The
actual PW-0016 layer-43 fixture selects nine heterogeneous experts across 64
expert positions: seven experts receive eight positions, one receives five,
and one receives three. A single Rust-owned Metal runtime can gather those
positions, pad incomplete batches explicitly, execute all nine exact experts,
apply frozen source route weights, and scatter-add the complete MoE output.

## Contract

Add a fixture-scheduled `metal-fp8-moe-block` path. Pass only if:

1. an independent generator recomputes PW-0016 source noaux-tc routes and
   normalized weights from the exact router, materializes the complete
   source-FP8 MoE oracle, and emits a duplicate-free manifest mapping every
   selected expert projection to one exact selected-range artifact;
2. Rust fails closed unless the manifest is the named layer-43, batch-eight,
   top-eight semantic with exactly 64 placements, nine experts, counts
   `{8×7,5,3}`, valid unique token/slot pairs, finite normalized route weights,
   known tensor names, and validated source FP8/scale layouts;
3. one process owns all expert buffers. It uses PW-0036 shared-weight GEMM8,
   pads five- and three-position experts with zero inputs, applies route weights
   only to real positions, and performs deterministic scatter-add into all
   `8×4,096` outputs. Padding (`72` processed versus `64` real positions) must
   be reported, not hidden;
4. add and independently fixture a route-weighted scatter-add Metal semantic.
   Complete output relative L2 versus independent Torch source FP8 must be at
   most `4e-5`, maximum absolute error at most `3e-8`; length, finiteness,
   create-new behavior, hashes, and repeated byte identity remain mandatory;
5. after five warmups, 30 serialized resident-buffer whole-MoE measurements
   report p10/median/p90. Median must be at most 25 ms. Report the measured
   routed-only `A=8` diagnostic over 47 layers, but no endpoint TPS claim;
6. report source/logical bytes, resident buffers, batch, concurrency, accepted
   positions, `A=8`, observed `U=9/8`, hardware, commit, cache state, and the
   fixture-static scheduling limitation.

Passing promotes heterogeneous expert execution and weighted reduction, not
dynamic router authority, representative route reuse, a transformer layer, or
an endpoint.

## Baseline and candidate

Correctness baseline is independently routed Torch source FP8. Historical
performance context is PW-0016's 9.83 ms affine-INT4/17%-error block and
PW-0036's 1.935 ms faithful full-batch single-expert primitive. Candidate is
the first faithful nine-expert source-FP8 Rust/Metal composition.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0037`.

## Isolated attribution

The frozen source router selects experts `{3,8,63,98,141,152,182,185,208}`.
Seven receive eight positions, expert 98 receives five, and expert 141 receives
three. The fixed batch-eight shared-weight kernel therefore executes 72 slots
for 64 real placements: 12.5% explicit padding overhead. The runtime validates
and holds `226,809,856` logical source/input/output bytes and reports
`228,186,112` resident buffer bytes.

After five warmups, the first 30 whole-MoE command measurements have a 16.1445
ms median, 15.9973 ms p10, and 16.2484 ms p90. The repeat has a 16.1581 ms
median, 16.0336 ms p10, and 16.2485 ms p90. First-dispatch times are 30.56 and
28.87 ms. Complete process wall is 1.76 and 1.46 seconds, including exact hash
verification of all selected artifacts, mapping, buffer construction, 35
dispatches, output `fsync`, hashing, and JSON.

Mean resident-buffer median is 16.1513 ms. Reusing this one layer's observed
`A=8`, `U=9/8` behavior over all 47 routed layers gives 10.539 routed-only
accepted TPS. It excludes router recomputation, every dense/attention/logit
path, storage misses, MTP draft work, and endpoint overhead and cannot be
treated as representative route reuse.

PW-0016's affine-INT4 block was faster at 9.83 ms and 17.31 routed-only TPS,
but differed from source FP8 by 17.02%. PW-0037 establishes the faithful native
cost rather than carrying that lower-fidelity number forward as a target-mode
default.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

The independent generator repeats byte-identically for input, complete Torch
source-FP8 output, and route manifest. It verifies all 12 selected artifacts
against PW-0016's pinned extraction hashes and emits a duplicate-free
six-tensor authority map for every expert.

The first runtime attempt failed closed because expert 63's up weight and scale
are split across shard-selected artifacts; no measurement was emitted. The
manifest/runtime were corrected to name and validate all six tensor authorities
independently. This preserved failure is evidence that source layout is not
inferred from projection grouping.

All `8×4,096` final values match independent Torch at `1.70815e-6` relative L2
and `7.27596e-11` maximum absolute error. The new route-weighted scatter fixture
is byte-exact. Repeated complete outputs are byte-identical with SHA-256
`fae802459b2c7ebccd5cf6d5e6b065ad2d00c12400e4a1f94894c5851fd47e8e`;
create-new rejection passes. Rust has 15 passing tests, Python has 21, and
clippy is clean with warnings denied.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0037`.
Its `SHA256SUMS` manifest hashes to
`aee7074c716b44ac00d51d1c213dd788ddf62ca08146da368baecfacf867df49`.

## Decision

Promote fixture-scheduled heterogeneous source-FP8 expert execution and
route-weighted reduction. The real selected expert set now causally determines
gather, padded native execution, weighting, scatter, and the complete observed
output under one Rust-owned runtime.

Do not promote dynamic routed-layer or endpoint claims. Route IDs and weights
are independently source-derived but frozen in the manifest; the runtime has
not yet recomputed router logits/top-k. The next native authority is the exact
layer-43 noaux-tc router feeding this scheduler, followed by base-layer hidden
states when EP0 completes.
