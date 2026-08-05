# PW-0037 — Fixture-scheduled source-FP8 heterogeneous MoE block

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `94bbc6b`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
