# PW-0038 — Native noaux-tc router

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `5484034`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; exact layer-43 router artifact;
  PW-0037 manifest SHA-256
  `a2b30be7ab767c754fd4680887420246dcd28d314bff242b144f664a8ff12470`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal
- Related records: PW-0003, PW-0016, PW-0017, PW-0037

## Hypothesis and mechanism

The exact layer-43 router is only an F32 `256×4,096` projection plus sigmoid,
correction-bias top-eight choice, uncorrected-score gather, and normalization.
A shared-weight batch-eight Metal projection followed by Rust selection can
replace PW-0037's frozen route authority without introducing another model
runtime or approximation.

## Contract

Add `metal-noaux-tc-router`. Pass only if:

1. `MappedSafetensors` validates exact F32 router weight `[256,4096]` and
   correction bias `[256]`; input is the exact finite PW-0037 `8×4,096` F32
   artifact and every source/artifact hash is checked;
2. add `f32_gemm8_shared_weight` with one 64-lane threadgroup per expert row,
   one decoded F32 weight load applied to eight positions, and exactly 2 KiB
   threadgroup reduction memory. A deterministic production-aligned scalar
   fixture must pass at maximum absolute error `2e-4`;
3. Rust computes sigmoid scores, chooses top eight using corrected scores with
   explicit descending-score/index tie order, gathers uncorrected scores, and
   normalizes with the source `1e-20` denominator. Unknown/non-finite/tied-at-
   boundary states fail closed rather than assuming Torch ordering;
4. every selected expert set must equal the independent PW-0037/Torch set.
   Normalized weights matched by expert must have maximum absolute error at
   most `2e-6`; route rows must be finite, unique, and sum to one within
   `1e-6`;
5. write a canonical create-new/hash-bound route artifact; two complete
   processes must be byte-identical. After five warmups, 30 serialized router
   projection measurements report p10/median/p90 with median at most 1 ms;
6. report compile/cold/full process wall, bytes, batch eight, concurrency one,
   accepted tokens, `A`, `U`, hardware, commit, and cache state. No layer or
   endpoint TPS claim is permitted.

Passing promotes native router decision authority on this fixture. It does not
yet make those decisions drive the heterogeneous scheduler inside one timed
request.

## Baseline and candidate

Baseline is PW-0037's independently computed Torch route IDs and normalized
weights. Candidate is exact F32 Metal projection plus Rust noaux-tc selection.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0038`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
