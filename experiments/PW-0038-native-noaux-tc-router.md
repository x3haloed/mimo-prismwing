# PW-0038 — Native noaux-tc router

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `29cff32`; implementation dirty
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

The Rust runtime validates `4,195,328` source bytes represented by the exact
`256×4,096` F32 weight and 256-value correction bias, plus the exact PW-0037
batch-eight input and reference manifest. The complete reported logical
footprint, including logits and route output, is `4,335,704` bytes. One
64-lane threadgroup owns each expert row and shares each F32 weight across all
eight positions using exactly 2 KiB of reduction memory.

After five warmups, two 30-measurement processes have medians of `0.324625`
and `0.347917` ms (mean `0.336271` ms). Their p10/p90 pairs are
`0.309792/0.339375` and `0.309375/0.384875` ms. The measured process took
0.09 seconds including artifact hashing, Metal compilation, the production-
width fixture, 36 real dispatches, selection, create-new output, and hashing;
its first real dispatch was 0.448916 ms. These are router-component timings,
not a layer or endpoint TPS result. Accepted tokens and `A` are therefore zero;
batch is eight, concurrency one, and the observed route union is `U=9/8`.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

The production-width deterministic kernel fixture has maximum absolute error
`5.995389e-9`, far below its `2e-4` gate. Every native selected expert set
equals the independent Torch reference. Route weights matched by expert have
maximum absolute error `1.4901161e-8`; every row is finite, unique, and
normalized. The smallest corrected-score top-eight boundary margin is
`1.5354156e-4`, so no tested decision relies on the explicit tie rule.

Separate complete processes write byte-identical canonical route artifacts
with SHA-256
`a4da77ae2400d652d22bdb6ebee9b28b2050159e806c6395ea4662e9ab4f3e40`.
Create-new rejection exits 1. Rust has 15 passing tests, Python has 21, and
clippy is clean with warnings denied. Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0038`; its `SHA256SUMS` manifest
hash is `80dbf1da3a96d56a2fabdbee468c061558e32e90c5a02d65adbbd4cfcca4d7b7`.

## Decision

Promote the exact layer-43 F32 Metal projection plus native Rust noaux-tc
selection as the router decision authority for this fixture. This replaces the
frozen route IDs and weights as the next computation boundary, while retaining
the PW-0037 Torch artifact as an independent parity oracle.

Do not promote a dynamic routed-layer or endpoint claim. This command emits
the correct route artifact but does not yet make it drive expert gather,
execution, weighting, and scatter inside one timed request. The next experiment
must integrate this authority with PW-0037's promoted heterogeneous scheduler;
base-layer hidden-state validation remains gated on an EP0 source artifact.
