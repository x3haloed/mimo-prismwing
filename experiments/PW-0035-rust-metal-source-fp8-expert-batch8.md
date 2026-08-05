# PW-0035 — Rust-owned Metal source-FP8 expert batch eight

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `b6cea66`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0034 source identities;
  PW-0015 deterministic batch-eight input semantic
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal;
  source artifacts read-only on external platter
- Related records: PW-0015, PW-0019, PW-0034

## Hypothesis and mechanism

MiMo verification and heterogeneous routing can reuse one expert across
multiple positions only if the faithful native kernel executes a real batch
rather than repeating batch-one dispatches. Flattening batch and output row
across threadgroups should amortize command and weight access enough to improve
per-position cost while preserving PW-0034 semantics.

## Contract

Add a Rust-owned batch-eight source-FP8 complete-expert path. Pass only if:

1. add one explicit `block_fp8_gemm8_parallel_lut_blocked` Metal kernel; its
   weight/scale layout, FP8 LUT, 64-lane reduction, and shape authority remain
   identical to PW-0033/PW-0034, with batch fixed and named as eight;
2. before real measurement, execute a deterministic production-aligned
   128×128 synthetic batch-eight fixture with varying inputs and scales and
   compare every output to the Rust scalar FP8 oracle at maximum absolute
   error at most `2e-4`;
3. use PW-0015's exact eight FP16-rounded input rows and the same real
   layer-43/expert-32 gate/up/SwiGLU/down tensors. Compare all `8×4,096`
   outputs with an independent Torch source-FP8 oracle at relative L2 at most
   `3e-5` and maximum absolute error at most `2e-8`;
4. every boundary is finite, output is exact-length/create-new/hashed, and two
   complete processes produce byte-identical output;
5. after five warmups, 30 serialized resident-buffer complete-expert
   measurements report median/p10/p90. Median must be at most 4 ms and
   per-position time must improve by at least 2× over PW-0034's 1.020875 ms
   batch-one median; no comparison may hide the batch-size difference;
6. report an idealized perfect-reuse DFlash-8 routed-only diagnostic using
   eight unique experts per layer and eight accepted positions, plus bytes,
   batch, concurrency, `A`, `U`, hardware, commit, and cache state. It is not
   endpoint TPS or representative route-union evidence.

Passing promotes a faithful native batch-eight expert primitive. It does not
promote a heterogeneous MoE layer, speculation acceptance claim, or endpoint.

## Baseline and candidate

Baseline is PW-0034's 1.020875 ms batch-one source-FP8 complete-expert median.
Candidate applies the same expert equation to eight PW-0015 rows in one batched
kernel chain. Correctness baseline is independent Torch source-FP8 matmul.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0035`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
