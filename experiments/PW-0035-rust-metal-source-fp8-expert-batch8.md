# PW-0035 — Rust-owned Metal source-FP8 expert batch eight

- Status: complete
- Disposition: rejected
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `ab5fe82`; implementation dirty
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

The new kernel flattens eight positions and each output row into independent
64-lane threadgroups. It therefore reduces command-level repetition but each
position still scans the same FP8 weights in a separate threadgroup; it does
not share weight tiles across positions.

After five warmups, the first report's 30 complete-expert measurements have a
5.1313 ms median, 5.0482 ms p10, and 5.2560 ms p90. The repeat has a 5.1108 ms
median, 4.9769 ms p10, and 5.3638 ms p90. Per-position times are 0.6414 and
0.6388 ms, only 1.592× and 1.598× faster than PW-0034 batch one. Both the 4 ms
median gate and 2× per-position gate fail.

Under the deliberately idealized `A=8`, `U=1` perfect-reuse case—eight unique
experts each receiving all eight positions at every routed layer—the two
medians imply only 4.146 and 4.163 routed-only accepted TPS. This excludes
routing, dense weights, attention, logits, storage, MTP, and endpoint work and
is not a representative route-union claim.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Correctness passes even though performance does not. All `8×4,096` outputs
agree with the independently dequantized Torch source-FP8 oracle at
`1.62608e-6` relative L2 and `2.47383e-10` maximum absolute error. The
1,024-output synthetic GEMM8 fixture is byte-exact to the Rust scalar oracle;
the inherited SwiGLU fixture remains within `1.97745e-7`.

The output SHA-256 is
`8c198563b12f73a7c5fd181e2d173ffa55692c64ddd1d5386cb0ccdfac2a1393`.
Two complete candidate processes and two independent fixture generations are
byte-identical. Existing output paths fail closed. Rust has 15 passing tests,
Python has 21, and clippy is clean with warnings denied.

The first implementation run correctly exited nonzero when the performance
gate failed but emitted no JSON. The reporter was then changed to preserve
complete failed-gate metrics while retaining explicit false gate fields; the
acceptance criteria were not weakened. Both the initial failure and measured
reports are preserved in raw evidence.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0035`.
Its `SHA256SUMS` manifest hashes to
`c2a49a43c27ecfef9a7b37dbc337f4b107ad0aede8ea3ea48db1627f38062cd6`.

## Decision

Reject flattened batch-row threadgroups as the faithful expert batching
schedule. Retain the kernel as a correctness reference and diagnostic, but do
not promote it as the batched runtime primitive.

The hypothesis failed for a specific causal reason: dispatch topology alone
does not realize expert-weight reuse. The next candidate must load a weight
tile once and apply it to all eight positions inside the same threadgroup (or
use a tuned native GEMM substrate), with a new predeclared experiment. No
heterogeneous MoE integration should build on this rejected performance path.
