# PW-0032 — Native mapped real FP8 projection

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `59cd4be`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; locked complete MTP SHA-256
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release build; source read-only on
  external platter
- Related records: PW-0004, PW-0026, PW-0031

## Hypothesis and mechanism

PW-0031's immutable source views can drive a readable native FP8/block-scale
projection without copying or decoding the weight tensor. A complete
14,848-by-4,096 MTP fused-QKV GEMV is the smallest production-width native
model computation that advances the Rust binary beyond artifact inspection.

## Contract

Add a native mapped FP8 GEMV and `fp8-gemv` CLI. Pass only if:

1. weight dtype/shape must be `F8_E4M3 [rows,columns]`, scale dtype/shape must
   be `F32 [ceil(rows/128),ceil(columns/128)]` for production multiples of 128,
   input must be exactly `columns` finite little-endian F32 values, and every
   byte range comes from `MappedSafetensors`;
2. unknown/mismatched dtypes, shapes, scale grids, input lengths, non-finite
   inputs/scales, absent tensors, and existing output paths fail closed before
   a result is promoted;
3. exhaustive FP8 decode remains green and a tiny independently computed
   multi-block fixture covers row and column scale selection;
4. generate the exact PW-0026 first normalized hidden state as a frozen F32
   input, run the complete learned layer-zero fused QKV projection, and compare
   all 14,848 native outputs with an independent MLX source-weight oracle at
   relative L2 at most `2e-5` and maximum absolute error at most `2e-4`;
5. sampled rows spanning Q, K, and V agree with float64 scalar dots at maximum
   absolute error `2e-4`; output length, finiteness, SHA-256, and create-new
   behavior are recorded;
6. report application-cold and repeated warm whole-command wall time, logical
   weight/scale/input/output bytes, batch one, concurrency one, hardware,
   commit, and cache state. Timing remains a projection diagnostic, not
   endpoint TPS.

Passing promotes this readable single-thread projection as the native
correctness reference. It does not make it the accelerated runtime default.

## Baseline and candidate

Baseline is MLX matmul over independently decoded source FP8 and block scales,
with float64 scalar row checks. Candidate is Rust accumulation directly over
the mapped FP8 bytes and mapped little-endian F32 scales.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0032`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
