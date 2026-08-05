# PW-0032 — Native mapped real FP8 projection

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `9e05f4e`; implementation dirty
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

The native command consumes the exact `4,096`-float PW-0026 normalized hidden
state (SHA-256
`7628eba7f72e52e089cdd9a38c548826ce94d5cdd15d41f5ca52273113ea5717`),
the mapped `60,817,408`-byte FP8 QKV tensor, and its mapped `14,848`-byte F32
scale grid. It emits all `14,848` finite F32 results through a create-new,
fsynced artifact.

The complete native output hashes to
`4c3e44409e7a3376c103867356c51e7d0e7be230e8390549fd61a1f4be5d0cd1`;
the independently decoded MLX output hashes to
`76f124f2a7e4e6b2877f9bf804ce5c5a0b35c1217ab66fed588a04cb28a3f424`.
Their relative L2 is `1.14891e-6` and maximum absolute difference is
`1.66893e-5`, passing the `2e-5` / `2e-4` gates.

Sampled native absolute differences from float64 scalar dots are:

| Row / partition | Absolute error |
|---|---:|
| 0 / Q | `5.96046e-7` |
| 1 / Q | `1.31130e-6` |
| 12,288 / K | `4.61936e-7` |
| 13,824 / V | `1.19209e-6` |

The readable single-thread command accounts for `60,908,032` logical bytes
(weight, scale, input, output). First-recorded application wall time is 0.69
seconds and the repeated run is 0.30 seconds, including mapping, validation,
GEMV, create-new output, `fsync`, hashes, and JSON. The source OS cache was
already warm from fixture generation, so neither value is cold-storage timing.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All six conditions pass. Exhaustive FP8 decode and the two-row/two-column scale
fixture remain green. Tests reject invalid dtype/shape/grid/input/finiteness,
and a second output attempt fails before mutation. A repeated full projection
is byte-identical to the first. The independent fixture generator is also
byte-identical across complete reruns.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0032`.
Its `SHA256SUMS` manifest hashes to
`d641e94e23858891ac96d9ad74a4ea18be7394bc3fe0ad9e5b1b47a4dab7aff5`.

## Decision

Promote the mapped single-thread FP8 GEMV as the native correctness reference
for source block-scaled projections. It is the first production-width learned
model computation owned end to end by the Rust binary rather than Python.

Do not promote it as the accelerated runtime default. A 0.30-second warm QKV
projection is far outside the eventual layer/token budget and omits attention,
output projection, residuals, MLP/MoE, and every endpoint concern. The next
implementation can optimize or delegate this same validated byte/shape
contract to Metal or pinned MLX while retaining the readable reference.
