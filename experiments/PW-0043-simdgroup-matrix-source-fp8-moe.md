# PW-0043 — SIMD-group matrix source-FP8 MoE

- Status: complete
- Disposition: rejected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `4010a41`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0039 exact authorities
- Hardware, OS, compiler, storage, memory pressure: Apple M1 GPU family 7;
  8 GPU cores; 16 GiB; macOS 26.4.1 (25E253); Metal 4 support; Rust release
  plus runtime-compiled Metal
- Related records: PW-0033, PW-0036, PW-0039, PW-0040, PW-0042
- External mechanism authority: Apple A14/M1 SIMD-group 8×8 matrix operations
  and feature tables

## Hypothesis and mechanism

PW-0040 through PW-0042 show that dispatch shape is not the direct-FP8
bottleneck. PW-0039's kernel decodes each weight once for eight positions but
then performs eight scalar F32 accumulations. Apple GPU family 7 provides
SIMD-group 8×8 matrix multiply-accumulate. Decode one source-FP8 `K×8` weight
tile to F32 threadgroup memory, load the corresponding `8×K` activation tile,
and let one SIMD group update an `8×8` output tile. This preserves exact source
values while replacing scalar inner products with the matrix execution unit.

## Contract

Add a SIMD-group-matrix mode to the dynamic native MoE runtime. Pass only if:

1. fail closed unless the device supports Apple GPU family 7-or-newer matrix
   operations and the compiled pipeline has exactly the expected 32-thread SIMD
   width. No silent scalar fallback may be reported as candidate;
2. validate and decode the same source FP8 bytes and F32 block scales. Construct
   explicit row-major `8×K` activation and `K×8` decoded-weight tiles; do not
   quantize, truncate, transpose by assumption, or change expert equations;
3. add an independent deterministic scalar fixture covering multiple output
   tiles, multiple 128-column scale blocks, all eight positions, signed FP8
   values, strides, and matrix stores. Maximum absolute error must be at most
   `2e-4`;
4. preserve PW-0039's router, derived schedule, SwiGLU, weighting, scatter, and
   complete parity gates. Candidate output need not be byte-identical to scalar
   control if matrix accumulation order differs, but must repeat byte-identically
   and pass `4e-5` relative L2 and `3e-8` maximum absolute error versus Torch;
5. use paired candidate/control then control/candidate process orders with five
   warmups and 30 complete-request measurements. Promote only if candidate mean
   median is at least 20% faster than control and at most 14 ms;
6. report p10/median/p90, cold/full process wall, compile time, thread/SIMD group
   shape, tile memory, logical/resident bytes, batch, concurrency, accepted
   tokens, `A=8`, `U=1.125`, hardware, commit, and cache state. Report routed-
   only diagnostic but no endpoint TPS.

Passing promotes the SIMD-group matrix primitive and dynamic MoE schedule on
M1-class hardware. Failure rejects this exact tile design, not all possible
SIMD-group or future Metal TensorOps kernels.

## Baseline and candidate

Control is PW-0039's promoted 64-lane scalar-accumulation shared-weight kernel.
Candidate changes only the three expert projections to decoded F32 8×8 SIMD-
group matrix multiplication; all runtime authority and surrounding work remain
shared.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0043`.

## Isolated attribution

The M1 reports Apple GPU family 7 support and the candidate pipeline reports the
required 32-thread SIMD width. One SIMD group owns each eight-output-row tile.
For each of 512 K tiles it decodes a `K×8` weight tile into 256 bytes of
threadgroup memory, loads an `8×K` activation tile, and executes one F32 8×8
matrix multiply-accumulate.

In candidate/control then control/candidate order, candidate medians are
`22.450083` and `22.531125` ms versus controls `17.107709` and `17.089417` ms.
Candidate mean is `22.490604` ms versus `17.098563` ms: a 0.7603× speedup, or
31.54% slowdown. Candidate p10/p90 pairs are `22.295041/22.748166` and
`22.270583/22.865250` ms. Cold requests are 34.17 and 33.18 ms; complete
process wall is 1.72 seconds. Logical and resident bytes are unchanged.

The fixed-fixture routed-only diagnostic is 7.5682 TPS at `A=8`, `U=1.125`.
It is not endpoint TPS.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

The deterministic scalar fixture covers two output tiles, two 128-column scale
blocks, signed FP8 patterns, all eight positions, strides, and matrix stores;
it is byte-exact. Separate candidate processes are byte-identical with SHA-256
`8145eb05e4e726294a75547de304f9a7bb6e3940b79b82f2c6bf14c99a49cfcc`.

The different accumulation order improves complete parity to `2.083638e-7`
relative L2 and `1.4551915e-11` maximum absolute error versus independent Torch
source FP8. Create-new rejection exits 1. Rust has 15 passing tests, Python has
23, and clippy is clean with warnings denied.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0043`.
Its `SHA256SUMS` manifest hashes to
`bd179891c03d1a936856da66c9cddde6e53d0dc2994b35a738ac1b6e1ffff131`.

## Decision

Reject this exact SIMD-group matrix tile design. Matrix accumulation is highly
accurate, but decoding and synchronizing 512 eight-column K tiles per
projection overwhelms the matrix-unit benefit. Retain the kernel as a
correctness-backed research control, not a default.

This does not reject all matrix kernels: a wider cooperative TensorOps tile or
predecoded tile cache could change the synchronization mechanism, but the
current M1/macOS path has now falsified the available direct 8×8 design. Keep
PW-0039 promoted and stop pursuing dispatch/tile rearrangements without a new
data-reuse mechanism.
