# PW-0033 — Rust-owned Metal real FP8 projection

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `9670ca7`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; locked MTP source and PW-0032
  normalized input/native output
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal;
  source read-only on external platter
- Related records: PW-0008, PW-0031, PW-0032

## Hypothesis and mechanism

PW-0008's 64-lane blocked-LUT Metal schedule can be brought behind the Rust
runtime authority. Rust should validate and map the exact production QKV
tensors, create immutable Metal buffers from those views, dispatch the kernel,
validate every output against PW-0032, and own the output artifact and report.

## Contract

Add a macOS-only Rust `metal-fp8-gemv` path. Pass only if:

1. all source dtype, shape, scale-grid, input, and output authority remains in
   the PW-0031/PW-0032 Rust code; the Metal bridge receives validated explicit
   dimensions and immutable bytes and cannot infer tensor layout;
2. the runtime-compiled kernel is exactly
   `block_fp8_gemv_parallel_lut_blocked`, uses a 256-entry exhaustive decode
   LUT, 64 lanes per row, explicit threadgroup memory, and rejects unsupported
   devices, kernels, dimensions, non-finite output, and existing output paths;
3. all 14,848 Metal outputs agree with PW-0032's readable Rust output at
   relative L2 at most `2e-5` and maximum absolute error at most `2e-4`; the
   MLX and float64 gates remain inherited diagnostics, not weakened limits;
4. the produced F32 artifact is finite, exact length, create-new, hashed, and
   byte-identical across repeated complete runs;
5. after five warm-ups, 30 serialized application-buffer-warm command-buffer
   measurements report wall median/p10/p90. Median must be at most 5 ms and at
   least 20 times faster than PW-0032's 300 ms repeated whole-command
   diagnostic; report the asymmetry that PW-0032 includes process/output
   overhead while the inner Metal series does not;
6. report full cold process wall, runtime compile time, logical bytes, batch
   one, concurrency one, accepted tokens, `A`, `U`, hardware, commit, and cache
   state. No endpoint TPS claim is permitted.

Passing promotes the Rust-owned Metal projection primitive, not a complete
layer or runtime. Failure retains PW-0032 as the native reference.

## Baseline and candidate

Baseline is PW-0032's readable single-thread mapped Rust QKV projection.
Candidate uses the same validated source/input contract and PW-0008 Metal
schedule, now dispatched and checked by Rust.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0033`.

## Isolated attribution

The Rust binary maps and validates the exact `60,817,408`-byte FP8 QKV tensor,
`14,848`-byte F32 scale grid, `4,096`-float normalized input, and
`14,848`-float output. Rust passes these explicit validated views and dimensions
to `block_fp8_gemv_parallel_lut_blocked`; the kernel uses one 64-lane
threadgroup per output row and a complete 256-entry FP8 decode LUT.

After five warmups, the first process's 30 serialized command-buffer wall
measurements have a 1.622 ms median, 1.606 ms p10, and 1.745 ms p90. The
repeat process has a 1.678 ms median, 1.610 ms p10, and 1.813 ms p90. The first
dispatches are 4.226 and 4.199 ms. Runtime Metal compilation is 279.4 ms in
the first process and 2.00 ms in the repeat; complete process wall is 2.26 and
0.17 seconds respectively.

The first median is 184.95 times faster than PW-0032's 300 ms repeated
whole-command diagnostic, passing the predeclared 20-times gate. This is an
intentionally asymmetric attribution: PW-0032 includes process startup,
mapping, validation, single-thread GEMV, output creation and `fsync`, hashing,
and JSON, whereas the Metal series measures command creation, dispatch, and
wait against resident application buffers. It is neither an endpoint speedup
nor a storage result.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All `14,848` outputs agree with PW-0032 at `1.14019e-6` relative L2 and
`1.57356e-5` maximum absolute error, passing the `2e-5` and `2e-4` gates. The
output SHA-256 is
`4701c3c38a11f996aa58d61958403ee550e18ccb35fc6a61198325a2e8bf260b`.
Two complete processes produce byte-identical output. A second attempt at an
existing output path exits nonzero before dispatch or mutation.

The shared validator rejects unknown dtype/layout, malformed scale grids,
non-finite inputs/scales/FP8 encodings, and unsupported dimensions. The
existing exhaustive FP8 oracle and production-width/tiny block fixtures remain
green. Rust has 14 passing tests, Python has 21, and clippy is clean with
warnings denied.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0033`.
Its `SHA256SUMS` manifest hashes to
`f662914724fe4f03efed9da2957df64b52ff4868e9f955e284c165a41329bcba`.

## Decision

Promote the Rust-owned Metal FP8 projection primitive. The accelerator consumes
the same mapped-byte and shape authority as the readable Rust reference, and
the real production-width output and repeated timing gates pass.

Do not promote a complete layer or runtime throughput claim. The measured
series excludes source mapping, buffer construction, and the remainder of the
decoder block; the fixture is MTP rather than a base MoE layer. The next branch
should compose this projection with the already-proven attention and dense
block semantics under one native runtime, while the base EP0 shard remains the
transition to routed-layer work.
