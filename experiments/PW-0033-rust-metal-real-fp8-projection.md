# PW-0033 — Rust-owned Metal real FP8 projection

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `f60983d`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
