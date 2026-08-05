# PW-0007 — Naive Metal FP8 projection throughput

- Status: complete
- Disposition: rejected
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `3014f32`; dirty benchmark/kernel changes
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; `model_mtp.safetensors`
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`
- Hardware, OS, compiler, storage, memory pressure: Apple M1 GPU; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift 6.3.3; weights in shared Metal
  buffers; no material memory pressure observed
- Related records: PW-0004, PW-0005, PW-0006

## Hypothesis and mechanism

The simplest direct FP8 Metal path—one output row per GPU thread with a serial
column loop—might deliver enough logical weight bandwidth to serve as the first
performance baseline while preserving source bytes and block scaling.

## Contract

Target-faithful component microbenchmark, not accepted TPS. Batch size one,
concurrency one, five warm-ups, 30 measured dispatches. The 64 MiB weight matrix,
16 KiB scale grid, input, and output remain in application Metal buffers; source
file loading is excluded. `A`, `U`, accepted tokens, SSD bytes, and rollback are
not applicable because this is one projection rather than a decode loop.

Pass for promotion: at least 25 GiB/s logical weight bandwidth with real FP8
bytes and correctness, providing a credible base for further tiling. Kill for
this kernel shape: below 10 GiB/s after warm-up.

## Baseline and candidate

Baseline correctness is PW-0005/PW-0006. Candidate is the updated MSL kernel
with full 128×128 row/column scale indexing, applied to the real 16,384×4,096
MTP gate projection. This matrix contains eight times the weight bytes of one
2,048×4,096 routed-expert gate or up projection.

Exact command:

```sh
swiftc -O -framework Metal tools/metal_fp8_benchmark.swift -o <binary>
<binary> <checkpoint>/model_mtp.safetensors \
  evals/fixtures/real/mtp-gate-fp8-gemv.json \
  kernels/block_fp8_gemv.metal
```

## Isolated attribution

- GPU median: 9.6183 ms; p10 9.4405 ms; p90 9.9181 ms.
- Complete dispatch/wait wall median: 9.9333 ms.
- Logical weight bandwidth median: 6.498 GiB/s.
- Weight bytes per dispatch: 67,108,864; scale bytes: 16,384.
- First-four-row maximum absolute error: `7.9163e-09`.

At that measured bandwidth, the source-FP8 cold routed bytes alone imply
approximately `8.815 / 6.498 = 1.356` seconds per ordinary token, or 0.737
routed-only TPS, before attention, dense weights, activations, dispatches, or
sampling. This is a diagnostic bound for this kernel, not endpoint TPS.

## End-to-end result

No endpoint result is claimed. The complete projection path includes real shard
read into application memory, Metal buffer creation, warm dispatches, GPU
execution, completion wait, output readback, and fixture correctness. Timed
runs exclude the one-time file read and buffer creation as declared.

## Correctness result

All 30 measured dispatches completed; the checked outputs retained PW-0006
parity. External result SHA-256:
`56057c575b1db2cd8e29a2ef023227504e77d62ba85b82c31337a864723353f6`.

## Decision

Reject the one-thread-per-row kernel as a performance architecture; it misses
the predefined 10 GiB/s kill threshold. Preserve it as a transparent accelerated
correctness path. The next cheapest falsification is a threadgroup-parallel
column reduction that should expose whether serial per-row work, FP8 software
decode, or memory bandwidth is dominant.
