# PW-0006 — First Metal FP8 GEMV parity

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation commit pending
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0005 fixture
  `3f544c2bf5f6273cf695af18ce570bb9e67af937feaae67e98c279df190ba8a6`
- Hardware, OS, compiler, storage, memory pressure: Apple M1 GPU; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift 6.3.3; runtime-compiled Metal
- Related records: PW-0004, PW-0005

## Hypothesis and mechanism

An MSL compute kernel can consume source FP8 bytes directly, apply the correct
128-column inverse scales, and accumulate real 4096-wide rows without an
intermediate expanded weight representation or semantic drift.

## Contract

Target-faithful accelerated component, using the exact PW-0005 fixture and
Apple M1 GPU. The first kernel assigns one output row to one GPU thread to keep
the causal path simple. It makes no throughput claim.

Pass: all four GPU outputs match the trusted fixture within `2e-6`, command
buffer completion succeeds, and the actual Metal device is reported.

## Baseline and candidate

Baseline: PW-0005 Rust scalar block-FP8 GEMV and safetensors/PyTorch expected
values. Candidate: `kernels/block_fp8_gemv.metal`, compiled at runtime and
dispatched by the Swift diagnostic harness. Both consume the same raw FP8
fixture bytes and f32 scales/activations.

Exact command:

```sh
swiftc -O -framework Metal tools/metal_fp8_gemv.swift -o <temporary-binary>
<temporary-binary> evals/fixtures/real/mtp-gate-fp8-gemv.json \
  kernels/block_fp8_gemv.metal
```

## Isolated attribution

The result reports device `Apple M1`, four rows, 4,096 columns, and maximum
absolute error `7.916241884231567e-09`.

## End-to-end result

Committed fixture bytes travel through real Metal buffer allocation, runtime
MSL compilation, command encoding, M1 GPU execution, shared-buffer publication,
and asserted human-readable JSON output.

## Correctness result

Every row passes. External raw result SHA-256:
`bdcc30382bf1ab9273d9c45287e6271875066ea0f455edfa83be0ccd5fe701e6`.
Kernel source SHA-256:
`3db54d69d9638f581dc285706dc5d6a335d93a86cdc1047026ed04126dcee488`.

## Decision

Promote this kernel and harness as the accelerated correctness rung. Do not
promote it as a performance path: one thread per row is intentionally serial
across columns, and four-row timing cannot predict a full 2,048/4,096-row
projection or complete expert.
