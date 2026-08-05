# PW-0008 — Parallel Metal FP8 projection

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `3014f32`; dirty candidate implementation
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; `model_mtp.safetensors`
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`
- Hardware, OS, compiler, storage, memory pressure: Apple M1 GPU; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift 6.3.3; warm shared Metal buffers;
  no material memory pressure observed
- Related records: PW-0006, PW-0007

## Hypothesis and mechanism

Cooperative column evaluation, a constant 256-entry FP8 decode table, and loops
aligned to the real 128-column scale blocks should remove the serial kernel's
dominant instruction and indexing costs while preserving direct source-byte
execution.

## Contract

Same target-faithful component contract and real 16,384×4,096 projection as
PW-0007. Batch one, concurrency one, five warm-ups and 30 measured runs per
candidate. Storage load and buffer creation are excluded and named. Candidate
widths 32, 64, 128, and 256 are diagnostic; promotion requires correctness and
at least 25 GiB/s. Interleaved control/candidate repetitions guard against
warm-up and thermal order bias.

## Baseline and candidate

Baseline is the PW-0007 one-thread-per-row kernel at approximately 6.50 GiB/s.
Candidates were introduced cumulatively:

1. parallel column reduction: best 15.10 GiB/s at 32 lanes;
2. FP8 decode lookup: best 21.99 GiB/s at 32 lanes;
3. scale-block-aligned lookup/reduction: best initial 37.50 GiB/s at 64 lanes.

The selected kernel is `block_fp8_gemv_parallel_lut_blocked` with 64 lanes.

## Isolated attribution

Interleaved order and medians:

| Run | Kernel | GPU ms | Logical GiB/s |
| --- | --- | ---: | ---: |
| control 1 | serial | 9.6309 | 6.4895 |
| candidate 1 | blocked LUT, 64 | 1.6717 | 37.3878 |
| control 2 | serial | 9.6152 | 6.5001 |
| candidate 2 | blocked LUT, 64 | 1.6386 | 38.1417 |

The mean of candidate run medians is 37.7648 GiB/s, a 5.82× gain over the
interleaved controls. Maximum checked error remained below `4e-9`.

At that measured component bandwidth, byte traffic alone corresponds to 4.284
source-FP8 routed-only TPS. Reaching 50 TPS would require `A/U >= 11.67` even if
all dense, attention, activation, synchronization, and sampling work were free.
Applying the same bandwidth hypothetically to the unbuilt INT4 byte estimate
would require `A/U >= 6.56`. These are diagnostic necessary conditions, not
predicted endpoint rates.

## End-to-end result

No endpoint TPS is claimed. The candidate executes a complete real projection
from raw FP8 bytes in warm Metal buffers through parallel GPU reduction and
checked output. The unified clean-checkout test runs both transparent and
selected accelerated kernels on the committed real fixture.

## Correctness result

Every candidate passed the fixture. Interleaved raw-result hashes:

- control 1: `e43cec569499399a89a9ebe2f3a27df6a57dce418b409927d9119e1c595e9d09`
- candidate 1: `2eb747e49aa83466d9a47da37c9b6d9c9daa6677f02fd19e28022f41cde77a95`
- control 2: `2ea41d47a46eb3ea5eee1cb7924ce2b61178acde1180f92ea5113d18dda1ad37`
- candidate 2: `1312ded931be10c93fe7c9f5fb8357281a520c5469146c6d95bc270e8cc95814`

Selected kernel source SHA-256:
`f7e5c99b4ba9d695715c704e0eaa3ae6332095c68abfe6e371a359ecd408a7e4`.

## Decision

Promote the 64-lane blocked-LUT kernel as the accelerated FP8 projection
baseline. It passes the component threshold and repeatability check. It is not
an inference performance default: fused gate/up/SwiGLU/down execution, expert
shapes, batch effects, dispatch aggregation, and complete decode remain to be
measured. The new bandwidth makes speculation efficiency a quantified rather
than rhetorical dependency.
