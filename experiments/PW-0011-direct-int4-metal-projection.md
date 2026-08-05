# PW-0011 — Direct signed-INT4 Metal projection

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `0a828ec`; dirty candidate implementation
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; source MTP
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`;
  fixture `178752671f4553756d62c31153f4949338fccbc99e46f5525daecc3718568346`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift 6.3.3; warm shared Metal
  buffers; no material memory pressure observed
- Related records: PW-0005, PW-0008, PW-0010

## Hypothesis and mechanism

A directly consumed groupwise INT4 representation should reduce real M1
projection time enough to leave a physically possible DFlash-8 routed-byte
window, without expanding weights before GEMV.

## Contract

Explicit L3 candidate, not target-faithful storage. Quantize each real
production row independently in 128-column groups with symmetric signed
nibbles `[-7, 7]`, f32 scale `max(abs(group))/7`, nearest-even rounding, and
low-then-high two's-complement nibble order. Correctness requires exact packed
fixture bytes and scales plus Rust/Metal parity with the quantized oracle.

Measure a complete real 16,384×4,096 projection, batch one and concurrency one,
with five warm-ups and 30 runs. Source load and one-time quantization are
excluded and named. Interleave the selected 32-lane candidate with the PW-0008
FP8 control. This component can be conditional only; promotion to an inference
default requires downstream route/logit and whole-model gates.

## Baseline and candidate

Candidate lane-width diagnostics:

| Lanes | GPU median ms | Executable GiB/s |
| ---: | ---: | ---: |
| 16 | 1.5527 | 21.3846 |
| 32 | 0.9780 | 33.9486 |
| 64 | 1.3428 | 24.7262 |

The 32-lane candidate was selected before interleaved confirmation. Its full
projection contains 33,554,432 packed weight bytes and 2,097,152 scale bytes.
Source-file loading and CPU quantization are installation diagnostics, not
included in the warm kernel timing.

## Isolated attribution

Interleaved medians:

| Run | Representation | GPU ms | Physical/executable GiB/s |
| --- | --- | ---: | ---: |
| control 1 | source FP8 | 1.5898 | 39.3143 logical weight |
| candidate 1 | signed INT4 | 1.0710 | 31.0020 executable |
| control 2 | source FP8 | 1.6950 | 36.8741 logical weight |
| candidate 2 | signed INT4 | 1.0705 | 31.0164 executable |

The candidate reduces mean-median projection time by 1.5338× despite lower
physical bandwidth because its representation is 53.11% of source expert
bytes. One routed expert occupies 13,369,344 bytes (12.75 MiB), and a cold
47-layer top-eight token occupies 5,026,873,344 bytes (4.681641 GiB).

At the mean candidate bandwidth of 31.009199 GiB/s, routed-only ordinary-token
rate is 6.623575 TPS. Fifty TPS requires `A/U >= 7.548793`. Published DFlash-8
can exceed that only in the narrow idealized region `U <= 1.059772` at perfect
`A = 8`, equivalent to at most 8.478 average unique experts per layer across
all eight positions. Its otherwise-free routed-only ceiling at `U = 1` is
52.988602 TPS.

## End-to-end result

No endpoint TPS is claimed. Dense/attention traffic, draft execution, KV,
dispatch, synchronization, sampling, and imperfect acceptance are excluded.
The 52.99 routed-only ceiling leaves only 5.98% above the 50-TPS target before
those costs. This does not diminish a future measured endpoint below 50; it
only bounds this component's ability to satisfy the primary Prismwing 50 gate.

## Correctness result

The committed production-width fixture is independently reproduced by a Rust
scalar oracle and the real M1 Metal kernel. Interleaved Metal maximum error
against the quantized fixture was below `4e-9`.

Quantization error remains material and separate from kernel correctness. On
four real rows, mean absolute projection change versus source FP8 is
`0.00064276`, maximum is `0.00145061`, relative L2 error is 9.84%, and cosine
similarity is 0.99610. This cancellation-heavy four-output slice is not a
whole-model fidelity result.

Raw evidence hashes:

- candidate 1: `c39de93b46e64c8a3f0f6c3ae442fed62c0f1c38f798b35b7a7eca2c93aeea28`
- candidate 2: `3c857420a9e917abe854b167c8a6f49bc6fdb08740a39def912870970faac7a4`
- control 1: `c54955868fee395532198fa66323b3a4de8b4fc8b2232252850d2c5c74bd1575`
- control 2: `ca14ab461e1e4e3eafa7911c4d0fe10e898bda9b4d7532a4ae71099471c544a3`

External evidence root:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0011`.

## Decision

Retain the directly executable 32-lane INT4 kernel as a conditional L3
research path because it gives a repeatable 1.53× real projection gain. Do not
promote it to the inference default: its four-row error is nontrivial, and the
idealized DFlash-8 performance window requires nearly identical routes plus
almost no non-expert cost. Actual `A/U` is the next performance kill test;
sampled-real layer-local route/logit drift is the next fidelity gate. A complete
endpoint at any speed remains valuable and must be reported separately from
this Prismwing 50 branch decision.
