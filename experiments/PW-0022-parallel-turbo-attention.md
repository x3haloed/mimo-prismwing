# PW-0022 — Parallel MiMo Turbo attention reduction

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `8e3404e`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0020 Atomic source lock
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift/Metal runtime compiler; internal SSD
- Related records: PW-0020, PW-0021

## Hypothesis and mechanism

Splitting KV tokens across one 32-lane Metal threadgroup and associatively
merging online-softmax states should preserve PW-0021 semantics while removing
the serial token bottleneck. A shared rotated query and 32 partial 128-wide
outputs fit within M1 threadgroup memory.

## Contract

Same target-faithful shapes, modified KV formats, deterministic fixture, and
component-only scope as PW-0021. The parallel candidate passes only if:

1. Turbo3 and Turbo4 pass contexts 17, 128, 1,024, and 8,192 with finite
   output, intact guards, Metal-versus-scalar relative L2 at most `2e-4`, and
   maximum absolute error at most `3e-4`;
2. exactly 32 lanes partition tokens without duplication or omission,
   including contexts smaller than 32 and nonmultiples of 32;
3. at context 8,192, two paired process orders per format (`serial,parallel`
   then `parallel,serial`) each use 10 warm-ups and 50 measurements, and the
   candidate's mean GPU median is at least 8× faster than the paired serial
   mean;
4. all reports retain batch one, concurrency one, one accepted token, bytes
   read, cold/warm state, wall/GPU median and p95, hardware, and commit. `A` and
   `U` remain not applicable.

Passing promotes only the parallel schedule as the synthetic accelerated
attention default. It is not endpoint TPS or fidelity evidence. Failure keeps
PW-0021 as the correctness oracle and triggers a different reduction topology.

## Baseline and candidate

Baseline is PW-0021's one-thread Metal kernel. Candidate shares its exact
packing and dequantization functions but uses one 32-thread group, a shared
rotated query, per-lane online-softmax state, and an associative lane-zero
merge. The scalar CPU reference remains authoritative for correctness.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0022`.

## Isolated attribution

All required parallel runs use batch one, concurrency one, one accepted token,
10 warm-ups, and 50 measurements:

| Format | Context | Bytes read | GPU median / p95 ms | Wall median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Turbo3 | 17 | 3,574 | 0.672 / 0.949 | 0.869 / 1.211 |
| Turbo3 | 128 | 20,224 | 0.900 / 1.319 | 1.119 / 1.606 |
| Turbo3 | 1,024 | 154,624 | 4.164 / 5.252 | 4.524 / 5.641 |
| Turbo3 | 8,192 | 1,229,824 | 31.806 / 32.318 | 32.260 / 32.794 |
| Turbo4 | 17 | 4,492 | 0.483 / 0.754 | 0.683 / 1.124 |
| Turbo4 | 128 | 27,136 | 0.858 / 1.202 | 1.063 / 1.685 |
| Turbo4 | 1,024 | 209,920 | 3.925 / 4.838 | 4.336 / 5.175 |
| Turbo4 | 8,192 | 1,672,192 | 29.968 / 30.355 | 30.361 / 30.820 |

First cold GPU/wall dispatches range from 0.537/2.884 ms at context 17 to
36.908/39.256 ms at context 8,192. Packed buffers are application-warm and no
model or storage I/O occurs. `A` and `U` are not applicable.

The paired 8,192-token runs are:

| Format | Serial GPU medians ms | Parallel GPU medians ms | Mean speedup |
| --- | --- | --- | ---: |
| Turbo3 | 931.238, 931.809 | 31.920, 31.793 | 29.242× |
| Turbo4 | 882.966, 883.343 | 30.078, 29.901 | 29.449× |

Both orders independently exceed the predeclared 8× requirement.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Every format/context pair passes. Context 17 proves the less-than-32 and
nonmultiple-of-32 partition case; 128, 1,024, and 8,192 prove progressively
larger exact partitions. Guards remain intact and outputs remain finite.

Metal-versus-scalar relative L2 ranges from `2.24e-7` to `3.46e-6`, below the
`2e-4` limit. Maximum absolute error is at most `2.99e-7`, below `3e-4`.
Associative softmax merging therefore preserves the PW-0021 semantic path.

Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0022`. The SHA-256 of its
`SHA256SUMS` manifest is
`13fee0dd425c980ba8fb54c7769fc34ae3633004d7d022e233547895c9109b57`.

## Decision

Promote the 32-lane schedule as the synthetic accelerated attention default.
Retain PW-0021's serial kernel as the independent Metal oracle. The mechanism
delivers a repeatable 29.2–29.4× gain and repairs the practical bottleneck
without changing packing or attention semantics.

This remains a one-head component kernel. It excludes GQA head scheduling,
all 48 layers, projections, RoPE, KV append, model weights, and endpoint work;
the numbers are not TPS. Next integrate multiple Q/KV heads and both MiMo
attention variants around this reduction, then validate real layer states.
