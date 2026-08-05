# PW-0025 — Shared-KV MiMo GQA attention

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `06792e8`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0020 Atomic source lock
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift/Metal runtime compiler; internal SSD
- Related records: PW-0023, PW-0024

## Hypothesis and mechanism

PW-0023 redundantly dequantizes each KV row once for every Q head in its GQA
group. A threadgroup per KV head can cooperatively dequantize small K/V token
tiles into shared memory, let 16 global or eight SWA query simdgroups consume
each tile, and distribute V output columns across lanes. This should reduce
both unpack work and threadgroup-memory pressure.

## Contract

Same target-faithful head mapping and PW-0024 attention semantics, with the
same modified Turbo formats. Pass only if:

1. each packed K/V element is dequantized once per KV-head tile, then consumed
   by every mapped Q head; no Q head, KV head, token, or V column is omitted or
   duplicated;
2. global 64Q/4KV and SWA 64Q/8KV modes preserve partial RoPE, V scale 0.707,
   and SWA sink semantics;
3. Turbo3 and Turbo4 pass context 128 in both modes and context 8,192 globally,
   agreeing with the PW-0024 scalar reference at relative L2 at most `4e-4`
   and maximum absolute error at most `7e-4`, with all guards intact;
4. at global context 8,192, two paired orders per format (`PW-0024 schedule,
   shared` then `shared, PW-0024 schedule`) use 10 warm-ups and 30 measurements.
   Shared-KV mean GPU median must be at least 2× faster;
5. report batch one, concurrency one, one accepted token, cold/warm wall and GPU
   median/p95, logical bytes, hardware, commit, and warm packed buffers. `A`
   and `U` are not applicable.

Passing promotes only the shared-KV component schedule. Failure preserves the
correct PW-0024 schedule and rejects this tiling topology. Neither outcome is
endpoint TPS or learned-model fidelity evidence.

## Baseline and candidate

Baseline is PW-0024's one-32-lane-threadgroup-per-Q-head schedule. Candidate is
one threadgroup per KV head with tiled cooperative dequantization, one simdgroup
per mapped Q head, simd reductions for QK scores, distributed V accumulation,
and exact output reconstruction.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0025`.

## Isolated attribution

All candidate runs use batch one, concurrency one, one accepted token, 10
warm-ups, and 30 measurements:

| Format/mode | Context | Logical bytes | GPU median / p95 ms | Wall median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Turbo3 global | 128 | 142,336 | 0.371 / 0.376 | 0.592 / 0.698 |
| Turbo3 global | 8,192 | 4,980,736 | 13.247 / 13.366 | 13.629 / 13.702 |
| Turbo3 SWA | 128 | 219,392 | 0.365 / 0.369 | 0.567 / 0.593 |
| Turbo4 global | 128 | 169,984 | 0.239 / 0.294 | 0.451 / 0.567 |
| Turbo4 global | 8,192 | 6,750,208 | 13.020 / 13.047 | 13.408 / 13.569 |
| Turbo4 SWA | 128 | 274,688 | 0.360 / 0.367 | 0.578 / 0.658 |

Cold GPU/wall times range from 0.252/0.991 ms to 19.334/20.683 ms. Packed
buffers are warm with no model/storage I/O. `A` and `U` are not applicable.

Paired 8,192-token process results:

| Format | Baseline GPU medians ms | Shared GPU medians ms | Mean speedup |
| --- | --- | --- | ---: |
| Turbo3 | 118.342, 118.563 | 13.230, 13.211 | 8.960× |
| Turbo4 | 113.190, 113.393 | 13.049, 13.055 | 8.680× |

Both formats exceed the predeclared 2× gate in both orders. Using candidate
Turbo4 medians gives a nine-global-plus-39-SWA attention-core diagnostic of
131.22 ms at context 8,192, down from PW-0023's 1,264.42 ms. This excludes all
other model work and is not endpoint TPS.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All five conditions pass. Every output and guard passes for both GQA ratios,
both formats, context 128, and global context 8,192. Metal-versus-scalar
relative L2 is at most `1.46e-6`, below `4e-4`; maximum absolute error is at
most `6.86e-7`, below `7e-4`.

The candidate cooperatively dequantizes each eight-token KV tile once per KV
head. One simdgroup per mapped Q head consumes the tile, distributes output
columns across lanes, merges sink mass where required, and performs inverse
WHT in shared memory.

Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0025`. The SHA-256 of its
`SHA256SUMS` manifest is
`9cf9253246c341fd0aa405de489415384bb0df2515bd69a1060259c9e0b5a3a8`.

## Decision

Promote shared-KV tiling as the synthetic GQA attention default. Retain the
per-Q-head schedule and scalar path as controls. The 8.7–9.0× gain is
repeatable and semantic-preserving.

This still excludes learned projections, norms, real KV distributions, and
full layers. Advance it into the transformer-layer fixture; do not convert its
timing into accepted TPS.
