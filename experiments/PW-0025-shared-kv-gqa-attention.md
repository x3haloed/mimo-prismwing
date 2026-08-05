# PW-0025 — Shared-KV MiMo GQA attention

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `4b54921`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
