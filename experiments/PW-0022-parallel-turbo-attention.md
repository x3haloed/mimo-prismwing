# PW-0022 — Parallel MiMo Turbo attention reduction

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `76a2dc6`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
