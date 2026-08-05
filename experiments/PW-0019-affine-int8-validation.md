# PW-0019 — Predeclared affine-INT8 block validation

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `9103991`; clean before this contract
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; committed INT4 and INT8 block
  fixtures
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2
- Related records: PW-0016, PW-0018

## Hypothesis and mechanism

Affine INT8 preserves the complete source-FP8 routed-block output much better
than INT4 while remaining close enough in wall time to become the default MLX
research representation.

## Contract

Predeclared L3 component gate. Use PW-0016's actual layer-43 router, nine
heterogeneous selected experts, exact position schedule, normalized route
weights, and weighted sum. Compare affine group-128 INT4 and INT8 in three
paired process repetitions with order `4,8`, `8,4`, `4,8`. Each process uses
10 warm-ups and 30 measurements. Source load and installation quantization are
excluded; all selected buffers are warm.

INT8 passes only if all conditions hold:

1. every run matches the committed INT8 fixture and source router selection;
2. relative L2 versus source FP8 is at most 0.02 and cosine at least 0.9998;
3. mean of the three INT8 wall medians is no more than 1.20 times the mean of
   the paired INT4 wall medians;
4. all six runs complete without numerical or integrity failure.

Passing promotes INT8 only as the default quality-oriented research substrate.
It does not pass whole-layer, whole-model, distributional, capability, or
endpoint gates. Failure retains INT4 for performance work and returns INT8 to
exploratory status.

## Baseline and candidate

Baseline is affine INT4. Candidate is affine INT8. Commands use
`tools/mlx_moe_block_benchmark.py --bits BITS --fixture FIXTURE` and distinct
immutable external evidence paths under PW-0019.

## Isolated attribution

Pending.

## End-to-end result

Pending. No endpoint claim is in scope.

## Correctness result

Pending.

## Decision

Unexecuted until this contract is committed.
