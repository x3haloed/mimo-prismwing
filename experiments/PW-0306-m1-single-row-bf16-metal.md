# PW-0306 — M1 single-row BF16 Metal specialization

- Status: running
- Disposition: unexecuted
- Date: 2026-08-24
- Owner: Codex
- Commit and dirty state: implementation pending from clean
  `c4142218bdbd00fbf84deddb6cd9661909af3878`
- Hardware: Apple M1 Mac mini (`Macmini9,1`), 16 GiB unified memory, macOS
  26.6.1 (25G76)
- Related records: PW-0195, PW-0207, PW-0211, PW-0215, PW-0216; stronger-worker
  PW-0305 handoff at `/private/tmp/mimo-exact-0.2tps-worker-handoff/`

## Hypothesis and mechanism

The current wide BF16 Metal linear always executes a compile-time batch of
eight, even when attention output projection or the proposal LM head supplies
one row. The stronger-worker handoff adds a one-row kernel with the same
64-lane reduction tree. Selecting it only for `active_rows == 1` removes seven
unused dot products and seven unused output rows without adding checkpoint or
resident bytes.

PW-0207 already established that moving proposal-side one-row LM-head calls
onto the wide BF16 Metal path improves a complete target transaction by
10.324490%. This experiment tests whether deleting its batch-eight padding is
a further M1 gain.

## Contract

This is target-faithful L1, function-preserving scheduling. The one-row kernel
must be byte-identical to the established batch-eight kernel for the real row
after BF16 output staging. Add that deterministic fixture before promotion.
The implementation must leave widths two through eight unchanged and allocate
no resident model state.

Isolated timing is diagnostic. Promotion requires identical endpoint tokens,
accepted-token accounting, `A`, `U`, logical bytes, and safety outcomes, plus
a repeatable interleaved full-path gain on this 16 GiB M1. Kill or retain only
as opt-in research if candidate median accepted TPS does not exceed control.

## Baseline and candidate

Baseline is clean commit `c4142218bdbd00fbf84deddb6cd9661909af3878` using
`bf16_gemm8_shared_weight` for every width. Candidate selects
`bf16_gemv_shared_weight` only for one row. Exact commands, checkpoint hashes,
prompt, cache state, batch, concurrency, accepted tokens, `A`, `U`, process
reads, and candidate commit will be recorded after execution.

## Isolated attribution

Pending.

## End-to-end result

Pending.

## Correctness result

Pending.

## Decision

Pending target-machine evidence.
