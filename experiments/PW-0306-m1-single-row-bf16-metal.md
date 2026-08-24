# PW-0306 — M1 single-row BF16 Metal specialization

- Status: running
- Disposition: unexecuted
- Date: 2026-08-24
- Owner: Codex
- Commit and dirty state: candidate implementation is clean
  `6957e3b2ddd555618e1d571a639ad2e344015c18`; batch-eight control is clean
  `ba50be23ae922dc83ead86fc4c119e15cb28c1ac`
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

The first candidate/control pair uses the existing PW-0211 ordinary native-MTP
q4 endpoint for seven verifier-authorized tokens. Candidate and control preserve
identical token IDs, `A=[3,3]`, `U=[6.053191489361702,5.377659574468085]`, and
427,197,245,056 logical source bytes.

Candidate target-verification wall is 52,407.856 ms versus 54,124.852 ms for
control, a `1.032762x` improvement. External CPU proposal timing moves in the
opposite direction (21,926.726 versus 21,129.913 ms), so it is not attributed
to the kernel. Candidate post-prefill proposal-plus-verification wall improves
`1.012379x` overall.

The first control attempt completed its 248,125-ms prefill but failed closed
before transaction acceptance because its external child authenticated the main
candidate worktree's commit. Candidate and failed-control target-hidden SHA-256
were identical (`5df877426383c5750a09c0d54e9d992d3d3f99e9f0c15ee5eaece5312659240c`).
The corrected control configuration points both script and working directory at
the control worktree; its hash is
`b6403b4a88cf4f660325ec89845b0fb58b9c7b81b3f86788850a8b1952630373`.
The failed child report is preserved with hash
`d3b4e0f967829fef54022ebda7011c95395a942904d8baf3609689a74e19eb5d`.

## End-to-end result

Candidate 1 completes in 324,809.095 ms at `0.0215511` accepted TPS. The
interleaved control completes in 329,247.747 ms at `0.0212606` accepted TPS.
Candidate 1 therefore improves complete wall/accepted TPS `1.013665x`.
Prefill is reported separately (249,266.689 versus 253,282.758 ms) because the
kernel primarily affects post-prefill one-row execution. Candidate 2 is pending
before repeatability or promotion is decided.

## Correctness result

All 106 Rust tests pass on the Apple M1, including a new deterministic 129-by-257
tail fixture that compares the single-row and batch-eight kernels after BF16
staging byte for byte. The release build passes. Candidate and control generate
identical seven-token output, acceptance, routes, and source-byte accounting.
Both record zero swap growth and zero new throttled pages; minimum free memory
is 64% and 65%, respectively.

Candidate 1 report SHA-256 is
`d822636eb9732649d10fde438eebe366338ae76a312aae93a894be47f462fefd`;
its progress log hashes to
`ee062ad99c69f1557f3837a3601771bb722be2256bba1c0a73bc4909789c9000`.
Control report SHA-256 is
`c4ca45d10d06f8377bba74882bf4fcc140919191c23518736991d9325ffa1c61`;
its progress log hashes to
`4704d53bdd55c8503f65fa3de733760fdc916d824a311f811feaa9fd1c0eefb7`.

## Decision

Pending target-machine evidence.
