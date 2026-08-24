# PW-0306 — M1 single-row BF16 Metal specialization

- Status: complete
- Disposition: production lower milestone
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

The executable batch-eight control is clean commit
`ba50be23ae922dc83ead86fc4c119e15cb28c1ac`; it retains the candidate kernel
and fixture but forces `bf16_gemm8_shared_weight` for every width. Candidate 1
is clean runtime commit `6957e3b2ddd555618e1d571a639ad2e344015c18`;
Candidate 2 is the same runtime code at clean evidence commit
`24ed19d0e5aee7e49f815c0918dc3bc557a47320`. Both select
`bf16_gemv_shared_weight` only for one row.

All three runs use the internal APFS checkpoint at revision `63651580`, model
lock SHA-256 `df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050`,
checkpoint verification SHA-256
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`,
kernel SHA-256
`fbd4bf7b8b7befc3ac4bbbebe27bebb311852e837735f63779389c155b01bf6e`,
the PW-0208 ordinary prompt, seven requested/accepted tokens, cold process
start, batch one, and concurrency one.

## Isolated attribution

The first candidate/control pair uses the existing PW-0211 ordinary native-MTP
q4 endpoint for seven verifier-authorized tokens. Candidate and control preserve
identical token IDs, `A=[3,3]`, `U=[6.053191489361702,5.377659574468085]`, and
427,197,245,056 logical source bytes.

Candidate target-verification walls are 52,407.856 and 53,969.804 ms versus
54,124.852 ms for control. Their 53,188.830-ms median improves that interval
`1.017598x`. External CPU proposal wall varies independently, so it is not
attributed to the kernel. Candidate median post-prefill proposal-plus-
verification wall is 73,474.074 ms versus 75,254.765 ms, a `1.024236x` gain.

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

The candidate-control-candidate complete walls are 324,809.095, 329,247.747,
and 327,252.117 ms. Candidate walls differ by 0.749%; their 326,030.606-ms
median is `0.0214704` accepted TPS versus control's `0.0212606`, a repeatable
`1.009868x` complete accepted-TPS gain. Candidate 1 and Candidate 2 are each
faster than control.

Prefill walls are 249,266.689, 253,282.758, and 253,920.833 ms. The candidate
mechanism is active in post-prefill one-row execution, so the independently
positive `1.024236x` post-prefill result is retained alongside—not substituted
for—the smaller accepted complete-path gain.

## Correctness result

All 106 Rust tests pass on the Apple M1, including a new deterministic 129-by-257
tail fixture that compares the single-row and batch-eight kernels after BF16
staging byte for byte. The release build passes. Candidate and control generate
identical seven-token output, acceptance, routes, and source-byte accounting.
All three record zero swap growth and zero new throttled pages; minimum free
memory is 64%, 65%, and 65%. Process reads are 420,400,340,992,
421,407,571,968, and 420,161,712,128 bytes. Peak resident-byte ledgers are
4,495,114,240, 4,506,091,520, and 4,503,748,608 bytes. The candidate adds no
resident model state.

Candidate 1 report SHA-256 is
`d822636eb9732649d10fde438eebe366338ae76a312aae93a894be47f462fefd`;
its progress log hashes to
`ee062ad99c69f1557f3837a3601771bb722be2256bba1c0a73bc4909789c9000`.
Control report SHA-256 is
`c4ca45d10d06f8377bba74882bf4fcc140919191c23518736991d9325ffa1c61`;
its progress log hashes to
`4704d53bdd55c8503f65fa3de733760fdc916d824a311f811feaa9fd1c0eefb7`.
Candidate 2 report SHA-256 is
`0e0c18446c0122fc742eb4115e2a85e63aad9ccad667e9c246eee709194c6f2e`;
its progress log hashes to
`a60deaea144130bc32edbaa3affc0fa939a04cef14ebeb7046d2039a54b0a27c`.

## Decision

Promote single-row BF16 Metal selection as the target-M1 default. It is an
exact, RAM-neutral, repeatable lower milestone: `1.009868x` complete accepted
TPS and `1.024236x` post-prefill wall on the frozen seven-token ordinary q4
path. It neither changes the 50-TPS status nor reopens the rejected large-cache
branch. The likely larger benefit in same-model q8 proposal loops remains an
unmeasured consequence, not part of this claim.
