# PW-0096 — Incremental token cost profile

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes reproducible profile
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0092 run hashes
  `18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25`
  and `ee1151c7a780545df922593b04e4e1c304541824a7a4d761ce42cdab70fa8078`
- Hardware/runtime: Apple M1 shared 16 GiB host; offline analysis of two clean
  real-checkpoint endpoint processes
- Related records: PW-0034 through PW-0043, PW-0092, PW-0095

## Hypothesis and mechanism

The source-exact one-token path is dominated by repeated routed-expert
FP8-to-F32 expansion and execution, not K/V attention or cache management.
Profile the already frozen PW-0092 reports without another full-model run.
Partition step-two wall into layer 0, 47 routed layers, and non-layer remainder;
partition logical bytes into the exact eight-expert-per-routed-layer source
payload and shared spine; partition FP8 expansion counts into expert and shared
matrices. Compare both clean processes and full/SWA routed-layer timing.

## Gates

Fail closed unless both input hashes, schemas, checkpoint/revision/fixture
identities, token IDs, cache lengths, batch/concurrency, and semantic projection
match PW-0092. Every partition must sum exactly to its parent ledger. Promote a
bottleneck belief only if both runs agree within 5% and the selected component
accounts for at least 50% of complete incremental wall or physical work.

Record source hashes, per-run cold-process timing, two-step actual disk reads,
logical bytes, expansion counts, expert executions, layer timing distribution,
and diagnostic token rate. This is post hoc attribution of accepted output,
not a new endpoint run; it cannot promote a performance default or accepted TPS.

## Result

Unexecuted.

## Decision

Unexecuted.
