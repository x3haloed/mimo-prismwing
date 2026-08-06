# PW-0096 — Incremental token cost profile

- Status: complete
- Disposition: scope-decision
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: profile tool and result prepared after contract at
  `caf26ba`; result commit follows
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

The fail-closed profile accepts both immutable report hashes and their exact
semantic projection after excluding only timing. Its evidence is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0096-profile.json` with SHA-256
`7638d5daafd9d4e5ed39f020e7ab2eca166cbd4d4d9613ef045ef29a1c4d2143`.
The throughput-model input hashes to
`4e8f5558421c90d0b0bebc44ea050c5502ad794a8aea457558970a375006f4bf`.

The retained-cache token takes 158,521.015 and 158,614.709 ms, diagnostic
rates 0.0063083 and 0.0063046 token/s. Relative spread is only `0.0005909`.
The 47 routed layers consume 149,403.983 and 149,533.377 ms, or 94.2487% and
94.2746% of complete incremental wall, with relative spread `0.0008657`.
Their mean layer walls are 3,178.808 and 3,181.561 ms. Full-attention routed
layers average 3,182.259/3,179.184 ms and SWA routed layers
3,178.100/3,182.049 ms, excluding attention policy as the material driver.
Layer zero takes only 2,152.459/2,118.597 ms; all non-layer work, including
final norm/LM head and measurement, takes 6,964.572/6,962.735 ms.

The one-token source ledger partitions exactly into 9,464,659,968 expert bytes
and 7,743,245,184 shared bytes, 55.0018% expert. More decisively, routed
experts cause 1,128 of 1,179 FP8 matrix expansions, 95.6743%, through exactly
376 unique-expert executions. Both complete two-step processes read about
85.457 GB from disk. All repeatability and dominance gates pass.

## Decision

Promote routed-expert FP8-to-F32 expansion and execution as the primary
one-token embodiment bottleneck. Do not spend the next performance experiment
on attention type, K/V compression, tokenizer, or sampling; those cannot
materially change the measured 94.25% routed-layer share at this context.

The first implementation candidate should vertically integrate the already
validated Rust-owned Metal source-FP8 complete-expert primitive into one-row
dynamic routed execution, initially behind an explicit candidate mode. It must
reuse exact source weights/routes, preserve the current exact path as control,
climb from real expert/layer parity to the full endpoint, and earn promotion
only through repeatable complete-token gain and unchanged target-faithful
correctness gates. The candidate is not yet a default or TPS result.
