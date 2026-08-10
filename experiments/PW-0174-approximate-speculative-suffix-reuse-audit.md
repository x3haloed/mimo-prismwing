# PW-0174 — approximate speculative suffix-reuse audit

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0170 exact `q=137`
  A770/storage envelope; TARGET L3 behavioral gates
- Execution mode: primary-source L3 mechanism audit and arithmetic comparison;
  no model execution or endpoint claim
- Related records: PW-0044, PW-0102, PW-0152, PW-0170, PW-0173; E2 and E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0173 rejects current exact/L2 released proposer configurations as too short
for PW-0170's minimum `A=56`. Approximate Speculative Decoding (ASD) changes a
different premise: it can accept bounded low-regret target/draft mismatches and
then reuse a suffix that remains target-greedy under the altered prefix. The
project's external target permits documented L3 differences if every hosted
distributional and capability gate passes, so exact-token rejection alone
cannot dismiss this mechanism.

Ask whether the released ASD configurations either span PW-0170's minimum
accepted horizon or already supply evidence strong enough to qualify their
changed trajectories for Prismwing's declared near-equivalence target.

## Exactness and scope

ASD is explicitly L3 for Prismwing because accepted mismatch tokens change the
target trajectory. Its local regret budget is not a substitute for TARGET's
100,000-position distributional, modality, long-context, capability, and
sequence gates. Reported task scores and hashes are useful evidence but cannot
be promoted across missing slices.

## Contract

1. Authenticate TARGET, PW-0170, PW-0173, and an immutable capture of the ASD
   paper by SHA-256. Record the canonical paper URL and capture hash outside
   Git.
2. Extract the released primary configuration, drafter horizon, verifier
   controls, mean strict/ASD accepted lengths, throughput change, hash
   divergence, and worst reported task-score change directly from the paper.
3. Grant a target bonus token when translating the DSpark block-seven horizon.
   Do not infer a longer accepted path from request regret budget `B=8`, which
   bounds cumulative approximation rather than proposal depth.
4. Reject the released configuration as PW-0170's direct proposer if its
   favorable maximum path is below `A=56`. Retain only a separately scaled
   `q>=137` ASD experiment if it reaches 56.
5. Independently require all TARGET distributional/capability slices before
   calling the L3 trajectory near-equivalent. Paper task accuracy without
   hosted logprobs, native modalities, long context, and confidence intervals
   is insufficient even when point scores are favorable.
6. Record the paper's favorable gains and unfavorable divergences without
   converting either into Prismwing TPS or a universal impossibility claim.
7. Apply Gate 8 to the local analyzer/source-capture phase. Record zero
   accepted Prismwing tokens, no endpoint TPS, and no purchase authority.

## Promotion and kill rule

Promote only if a released configuration reaches a favorable path of at least
56 and has evidence covering the declared L3 gates. Otherwise reject that
configuration as a direct Prismwing mechanism. Preserve a separately named
scaled MiMo-specific ASD branch as unproven rather than impossible.

## Result

Pending source capture, fail-closed analyzer, and execution from a clean
implementation commit.
