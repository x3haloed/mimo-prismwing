# PW-0162 — global-attention top-20%-history oracle

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0157 512-prefix route authority
  `32fa8954e875e6c8c53b5092827820940f51225d2bf24322caf5b782295004b9`;
  PW-0158 and PW-0161 analysis manifests to be authenticated at execution
- Execution mode: target-faithful source pass with a non-causal shadow L3
  diagnostic; the oracle output never enters model state
- Related records: PW-0020 through PW-0029, PW-0112, PW-0151, PW-0157 through
  PW-0161; E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0158 rejects ordinary dense 1M attention on the inexpensive two-P100
candidate. PW-0161 rejects the standard 32-GB V100 even at its favorable
Tensor peak and leaves V100S outside the complete current cost envelope. The
remaining cheap-hardware premise must therefore change the quadratic global-
attention work rather than merely tune its kernel.

For two P100s plus the owned EPYC, TARGET's 1,800-second limit grants
`68,656,320,000,000,000` impossible FLOPs. After mandatory matrices and exact
sliding attention, only `38,810,714,295,992,320` FLOPs remain for the nine
global layers: `21.056139%` of their ordinary work. Retaining only 20% of
history leaves a tiny idealized overhead margin and is the concrete mechanism
to falsify first.

Ask the most favorable local numerical question before inventing a selector:
on exact source states, if an oracle already knows the largest source
attention probabilities, does renormalizing the best 20% of visible value rows
preserve each sampled global-attention head output? Failure kills simple token
pruning; success does not make the non-causal oracle executable.

## Shared construction and compression-depth contract

Capability invariant: the authoritative pass preserves all source weights,
all positions in the frozen prefix, every head and dimension, exact source
routes, exact dense and routed layers, and native attention semantics. The
observer is shadow-only and must not change hidden states, caches, routes, or
the source output.

Authorized embodiment boundary: the candidate is explicitly L3. It may keep
a strict subset of global-attention history, compact selected rows in original
causal order, renormalize retained source probabilities, and use the source
four-lane value-dot reduction with final BF16 rounding. Grant retained-mass
summation and probability renormalization in F32; this deliberately favors the
oracle over a strict BF16 candidate and makes a failure stronger. It may not
call this oracle selection a realizable runtime, extrapolate 512 positions to
one million, or waive hosted and capability gates.

## Contract

1. Authenticate TARGET, config, checkpoint verification, the frozen original
   PW-0156 fixture, PW-0157's exact 512-prefix route authority, PW-0158's global
   attention ledger, and PW-0161's complete arithmetic by SHA-256. Require the
   same 512 input-token hash and exact route-trace hash after observation.
2. Add a deterministic tiny correctness fixture before the real walk. Given
   frozen probabilities and value rows, require exact retained-count rounding,
   descending-probability selection with lower-index tie choice, original-index
   compact execution order, renormalization, and a bit-exact 100% control.
3. Walk the first 512 positions of the original frozen 8K corpus once with the
   target-faithful CPU source path. Observe only the nine global layers and
   absolute query positions `63, 95, ..., 511`; observe all 64 query heads.
   Reject any missing layer, position, head, or non-finite value.
4. Freeze retained-history fractions `1%`, `5%`, `10%`, `20%`,
   `21.056139043683178%`, `25%`, and `100%`. For each sampled head-query,
   select `max(1, ceil(fraction * visible_positions))` source positions. Report
   retained probability mass, reference/output norms, relative L2, and maximum
   absolute error without aggregating away the raw distribution. Record the
   favorable F32 renormalization grant in the raw evidence identity.
5. Require the 100% control to reproduce every observed source head output
   bit-exactly. Require the observer run's route hash to equal PW-0157's exact
   route hash. Either failure invalidates the experiment rather than rejecting
   pruning.
6. The 20% continuation gate requires aggregate relative L2 at most 1%, every
   global layer at most 2%, and head-query relative-L2 p99 at most 5%. These are
   phase-A falsification thresholds, not TARGET acceptance thresholds. Report
   the exact `21.056139%` boundary separately.
7. Kill simple probability-ranked 20%-history pruning if the oracle fails any
   continuation threshold. Because an implementable selector has less
   information than this oracle, it cannot repair the same fixed subset
   mechanism without changing the premise.
8. If the oracle passes, promote only a phase-B experiment: a causal selector,
   accumulated 512-position candidate state, route/logit comparison, then
   broader held-out and true-long-context hosted gates. Do not promote a
   kernel, hardware purchase, 1M capability, or endpoint from this result.
9. Apply Gate 8 with phase-level RSS, physical footprint, memory-free, swap,
   throttling, release-boundary, and protected-service checks. Record zero
   accepted tokens and no endpoint TPS.

## Promotion and kill rule

Reject the experiment itself on source identity drift, sample incompleteness,
observer non-interference failure, 100% control mismatch, missing release
evidence, or Gate-8 failure.

If the valid 20% oracle fails aggregate 1%, any-layer 2%, or p99-head 5%, kill
simple global-history pruning at the arithmetic fraction required by the
cheap two-P100 envelope. This does not kill learned linear/recurrent attention,
changed weights, retrieval with repair, or a faster future card.

If it passes, retain only the numerical possibility. A causal selector must
still avoid computing the discarded scores, fit the `$500`/power system, and
pass accumulated local, hosted distributional, capability, modality, and
one-million-token gates.

## Result

Pending implementation and execution after PW-0157 releases the shared host.
