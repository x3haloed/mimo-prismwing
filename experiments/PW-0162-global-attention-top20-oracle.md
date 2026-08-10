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
  same-shape observer-disabled control
  `480b02816b293ed8a2275e3c2810ee940fa0916db31fd1d730d6331e9f00a025`;
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
   same 512 input-token hash and exact semantic-route hash after observation.
   Semantic route identity includes layer number, ordered selected experts, and
   ordered route weights only; it excludes attention/cache/union/timing
   metadata.
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
   bit-exactly. Require the observer run's semantic-route hash to equal
   PW-0157's exact semantic-route hash. Either failure invalidates the
   experiment rather than rejecting pruning.
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

The first observer run at commit
`e02139cb2687e9d2a4844c5e390bb3c2ba156926` correctly failed closed with
`PW-0162 shadow observer changed exact source routes`, but the follow-up
observer-disabled same-shape control proved that this was a guard defect, not
observer interference. The control completed in `1,677.895583` seconds and
its raw manifest hashes to
`480b02816b293ed8a2275e3c2810ee940fa0916db31fd1d730d6331e9f00a025`.
Its old `layer_routes_sha256` was
`234b4a078d6bd24fa85b91ffa7365923f43e5531effe3d5980738976c4599710`,
different from PW-0157's
`eff0dd3c993d132bd2ef66008c42c10e7b6b0b604ccad93ba0c72f894023a903`.
However, direct comparison found all `24,064 = 47 * 512` ordered expert rows
and route-weight rows bit-exact. The old hash covered the entire
`LayerRouteTrace`, including nondeterministic `wall_ms`, and therefore could
not be a cross-run route identity.

The corrected semantic payload hashes only layer number, ordered expert IDs,
and ordered route weights. Both the authenticated PW-0157 authority and the
control produce
`c0e5c8fd8c72f148895d39fdf38b95e84e93228206563ea49b242f48b0c69872`.
A deterministic fixture now requires timing-only changes to preserve that hash
and a route change to alter it. No pruning result, endpoint TPS, or mechanism
decision is inferred from the invalid first observer attempt.

The corrected full execution at commit
`a9abb396bb9a44b21a874633d42a8417dc1d1ff2` then failed the semantic guard
after `27` minutes with `PW-0162 shadow observer changed exact source routes`.
Unlike the first failure, the repaired hash makes this genuine observer
interference. The measuring instrument performed allocations, sorting,
renormalization, and repeated attention reductions between authoritative head
computations. Its nominally shadow-only dataflow was therefore not a passive
execution boundary. This invalidates the instrument; it is still not evidence
for or against pruning.

The next implementation preallocates bounded Q/K/V and reference-output
storage before the source walk. During the authoritative pass it only copies
the nine global layers' 15 sampled query rows, their 512-position K/V state,
and sampled source head outputs. It performs no oracle arithmetic until after
the exact semantic-route guard passes. Offline replay must reproduce every
captured source head output bit-exactly before candidate errors are accepted.
A deterministic fixture proves sampled writes reuse fixed-capacity storage and
reject duplicate identities.

That passive-capture attempt ran at commit
`5a7b41c262a89747a9cca4ea14908e3ab436b6b9` for `1,679` seconds and still
failed the semantic route guard before oracle analysis. Its failure manifest
hashes to
`026f116129543b02285d81e20bb0a3a7746c91623f4403a0aa10f262b9d87189`.
This proves that moving sorting, renormalization, and candidate reductions
offline was insufficient. It does not yet attribute the drift to capture
preallocation/copies versus changed-binary or Accelerate behavior. A
same-commit 512-position no-capture control is therefore the next cheap
discriminator. No fourth oracle walk is authorized without emitting the
actual semantic hash and exact expert/weight mismatch counts.

The successor capture implementation uses one anonymous mmap with fixed
offsets and fixed-size identity bitmaps instead of 36 heap-backed tensor and
metadata vectors. Before another 512-position attempt, a same-commit
64-position no-capture authority and capture run must pass exact semantic
routes, bit-exact offline replay, the single frozen query at position 63, and
Gate 8. This smoke gate is an instrumentation check only and cannot adjudicate
the 20%-history mechanism.

The control passed Gate 8: minimum free memory was `73%`, maximum physical
footprint `841,980,032` bytes, peak RSS `933,265,408` bytes, zero swap growth,
zero new throttled pages, `374,131,456` bytes after checkpoint release, and all
protected services remained resident. It reports zero accepted tokens and no
endpoint TPS.
