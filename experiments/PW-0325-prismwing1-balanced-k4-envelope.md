# PW-0325 — Prismwing-1 category-balanced K4 envelope

- Status: complete
- Disposition: continuation approved — bounded six-of-eight density falsifier
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0181, PW-0208, PW-0216, PW-0318, PW-0319,
  PW-0320, PW-0324

## Question

Does changing the K4 bank objective from PW-0319's three-hit coverage order to
a deterministic category-balanced byte-reduction order expose a physically
credible onboard path above one sustained accepted TPS before any expensive
bank construction?

## Changed premise

PW-0319's greedy score deliberately stops valuing a routed row after its third
selected K4 identity. That was correct for authorizing the `(3,5)` layer
boundary, but it is not a byte-minimizing order after every row reaches three
hits. PW-0320 reused that order only through 2,048 identities for a two-TPS
falsifier. Prismwing-1 may depend on later identities, so extending the capped
order would silently optimize the wrong quantity.

This experiment changes only the analytical selection objective and the named
throughput tier. It does not qualify another K4 weight, cache policy, proposer,
kernel, or endpoint.

## Authorities and modes

- authenticate PW-0208's 32 corrected primary q8 windows and their exact `A`,
  category, route, and identity authorities;
- authenticate PW-0318's `(3,5)` K4/source executable-record sizes and
  component boundary;
- authenticate PW-0319 and PW-0320, preserving their decisions and canonical
  hashes;
- use PW-0136's measured two-worker cold internal-SSD bandwidth;
- keep source FP8 as the target-faithful control and every selected K4 identity
  explicitly L3 modified;
- keep companion capacity, storage, and compute excluded.

## Deterministic selector

Evaluate 4, 6, and 8 GiB expert-cache capacities. Eight GiB is the largest
analytical point and remains an oracle until a complete process manifest fits
under Gate 8 with retained K/V, native MTP, common weights, staging, and
eviction headroom.

For each cache capacity and category storage target in `{1.10, 1.25, 1.50}`
accepted TPS:

1. Begin with every routed identity in source FP8.
2. Give every window only the conservative guaranteed cache credit
   `capacity - one source record`; also report the stronger exact whole-record
   subset oracle separately.
3. At each step select the unchosen `(layer, expert)` that maximizes the sum,
   across still-deficient categories, of its clipped normalized byte reduction.
   Ties resolve by layer and then expert. One identity is charged K4 bytes in
   every window where it occurs.
4. Stop only when all four category aggregates clear the requested
   storage-only target or no identity can reduce a deficit.
5. Emit the complete deterministic order, its content hash, layer/category
   coverage, executable and construction bytes, M1/M4 construction estimates,
   exact per-window `A`, identities, bytes, storage wall, and optimistic TPS.

The selector must reproduce byte-for-byte without a solver-dependent tie or
future route information. A category aggregate is total accepted tokens divided
by total storage wall, not a median of window ratios.

## Continuation gates

A candidate physical envelope survives only if one deterministic bank of at
most 4,096 identities, with no more than 8 GiB oracle cache:

- reaches at least `1.25` optimistic storage-only TPS in every text category;
- reaches at least `1.0` optimistic storage-only TPS at the nearest-rank p10
  window;
- leaves an installed hybrid expert bank smaller than the all-source expert
  bank and records the non-coexistence installation requirement on the current
  internal SSD;
- records no endpoint or accepted-token performance claim; and
- passes authority and Gate 8 checks.

A pass authorizes only three cheap prerequisites, in this order:

1. complete the four missing K4 identities for PW-0316's authenticated
   layer-4 eight-expert row, enumerate every six-K4/two-source subset, and kill
   the density premise if even the impossible-best subset misses either
   unchanged exclusive one-percent routed or layer-final gate;
2. if density survives, measure recursive native-MTP q8 acceptance on already
   authenticated hidden histories, without constructing a bank;
3. if acceptance survives, measure a production-shaped q8-batched mixed
   K4/source layer transaction before selecting a construction tranche.

The four-identity density falsifier is a bounded correctness experiment, not a
bank work order. No construction tranche is authorized until all three
prerequisites pass and a separately frozen early/middle/deep semantic-risk
panel demonstrates that the selected frequency-biased identities can satisfy
the unchanged K4 correctness ladder. Failure of the physical gate restores
PW-0181's onboard one-TPS closure for the current portfolio.

## Claims excluded

- achieved or modeled endpoint TPS;
- general K4 fidelity, full-capability promotion, or a runtime default;
- a real causal cache policy or free cache installation;
- construction or relocation of the source checkpoint or large artifact bank;
- Prismwing-2, Prismwing-50, or a target rewrite.

## Result

Clean analyzer commit `371431ca7febc5982401159e48e8b8d269f729ac`
authenticated PW-0208, PW-0318, PW-0319, and PW-0320, including independent
cross-checks of the corrected route authority. It evaluated every predeclared
cache/target pair with a deterministic canonical tie break and emitted each
window's exact `A`, `U`, identities, representation, physical bytes, storage
wall, and optimistic storage-only TPS.

The only predeclared candidate point uses an oracle 8 GiB cache and the 1.25
category target. Its selector chooses 3,925 identities; the order hashes to
`d5a68bb4291076fbf62c8def45837b6a948d06438bbde298077fdec380f6b25a`.
Those identities cover `75.446470489%` of observed identity-window
occurrences. Conservative category storage-only ceilings are `1.250026` code,
`1.315182` ordinary, `1.370841` multilingual, and `1.308061` rare-route TPS.
The stronger exact whole-record oracle yields `1.251916`, `1.317077`,
`1.372790`, and `1.309836` respectively.

Across all 32 authenticated q8 windows, observed `A` totals 213 and the exact
oracle moves `562,657,289,624` bytes in `162.128129` modeled storage seconds:
`1.313776` aggregate optimistic TPS. Nearest-rank p10 is `1.149252`; the worst
window is only `0.440463`, so this is neither a per-window guarantee nor an
endpoint claim. Batch size and concurrency are one. Prefill is excluded, and
no cache was populated or timed.

The installed hybrid expert-bank estimate is `253,738,465,276` bytes versus
`302,869,118,976` all-source bytes, but construction would create an estimated
`117.75` GB of artifacts and take `1,962,500` M1 seconds under the inherited
per-identity constant. Current internal storage cannot hold source and build
artifacts concurrently; this experiment authorizes neither relocation nor
construction.

All five continuation gates pass, including Gate 8. The canonical report is:

`/Users/chad/Models/mimo-prismwing/evidence/PW-0325/analysis-002/analysis.json`

SHA-256:
`9391b3b8bc8b4264ec1e74378743f00780f724eab953fb31841434d0516e81c1`.

Analysis-001 hashes to
`ec5ac8ae869ab76dc973d36b3969c0af889261167eb103db1412d7a6c5653a98`
and is preserved but superseded. It produced the same numerical decision but
omitted the contract's exact per-window identity list, did not carry `U`, and
did not cross-check the recomputed route hash against both upstream reports.

## Decision

Authorize only a bounded impossible-best six-K4/two-source density falsifier.
PW-0325 demonstrates a storage envelope worth testing, not a way to
run the model at one TPS: K4 fidelity at the required density, a causal cache,
q8-batched compute, common weights, attention, proposal, sampling, and the
complete endpoint all remain unqualified. No runtime default changes.

Subsequent source audit found that the full-match q8 commit violates PW-0204's
target-bonus and next-anchor contract. That prediction error does not change
this report's historical bonus-free arithmetic, but it preempts density
construction: PW-0326 must repair the transaction and regenerate causal q8
authority first. Only a new envelope may decide whether the density falsifier
is still needed.
