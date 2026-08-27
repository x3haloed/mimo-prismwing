# PW-0325 — Prismwing-1 category-balanced K4 envelope

- Status: proposed
- Disposition: unexecuted
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

Unexecuted.

## Decision

Unexecuted. Commit this contract before implementing or inspecting canonical
PW-0325 output.
