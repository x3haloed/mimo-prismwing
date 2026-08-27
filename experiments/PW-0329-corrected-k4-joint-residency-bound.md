# PW-0329 — Corrected K4 joint-residency Prismwing-1 bound

- Status: proposed; unexecuted
- Disposition: unexecuted
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0136, PW-0207, PW-0316, PW-0318, PW-0325,
  PW-0327, PW-0328

## Question

Does any evidence-backed onboard K4 route remain above one sustained accepted
token/s after the repaired PW-0328 acceptance and routes, the complete q8
target shared-weight traffic, and one joint common-plus-expert residency budget
are charged before constructing another K4 identity?

This is a joint-residency and correctness-density upper bound, not a rerun of
PW-0325's expert-only selector. Companion storage, memory, compute, and
proposal work are inadmissible.

## Prediction error and input freeze

The first post-PW-0327 planning calculation used either
`7,743,236,992` bytes, the fixed shared-weight scan, or `7,743,294,336`
bytes, PW-0207's selected resident source set, as a complete q8 verifier-pass
constant. Neither is that quantity.

The fixed 381 shared checkpoint objects sum to `7,743,236,992` logical source
bytes. The runtime additionally gathers one `4,096 × BF16 = 8,192`-byte token
embedding row per verifier position, so exact q8 target shared traffic is
`7,743,302,528` bytes. PW-0207's `7,743,294,336`-byte source set contains the
381 fixed objects plus only seven reusable embedding rows; its
`7,745,585,152`-byte value is page-aligned resident allocation, not traffic.
Keep logical traffic, physical reads, and resident allocation separate.

PW-0328 is still capturing while this record is first committed, so no corpus
hash exists to freeze yet. This proposal is deliberately non-executable. After
the complete builder authenticates all four raw captures, amend this record in
a clean commit with the exact canonical PW-0328 manifest path and SHA-256.
Only then may the analyzer be authored, and it must hard-code that hash rather
than accept a caller-asserted digest or an older PW-0208 route.

## Authorities and fail-closed authentication

The executable contract must authenticate and independently close all of the
following:

1. The completed PW-0328 schema, evidence semantic, clean capture/builder
   commit, and exact manifest hash. It must contain exactly 32 primary windows:
   transaction indices zero through seven in each of four frozen categories.
   Use full verifier-authorized `A`, never terminally clipped `observable_A`.
2. Every raw generation, progress, verifier-hidden, prefill-report, and
   prefill-hidden hash named by PW-0328. Reconstruct every eight-expert route
   row, per-layer union, `(layer, expert)` identity, and `U`; all must close to
   the manifest. Transaction-zero prefill history remains mandatory.
3. PW-0318 summary SHA-256
   `a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f`
   for source-FP8 `25,171,968`-byte and K4 `12,654,604`-byte executable expert
   records and the narrow `(3,5)` boundary.
4. PW-0316 rejection SHA-256
   `7e5560cf2cdc2abdec8ec1a17af0462f69fa7204f8ba528808ce1f046d0e6ff4`:
   its four-K4/four-source routed row reaches `0.0109888419` relative L2 and
   fails the unchanged exclusive `0.01` gate. Higher density is therefore a
   falsifier premise, never already-qualified fidelity.
5. PW-0207 `offline-002.json`, SHA-256
   `1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6`,
   the receipt-authenticated checkpoint index, and the local tensor census.
   Independently sum the 381 fixed objects to `7,743,236,992` source bytes and
   `7,745,470,464` page-aligned allocation bytes. The largest object is the
   `1,249,902,592`-byte LM head.
6. PW-0136's authenticated cold internal-SSD transport evidence and exact
   `3,470,448,309.677419` logical bytes/s constant.
7. The current `TARGET.md`, `RED_LINES.md`, model revision, hardware, batch one,
   concurrency one, Gate 8 limits, and explicit companion-hardware exclusion.

The exact traffic model includes eight embedding rows per q8 verifier. The
strongest relaxed kill model may omit those `65,536` bytes as an explicitly
favorable grant. It may not omit fixed shared objects, add another scan for the
already-computed target bonus, or double-spend residency between common and
expert objects.

## Predeclared scenarios

Evaluate density limits `d ∈ {3, 6, 8}` against total joint residency budgets
`R ∈ {4, 6, 8, 12} GiB`.

- Four GiB is the demonstrated post-phase residency scale.
- Six and eight GiB retain comparability with PW-0325 and are analytical until
  a concrete cache manifest passes.
- Twelve GiB is the target's maximum declared persistent working set. It is an
  impossible-best, pressure-conditional ceiling, not an authorized process:
  the separate 13 GiB process ceiling still must hold with K/V, proposer,
  runtime, staging, and arenas.

For each `(d, R)`, emit two distinct models.

### Relaxed density ceiling

For window `w`, routed layer `l`, and authenticated union cardinality `n_wl`,
grant

```text
k_wl(d) = min(n_wl, 8d)
```

K4 records and charge every remaining record as source FP8. This ignores
cross-row overlap, global identity consistency, construction, and whether one
fixed bank can realize the placement. It is intentionally more favorable than
an executable bank and is the byte-floor kill authority.

### Row-feasible fixed bank

For `d ∈ {3, 6}` construct deterministic candidates capped at 4,096 global
identities. Report `d=8` only as an unqualified diagnostic.

Start with every identity source FP8. An identity may be selected only when
adding it leaves every authenticated eight-expert route row at or below `d`
selected identities. Charge a selected identity as K4 in every window and
layer union in which it appears. For category storage targets
`{1.10, 1.25, 1.50}`, choose the identity maximizing the sum across deficient
categories of clipped byte reduction divided by that category's initial byte
deficit. Resolve exact ties by layer then expert.

Emit the complete order and hash, selected count, row-density histogram,
layer/category coverage, installed hybrid bytes, construction-artifact bytes,
M1 construction seconds, and every rejected selection caused by the row cap.
A greedy miss is not an impossibility proof; the relaxed ceiling remains the
only hard byte-floor authority.

Also replay PW-0325's exact 3,925-identity order on the new corpus for historical
comparison. Do not re-label its old `A`, routes, or expert-only cache result as
current evidence.

## Byte, cache, and throughput formulas

For a relaxed or fixed-bank placement:

```text
E_w = sum_l(k_wl * 12,654,604 + (n_wl - k_wl) * 25,171,968)
S_fixed = 7,743,236,992
S_q8_exact = S_fixed + 8 * 8,192 = 7,743,302,528
```

The maximally favorable fractional joint-residency miss floor is:

```text
M_fractional_w = max(0, S_fixed + E_w - R)
t_storage_w = M_fractional_w / 3,470,448,309.677419
TPS_storage_w = A_w / t_storage_w
```

This ceiling omits embedding rows, grants perfect future knowledge, permits
byte-fraction packing and free reshaping between windows, and charges no fill,
compute, attention, routing, native-MTP proposal, synchronization, sampling,
rollback, or endpoint work. It is never achieved TPS.

Report an exact-logical variant using `S_q8_exact`. Report a conservative
whole-object allocation model using 16 KiB page rounding:

```text
K_alloc = 12,664,832
source_alloc = 25,182,208
S_alloc = 7,745,470,464
L = max(1,249,902,592, largest selected expert allocation)
M_guarded_w = 0                               if total_alloc <= R
              total_alloc - max(0, R - L)     otherwise
```

The largest-object slack prevents a fractional packing claim. Page allocation
belongs only to this residency model and must not be charged again as logical
traffic.

For every window, category, and scenario report `A`, `U`, route identities,
row hits, logical/allocated bytes, cache credit, storage wall, and optimistic
storage TPS. Category and aggregate TPS are token totals divided by wall-time
totals. Nearest-rank p10 over 32 window TPS values is the fourth-lowest value.

At a one-TPS target, report omitted-work headroom separately:

```text
H_corpus = sum(A_w) - sum(t_storage_w) seconds
H_per_accepted_token = H_corpus / sum(A_w)
```

Negative headroom is a storage-only rejection. Positive headroom is only the
maximum allowance for every omitted complete-path component; it is not a
prediction that those components fit. Separately report PW-0308's
`351.680083`-ms p90 repeated mixed routed component scaled by `sum(U)` as a
diagnostic, never as a theorem for all-K4 distinct layers or a hard gate.

## Frozen gates and disposition order

1. **Absolute corpus byte rejection.** On the strongest relaxed `d=8`,
   `R=12 GiB` fractional ceiling, an overall or required-category aggregate at
   or below one TPS rejects the current K4 construction portfolio before
   compute or proposal. This is decisive for the authenticated corpus and
   portfolio, not a theorem about every possible future 512-token generation.
2. **Tail continuation gate.** The same strongest ceiling must have strict
   nearest-rank p10 above one. Failure rejects construction continuation under
   the frozen PW-0325 tail criterion, but is not by itself a universal endpoint
   impossibility claim.
3. **Density survival.** Apply the same strict overall, required-category, and
   p10 gates to relaxed `d=6`, `R=12 GiB`. If `d=8` survives but `d=6` fails,
   close the currently evidenced K4 route: only density seven or eight survives
   storage, while a real four-K4 row already fails and no higher-density
   semantic premise is qualified.
4. **Construction-prerequisite margin.** One row-feasible bank of at most 4,096
   identities must reach at least 1.25 guarded storage TPS overall and in every
   category, strict fractional p10 above one, a smaller installed bank than all
   source, complete selector/authority closure, and Gate 8. The 1.25 threshold
   reserves only 0.2 seconds per accepted token for omitted work and still
   authorizes no construction.
5. If a row-feasible `d=3` bank clears the margin, authorize only native-MTP q8
   acceptance/latency on PW-0328 histories, an early/middle/deep arbitrary-
   identity fidelity panel, and then a production-shaped mixed q8 layer.
6. If only `d=6` clears, authorize first the bounded six-of-eight falsifier:
   complete PW-0316's four missing identities and exhaust all 28 six-K4 subsets
   under the unchanged routed and layer-final gates. Native-MTP acceptance and
   mixed q8 timing remain later prerequisites.
7. Any survivor requiring more than 8 GiB interposes PW-0207's declared-
   residency/pressure-observer requalification and a complete process memory
   manifest. Twelve-GiB analysis is not physical authorization.

If relaxed density survives but no fixed bank clears the margin, emit a
conditional analytical survivor and no work order. Every disposition records
which branch was killed or retained and why.

## Required fixtures

Before execution, add deterministic tests that:

- reject `observable_A` substitution and wrong category/transaction
  cardinality;
- reconstruct raw route rows, unions, identities, and `U` exactly;
- independently sum fixed shared objects and keep embedding rows distinct;
- reject a fourth selected identity in a `d=3` route row;
- reproduce `k=min(n,8d)` and canonical selector ties;
- prove common and expert residency cannot both consume the full `R`;
- cover 16-KiB allocation, largest-object guard, and the fit-all case;
- compute the fourth-lowest p10 and token-total-over-wall aggregates; and
- enforce strongest-scenario disposition precedence.

## Claims excluded

- achieved, measured, or complete endpoint TPS;
- target-faithful labeling for K4 or general fidelity from a density ceiling;
- construction, checkpoint relocation, a cache allocation, or a runtime
  default;
- a qualified proposer, compute overlap, multimodal/long-context performance,
  or final TARGET.md promotion;
- companion hardware in any form.

The report accepts zero tokens and must emit `performance_claim: null`.
