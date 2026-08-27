# PW-0321 — Stitched corrected wide-horizon diagnostic

- Status: complete
- Disposition: conditional
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0320

## Question

What is the smallest verifier horizon among 16, 32, and 64 whose corrected
expert-union evidence can cross the two-TPS storage bound strongly enough to
justify an expensive real teacher-forced capture?

## Contract

Within each PW-0208 category, union consecutive groups of two, four, or eight
primary width-eight windows. This stitching is a route-diversity diagnostic,
not a causal wide transaction: corrections occur between source windows and
the candidate paths differ. Use the strongest still-numerical PW-0320 premise:
the 2,048-identity bank, perfect free 4 GiB cache, measured 3.470 GB/s cold
transport, zero compute/common-weight cost, and authenticated executable sizes.

For each group report union records/bytes and optimistic TPS twice: at the
structural `A=horizon` ceiling and at the sum of observed source-window `A`.
Authorize a real capture only for the smallest horizon whose structural TPS is
at least 2 in every group/category and whose observed-sum diagnostic reaches 2
in at least half of categories. Otherwise reject wider capture through q64.

No stitched value is accepted-token TPS, a proposer result, a K4 construction
authorization, or proof of q16/q32/q64 route behavior.

## Result

q16 fails its structural gate in every group; even perfect acceptance produces
only `1.088`–`1.399` optimistic TPS. q32 also fails every group at
`1.461`–`1.952` structural TPS. Neither horizon has an observed-sum category
pass.

q64 is the first structural pass. Its four category unions contain
4,079–4,935 identities and move 82.910–100.977 GB after the free oracle cache.
Perfect `A=64` gives `2.200`–`2.679` optimistic TPS. Summed observed acceptance
is 53 ordinary, 48 code, 56 multilingual, and 56 rare-route tokens; the
corresponding diagnostics are `2.119`, `1.669`, `2.344`, and `1.925` TPS.
Ordinary and multilingual satisfy the observed-sum gate, meeting the declared
two-category threshold.

Canonical evidence:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0321/analysis-001/analysis.json`,
SHA-256
`3f69ad6b9da5f0403db5178f26c1c19c4d0828056df9f43abea31a69a1636358`.
Gate 8 retained 70% free memory, at most 138,264,576-byte peak RSS, zero swap
growth, zero new throttling, and stable protected services.

## Decision

Authorize one real width-64 teacher-forced corrected-route capture with no K4
construction. Preserve q16 and q32 as rejected for this byte premise. The real
capture must measure a single causal 64-position target transaction and may
promote only if its actual union/acceptance bound survives; stitched unions are
not an implementation answer key or performance result.
