# PW-0321 — Stitched corrected wide-horizon diagnostic

- Status: planned
- Disposition: pending
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
