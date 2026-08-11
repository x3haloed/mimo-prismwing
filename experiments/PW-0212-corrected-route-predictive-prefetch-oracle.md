# PW-0212 — Corrected-route predictive-prefetch oracle

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-11
- Execution mode: L1 analysis of corrected target routes; no runtime prefetch
- Related records: PW-0104, PW-0112, PW-0205, PW-0207, PW-0208, AN-0001

## Hypothesis and mechanism

Reactive global LRU was zero on the old-layout PW-0112 trace, but this does not
bound predictions issued before demand. PW-0208 now supplies corrected target
routes across ordinary, code, multilingual, and rare-route transactions.

Hypothesis: current hidden/router state or recent corrected routes predict a
bounded portion of the next demand-critical layer/expert set early enough to
overlap its checkpoint acquisition, with enough precision that wasted reads do
not starve demand I/O. This attacks exposed acquisition latency; cache hits or
recall alone are not the objective.

## Contract

Authenticate the PW-0208 complete-history manifest and every underlying report
and route payload. Preserve chronological category and transaction boundaries.
Separate calibration from holdout before choosing a predictor. Report
recall@8, precision, bytes prefetched, useful bytes, late bytes, duplicated
bytes, lead distance, and category/tail slices. A layer/expert identity is
local to its layer. Do not use future routes, fixture-specific token tables, or
unbounded speculative I/O in any causal result.

First establish a no-learning family of discriminating controls: last-route,
same-layer previous-position, previous-layer same-position, frequency-only,
and an offline future oracle. The oracle is an upper bound, not a policy. The
fixture must contain transitions on which the controls disagree; aggregate
recall without such transitions is non-discriminating.

## Cheap falsifier and gates

Convert useful and wasted predictions to a conservative acquisition model
using measured checkpoint record bytes and bandwidth. Charge all speculative
bytes at the device bottleneck and grant overlap only inside measured causal
lead time. Kill implementation if even the offline bounded oracle cannot hide
at least 10% of complete transaction wall without exceeding a fixed 25% demand
bandwidth tax. Preserve smaller logical predictability results as diagnostics.

Only a calibration-frozen predictor that passes every text category may enter
a cold runtime pilot. Runtime promotion requires bounded queues, demand
priority, exact output parity, Gate 8, attributed physical I/O, and a repeated
interleaved complete-path TPS gain. Multimodal and long-context corrected
traces remain required before any general default.

## Decision

Unexecuted. Scheduled after PW-0211's correctness/latency falsifier and before
PW-0210 implementation. Missing 50 TPS alone does not reject a repeatable
positive complete-path prefetch gain.
