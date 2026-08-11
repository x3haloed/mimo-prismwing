# PW-0212 — Corrected-route predictive-prefetch oracle

- Status: complete
- Disposition: rejected for runtime implementation under the frozen tax and
  complete-wall gate; logical predictability retained as diagnostic evidence
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

Model cacheable/mapped transport separately from uncached owned transport.
DS4 mainline's bounded `F_RDADVISE` is a design prior for issuing known ranges;
the `F_NOCACHE`/disabled-`F_RDAHEAD` prototype in issue 437 is not a free grant.
PW-0212 may authorize which ranges and lead times matter, but PW-0213 must
measure whether the alternate transport changes physical latency or page-cache
pressure.

Only a calibration-frozen predictor that passes every text category may enter
a cold runtime pilot. Runtime promotion requires bounded queues, demand
priority, exact output parity, Gate 8, attributed physical I/O, and a repeated
interleaved complete-path TPS gain. Multimodal and long-context corrected
traces remain required before any general default.

## Execution and evidence

The analyzer authenticated the PW-0208 complete-history manifest
`a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b`
and every referenced report, progress payload, and hidden-state payload. The
first four chronological windows in each of ordinary, code, multilingual, and
rare-route text calibrated frequency controls; the final four per category
formed a 16-window holdout. All 6,016 holdout events discriminate among the
controls. Source record size is 25,171,968 bytes per layer/expert and the
measured cold eight-expert acquisition is 58.033833 ms.

Logical holdout results preserve real but differently actionable signal:

- last route recalls 38.370595% of selections, but is known only after the
  same layer's route and therefore has no acquisition lead in the current
  batched verifier;
- calibration-frozen category frequency recalls and precisely predicts
  32.415642%, with 4,818 useful and 1,198 wasted unique records;
- same-layer previous position recalls 26.504322% at 30.290653% precision, but
  likewise has no physical lead in the current batched route;
- previous transaction at the same position recalls and precisely predicts
  23.859292%; and
- the causal previous-layer/same-position control recalls only 2.944232% at
  3.008237% precision.

The offline future oracle reaches 100% logical recall and precision. Even
granting that impossible oracle exact future knowledge and optimistic full
overlap, the 25% demand-traffic budget permits only 6,386 of 26,710 unique
demand records (23.908648% tax). It can hide at most 46,325.507 ms across the
holdout: 7.988956% of verifier wall but only 1.616833% of complete transaction
wall. No category exceeds 1.727160%, and the best individual window reaches
1.783493%, versus the frozen 10% complete-wall requirement.

Clean report:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0212/corrected-route-prefetch-oracle-001.json`
(SHA-256
`2365033116e194b6bac34d2017f644c3499c5fb92a3727f7db9162dce318587f`),
generated from clean commit `098d43224a2cbbce706bca82b34bb2bc75a3033f`.

## Decision

Reject a runtime predictive-prefetch implementation under the frozen 25%
traffic-tax and 10% complete-wall gate. This rejection is stronger than any
measured causal predictor because the physically bounded future oracle also
fails. Preserve the logical signals for residency ordering, future changed
critical cuts, and PW-0214 horizon work; do not present them as hidden latency.

PW-0213 remains open and independent: it changes the acquisition transport and
page-cache topology for bytes already demanded rather than relying on route
prediction. A future materially different endpoint may reopen prefetch only as
a separately gated experiment. Missing 50 TPS did not kill this branch; its
own predeclared complete-path gate did.
