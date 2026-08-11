# PW-0214 — Cost-adaptive verification-horizon oracle

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-11
- Execution mode: L1 analysis followed by exact verifier scheduling only if authorized
- Related records: PW-0206, PW-0208, PW-0211, PW-0212, AN-0002

## Hypothesis and mechanism

Verifier width `q` is a physical control variable. An extra speculative
position can add little expected accepted progress while introducing many new
layer/expert records and a wider target pass.

Hypothesis: a calibration-frozen policy can stop extension at the horizon whose
marginal expected accepted progress no longer pays for measured proposal,
shared-spine, routed-union, and acquisition cost. This may preserve a smaller
gain even though PW-0208 proved that no q=4/q=8 selector can reach its old 2x
expert-byte gate.

## Contract

Authenticate the corrected PW-0208 corpus and derive exact prefix acceptance,
per-layer expert union, `A`, `U`, logical bytes, and measured verifier/proposer
walls for every available `q=2..8`. Keep chronological category boundaries.
Separate calibration and holdout before selecting any threshold or policy.

Controls are fixed q4, fixed q8, best calibration-fixed q, previous-window q,
and an offline per-window future oracle. The future oracle is a ceiling, never
a runtime policy. A causal predictor may use only information available before
the next speculative position. Exact verifier-only commit and rollback remain
unchanged.

## Cheap falsifier and gates

Kill runtime work if the offline q oracle cannot improve modeled complete
transaction TPS at all after charging proposal and verifier walls. Preserve
smaller acceptance/union improvements as diagnostics. Authorize implementation
only if a calibration-frozen causal policy improves modeled TPS in every text
category and by at least 5% overall without increasing tail wall.

Runtime promotion requires identical verifier-authorized output, Gate 8, and a
repeatable interleaved complete-path TPS gain of any positive size. No DS4
confidence threshold or cross-model acceptance rate may be imported.

## Decision

Unexecuted. Scheduled after PW-0212 and the first PW-0213 transport falsifier.
It remains separate from native-MTP proposal latency and predictive prefetch so
optimistic grants cannot be multiplied.
