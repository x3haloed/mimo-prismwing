# PW-0214 — Cost-adaptive verification-horizon oracle

- Status: complete
- Disposition: runtime policy rejected; offline ceiling and code-slice gains
  preserved as diagnostics
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

The analyzer authenticates the corrected PW-0208 manifest and all four source
reports, progress streams, and hidden-state payloads. Transactions 1--4 in
each category calibrate; transactions 5--8 hold out. For every q=2..8, exact
proposal/posterior prefixes reconstruct `A`, and exact route prefixes derive
per-layer expert union. The physical model charges measured q8 same-model
proposal wall by `(q-1)/7`, exact expert units at 7.254229125 ms each, and the
remaining measured q8 verifier wall as shared-spine/compute cost scaled by
`q/8`.

The globally calibration-selected fixed q5 regresses holdout modeled TPS by
1.717985% versus q8. A category-calibrated policy selects q5 for ordinary and
code and q8 for multilingual and rare-route. It improves aggregate holdout TPS
by 0.933803% and code by 3.268060%, but ordinary regresses 2.869413% and the
other two categories are unchanged. The previous-window control regresses
0.501762%. Preserve the aggregate and code-slice gains; neither is a policy
that passes every required category.

A Dinkelbach-optimal offline future oracle maximizes the ratio of total
accepted tokens to total modeled wall over every combination of q choices. It
chooses q2 three times, q3 once, and q8 twelve times, yet reaches only a
4.681640% holdout gain. This omniscient ceiling is below the frozen 5% causal
implementation gate. The locally best per-window future oracle reaches
4.422963%. No learned causal predictor can exceed the aggregate future oracle
under the frozen model.

Clean report:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0214/cost-adaptive-horizon-oracle-001.json`,
SHA-256
`0a7204c271c60cf7f362cdbf5512cc207b4d7ddc97716d0e4e9c0aeea852cc8e`,
generated from clean commit `4b47d6a36d9d3670317276541e185d704103f9a3`.

Reject runtime policy implementation under this contract. The cheap
falsifier's weaker "any oracle gain" test passes, so preserve the 4.681640%
ceiling, 0.933803% category-policy aggregate, and 3.268060% code slice as real
modeled signals. The stronger predeclared implementation gate fails even for
the future oracle. Reopening requires changed q-specific physical walls or a
different proposer/critical cut, not fitting a more complex predictor to the
same ceiling. This record makes no endpoint TPS claim and does not alter
PW-0211's measured native-q4 ordinary milestone.
