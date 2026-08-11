# PW-0208 — Native MTP cost-aware proposer

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-10
- Execution mode: L2 target-distribution-preserving verification goal; trained variants separately named L3/L4
- Related records: PW-0103, PW-0150, PW-0186, PW-0187, PW-0203, PW-0205, PW-0206

## Hypothesis and mechanism

Published MiMo-V2-Flash evidence shows that MiMo-family MTP can reach useful
accepted length, while FastMTP argues recursive-inference alignment matters and
EcoSpec shows that high-confidence branches can scatter across MoE experts.
Those are priors, not MiMo-V2.5 measurements.

Hypothesis: after corrected-QKV regeneration, the pinned native MTP can be
selected or shallowly scheduled by marginal expert cost so that accepted
tokens per unique expert byte improves at least 2x over corrected Jacobi,
without changing verifier-authorized output.

## Contract

PW-0206 must complete first. Measure proposal latency, target positions `q`,
accepted tokens `A`, per-layer expert union `U`, unique expert bytes, resident
hits, rollback, and complete verifier wall. The primary objective is accepted
tokens per target critical-path byte; acceptance length alone cannot promote a
candidate.

Untrained scheduling over the pinned MTP is L2 only when ordinary exact
verification/correction determines committed tokens. Any fine-tuning,
vocabulary compression, route-cost logit, or changed draft weights receives a
separate L3/L4 artifact, training manifest, and held-out evaluation.

## Cheap falsifier and gates

Run q=4 and q=8 on at least 32 corrected arbitrary-text windows spanning
ordinary, code, multilingual, and rare-route prompts. Kill native MTP if it
cannot beat corrected Jacobi on the complete quantity
`proposal wall + target miss wall per accepted token`, or if its accepted
tokens per unique expert byte improves by less than 2x.

Only a passing trace may enter one real wide-verifier transaction. Endpoint
promotion requires verifier-identical committed tokens and at least 1.5x
complete-path TPS over the same residency state.

## Decision

Unexecuted. Published acceptance or GPU speedup is not substituted for the
pinned model's measured `A/U/bytes`.
