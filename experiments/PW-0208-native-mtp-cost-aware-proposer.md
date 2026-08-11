# PW-0208 — Native MTP cost-aware proposer

- Status: complete
- Disposition: rejected — 2x expert-byte gate is impossible; latency-only value preserved in PW-0211
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

Rejected for its predeclared cost-efficiency promotion gate. The corrected,
balanced corpus contains the first eight MTP-evaluable width-eight verifier
windows after prefill for each of ordinary, code, multilingual, and empirically
rare-route prompts. Its complete-history manifest is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0208/corpus-001/manifest.json`,
SHA-256 `a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b`.
It binds 32 windows to exact prompt and retained target layer-47 histories,
tokens, verifier routes, and hidden-file offsets. The corrected Jacobi control
commits `A=213` across 53,251 exact layer/expert units, or
1,340,432,467,968 unique expert bytes. The rare-route slice contributes 942
layer/expert pairs absent from the other 24 primary windows and spans all 47
routed layers.

The clean perfect-proposal upper bound is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0208/native-mtp-cost-upper-bound-001.json`,
SHA-256 `3aaca59be0e000cac77d5a36b8e3b9d2e2fc5bbb02792c8846dae3da16747f8c`.
It authenticates every selected expert tensor against the pinned checkpoint and
derives 25,171,968 bytes per complete layer/expert bundle. Under the endpoint's
existing commit semantics, even an omniscient native proposer can expose at
most `A=3` at q=4 or `A=7` at q=8. Perfect q=4 reaches only 0.721493x the
control accepted-token/unique-byte rate. Perfect q=8 reaches 1.051643x, and an
omniscient per-window q=4/q=8 selector chooses q=8 for all 32 windows and cannot
do better. This is far below the required 2x, so real draft execution cannot
rescue the gate.

Two earlier corpus manifests are preserved as superseded evidence rather than
erased. SHA `c31a3967d41dd04b73f36e96ac328050c7f64cec7e2b024ce231b67fd9dfd39a`
used the same transaction's hidden block and missed the autoregressive shift.
SHA `2381319c27942dfdd319e2ac700fa3615fb3ceed5cf11af9f8aabf94479b13fb`
corrected the preceding-row alignment but omitted the complete MTP attention
history. No MTP score was computed from either.

This result kills only the 2x expert-byte hypothesis. Native MTP may still
replace roughly 140 seconds of same-model Jacobi proposal work and yield a
smaller but valuable TPS gain. Preserve that pathway under PW-0211 rather than
discarding it because PW-0208 or the 50-TPS target is missed.
