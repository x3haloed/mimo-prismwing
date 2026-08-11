# PW-0180 — Fixed-index centroid recovery

- Status: complete
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0178 run-001
- Execution mode: private-codebook L3 training capacity preflight; source
  routes and outputs remain authoritative
- Related records: PW-0142, PW-0144 through PW-0146, PW-0178, PW-0179

## Contract

Keep every PW-0178 UINT8 assignment fixed and optimize only its input-vector
centroids through complete-expert output loss. This differs from failed scalar
straight-through QAT: parameters are continuous executable vector values and
there is no rounding boundary during optimization.

Use layer-46 expert 28, only schedule placements below 112, full-batch Adam for
64 steps, learning rate `5e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, and
`1e-5` mean-square anchoring to initial centroids. Execute the differentiable
training oracle in F32 on MPS. At the end, stage centroids once through F16 and
evaluate source BF16 semantics on validation positions `112..167`. Do not load
validation targets before training finishes; keep positions `168..223` sealed.

Before the real run, add a tiny fixture proving fixed-index centroid gradients
are finite and update the decoded weight. Record train loss every eight steps,
initial/final centroid hashes, displacement, artifacts, validation projection
and complete-expert errors, MPS allocation, and Gate 8.

Continuation requires train loss to fall at least 50%, complete validation L2
at most 2%, maximum-row error at most 5%, and gate/up validation L2 at most 2%.
A pass authorizes only a separately frozen shared-codebook multi-expert fit;
private codebook bytes are not a deployable ledger. Validation above 10% kills
fixed-index centroid-only recovery on this code topology. Report zero accepted
tokens and no TPS.

## Result

The valid report hashes to
`cbc780d45eda7be74c50955019b62514e0a5b885cb1c16d0796f85dbfa3a80c3`.
The schedule reduces its F32 train objective by `55.875%`, from `0.036973` to
`0.016314`, after an early excursion to `0.320832`. The continuous parameters
move and all safety gates pass, so this is not the inert-code failure seen in
PW-0145.

Reject generalization decisively after the frozen train phase. F16-staged
centroids produce `0.340665` complete-expert validation relative L2 and
`0.445062` maximum-row error. Gate/up validation errors are
`1.234581/0.742788`. This is memorization rather than a transferable centroid
rule and misses the 10% family-kill gate by a wide margin.

Kill fixed-index centroid-only recovery on PW-0178's topology. Do not expand
to shared multi-expert training or tune the schedule on visible validation.
The holdout remains sealed; zero tokens are accepted and no TPS or throughput
constant changes.
