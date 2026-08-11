# PW-0178 — Input-subvector code capacity oracle

- Status: complete
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; authenticated PW-0116
  routed-activation corpus; layer-46 hot expert 28
- Execution mode: favorable shadow L3 representation oracle beside the
  target-faithful source expert; candidate state never enters the model
- Related records: PW-0116, PW-0129, PW-0147 through PW-0149, PW-0177; E5

## Realization and compression contract

Capability: determine whether the physically eligible two-index-bit-per-weight
input-subvector representation has enough private code capacity to preserve one
real deep MiMo expert before paying for a shared-layer fit or Metal kernel.

Envelope: one layer-46 hot expert, four adjacent input weights per vector, 256
FP16 centroids per input group, one UINT8 code per vector, batch one source BF16
semantics, train placements below 112, validation placements `112..167`, and a
sealed pilot holdout. K-means is private per expert and therefore a favorable
capacity oracle for the intended layer-shared codebooks.

Physical hypothesis: layer-shared input-group codebooks let gate/up construct
centroid dot-product tables once from their common routed activation and reuse
them across eight experts. Expert-specific work becomes compact index lookup
and accumulation. Down tables remain expert-specific because SwiGLU activations
differ. The eventual representation is 6,291,456 code bytes/expert plus
5,242,880 resident FP16 codebook bytes/layer, before container metadata.

Exclusions: no new or sidecar hardware; no Core ML package switching; no dense
candidate artifact at runtime; no shared fit, kernel, layer, token, endpoint,
holdout, or TPS claim. Dense reconstruction is only the numerical oracle.

## Contract

1. Authenticate checkpoint, revision, PW-0116 corpus, captures, shapes, routes,
   and expert schedule. Fail closed on drift.
2. Add a deterministic tiny fixture for UINT8 input-subvector decode and the
   exact index/codebook byte ledger before the real run.
3. For each input group, transform source weight vectors by that group's train
   activation covariance, fit 256 centroids with frozen deterministic
   MiniBatchKMeans settings, invert the transform, and assign UINT8 indices.
   Gate and up use routed MoE inputs; down uses source train SwiGLU activations.
4. Preserve raw indices and FP16 codebooks outside Git with content hashes.
   Record reconstruction error and the exact physical ledger. Do not charge the
   dense oracle reconstruction as a deployable artifact.
5. Evaluate source and candidate gate/up projections and the complete expert on
   all 56 validation placements. Require source complete-expert relative L2 at
   most 0.1% and maximum-row relative L2 at most 0.2%, or invalidate the run.
6. Candidate promotion requires complete-expert validation relative L2 at most
   2% and maximum-row relative L2 at most 5%. Gate and up projection relative
   L2 must each be at most 2%; report every projection without averaging away a
   failure.
7. Physical eligibility requires exactly 6,291,456 expert index bytes, no more
   than 5,242,880 shared FP16 codebook bytes/layer, and projected code traffic
   for 376 expert executions no greater than 2,365,587,456 bytes/token before
   cache. Record lookup/add and centroid-table MAC counts for top eight over all
   47 routed layers.
8. If the favorable private oracle fails complete validation above 10%
   relative L2, kill the single-codebook two-bit input-subvector family. A
   smaller miss permits activation-aware assignment or codebook training; a
   pass promotes only a multi-expert shared-codebook fit with matched private
   controls.
9. No result promotes performance until a direct packed Metal transaction
   measures the complete routed layer and then an accepted incremental token.
   Report zero accepted tokens here.

## Result

The valid contract-bound report hashes to
`1311a8ced8ea4d376229efc9e1508542e5023d41b8f9cec546fcaab3548ac559`.
The source expert reproduces all 56 validation captures bit-exactly. The
physical ledger also passes exactly: 6,291,456 expert code bytes,
5,242,880 shared codebook bytes/layer, 2,365,587,456 code bytes for 376 expert
executions, and 246,415,360 resident codebook bytes across 47 layers.

Reject the numerical representation. Gate and up validation relative L2 are
`0.119019` and `0.084431`; complete-expert validation relative L2 is
`0.207785`, with `0.240740` maximum-row error. Even training error is
`0.191608`. Weight relative L2 is `0.337165/0.342319/0.520490` for gate, up,
and down. The favorable private codebooks miss the 10% family-kill boundary,
so a layer-shared codebook at the same rate cannot rescue capacity.

Kill the single-codebook two-index-bit input-subvector family. Do not build its
shared fit or kernel. Preserve the exact physical ledger for distinct residual,
multi-codebook, or trained representations, but require those branches to pay
and verify every added byte and operation. The pilot holdout remains sealed;
the run records zero accepted tokens and no endpoint TPS.
