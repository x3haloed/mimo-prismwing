# PW-0175 — structured sparse-prefill continuation audit

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0158 exact one-million-token
  attention ledger; TARGET L3 behavioral gates
- Execution mode: primary-source L3 mechanism and released-configuration
  audit; no model execution or endpoint claim
- Related records: PW-0158, PW-0160, PW-0162; E7

## Question and changed premise

PW-0158 rejects ordinary dense attention on the retained two-P100 branch: even
combined advertised FP16 peak needs 4,933.814 seconds for mandatory attention
alone at one million tokens. PW-0162 tests a different approximation, retaining
the globally largest exact attention probabilities at sampled source positions.
That oracle does not test whether a cheap structured selector can find useful
vertical, diagonal, local, or query-dependent regions without first computing
the full matrix.

MInference claims a training-free, model-calibrated combination of A-shape,
vertical-slash, and block-sparse prefill masks. Quest instead selects KV pages
for autoregressive decode. Audit their immutable papers and released code before
either porting a CUDA mechanism or dismissing structured sparsity from the
probability-ranked PW-0162 result.

## Exactness and scope

Any accepted structured sparse attention output is L3. Training-free does not
mean function-preserving, and attention-mass recall is not a Prismwing fidelity
gate. Published latency, benchmark, and cross-model results are priors only.

This experiment may promote a MiMo-specific structured-mask oracle. It cannot
promote a runtime default, change TARGET thresholds, authorize hardware, or
report Prismwing TPS.

## Contract

1. Authenticate TARGET, PW-0158's authoritative manifest, immutable PDF
   captures of MInference 1.0 and Quest, and immutable captures of the exact
   released source revisions used for the audit. Bind every input by SHA-256.
2. Extract directly from the primary sources and released code:
   - whether the mechanism changes prefill, decode, or both;
   - whether training or model-specific offline calibration is required;
   - supported attention topology, context lengths, model families, hardware,
     and released kernels;
   - the exact selector and retained-work parameters of one released 1M
     MInference configuration;
   - fidelity workloads and missing Prismwing slices.
3. Recompute, from the authenticated released configuration rather than paper
   speedup, a favorable nominal upper bound on selected causal QK pairs at
   `N=1,000,000`. Count `min(position, vertical+slash)` for every position and
   head. Report dynamic-index construction separately and do not treat overlap
   or GPU efficiency as proven savings.
4. Compare that favorable retained-pair fraction with PW-0158's
   `21.056139%` global-attention continuation ceiling. Passing this structural
   comparison only establishes that a MiMo-specific oracle is worth testing.
5. Reject any released method as the PW-0158 prefill repair when its own code
   explicitly leaves prefill dense. Preserve useful decode-only mechanisms as
   out of this experiment's causal scope.
6. Promote a follow-up only when the method is post-hoc, acts on prefill, and
   its released configuration's favorable pair fraction fits below the
   continuation ceiling. The follow-up must derive MiMo's per-layer/per-head
   patterns from source Q/K, reproduce the released online selector rather than
   use an exact top-probability oracle, and test output, route, logit, long-text,
   and native-modality fidelity before any runtime work.
7. Fail closed on a malformed configuration, unsupported pattern, source hash
   mismatch, ambiguous parameter meaning, or arithmetic mismatch. Do not copy a
   GLM/Llama/Qwen head map into MiMo.
8. Apply Gate 8 to the analyzer/source-capture phase. Record zero accepted
   Prismwing tokens, no endpoint TPS, and no purchase authority.

## Promotion and kill rule

Promote one MiMo-specific structured-mask oracle only if an authenticated,
training-free released prefill method has a favorable nominal causal-pair
fraction below `21.056139%` at 1M and its selector can be implemented from
source Q/K without changing weights or topology. This is a research promotion,
not fidelity or performance promotion.

Otherwise reject the audited released mechanisms as the PW-0158 continuation
and preserve only distinctly trained L4 attention replacements as unproven.

## Result

Unexecuted. No conclusion may be drawn from this record yet.
