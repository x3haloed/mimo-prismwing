# PW-0175 — structured sparse-prefill continuation audit

- Status: completed
- Disposition: conditional — promote a MiMo-specific MInference-style oracle;
  reject released Quest as the PW-0158 prefill repair
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0158 exact one-million-token
  attention ledger; TARGET L3 behavioral gates
- Execution mode: primary-source L3 mechanism and released-configuration
  audit; no model execution or endpoint claim
- Related records: PW-0158, PW-0160 through PW-0162; E7
- Implementation commit and dirty state:
  `ef6be18ff16c8f9c8fe0e1839a6e83a1e148ab46`, clean

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

1. Authenticate TARGET, PW-0158's authoritative attention manifest,
   PW-0161's complete two-P100/EPYC arithmetic manifest, immutable PDF captures
   of MInference 1.0 and Quest, and immutable captures of the exact released
   source revisions used for the audit. Bind every input by SHA-256.
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
4. Reproduce the `21.056139%` global-attention continuation ceiling from
   PW-0158's attention work and PW-0161's complete arithmetic allowance, then
   compare the favorable retained-pair fraction with it. Passing this
   structural comparison only establishes that a MiMo-specific oracle is worth
   testing.
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

The clean authoritative execution produced a manifest with SHA-256
`e5ac56b7f710285cdeb0088f9fa750748ad74cbc68cd6d4dcb627061209a37ab`.
It authenticates TARGET, PW-0158, PW-0161, both primary papers, both complete
released-source archives, the exact GLM-4-9B-1M head map, and the critical
released selector/orchestration files.

PW-0158 and PW-0161 independently reproduce the full-system continuation
ceiling. Two P100 advertised FP16 peaks plus the EPYC's impossible peak grant
`68,656,320,000,000,000` FLOPs in 1,800 seconds. After mandatory matrices and
sliding attention, `38,810,714,295,992,320` FLOPs remain for global attention,
exactly `21.056139043683178%` of the ordinary global work.

The released GLM-4-9B-1M MInference configuration contains 1,280
layer/head records, all using vertical-slash masks. Without deduplicating
vertical/slash overlap, `min(position, vertical+slash)` retains at most
`7,873,793,997,845` of `640,000,640,000,000` causal pairs, or
`1.2302790818841993%`. Charging its last-64-query online index pass at the QK
dot cost raises the favorable effective fraction to `1.2379590742042071%`.
This has substantial structural margin below `21.056139%`, even though perfect
kernel efficiency remains an explicit grant.

Promote a separate MiMo-specific source-state oracle that derives every
layer/head pattern from MiMo Q/K and then executes the released online
selector. Do not reuse GLM's map. The follow-up must test source output, route,
logit, long-text, and native-modality fidelity before kernel work. MInference
publishes no MiMo configuration, Metal/P100 kernel, hosted top-20 gate, or
native-modality evidence, so no fidelity, runtime, or performance default is
promoted.

Quest is rejected only as PW-0158's prefill repair: its released code states
that prefill optimization is unsupported and uses dense FlashInfer prefill.
Its query-aware sparse decode mechanism remains outside this experiment and is
not rejected for decode.

Gate 8 passes with 72% minimum free memory, 50,085,888-byte peak RSS,
18,613,632-byte maximum physical footprint, zero swap growth or throttling, an
explicit release boundary, and stable protected services. PW-0175 records zero
accepted Prismwing tokens, no endpoint TPS, no measured throughput-model
constant changes, and no purchase authority.
