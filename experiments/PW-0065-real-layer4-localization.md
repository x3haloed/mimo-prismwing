# PW-0065 — Real layer-4 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0064 comparison
  `cd057b3eb6ecb7c7075599d432595b3f3dbdd6d246c3816822437bede55d13b0`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060 through PW-0064

## Hypothesis and contract

PW-0064 proves layer 3 is an exact accumulated source boundary and layer 4 is
the first failure. A substage trace can separate attention, routing, expert,
scatter, and residual causes without another full-model walk.

The Rust authority must causally recompute layers 0–3 with production
functions, then execute and capture production layer 4. The independent
PyTorch authority must load the hash-verified layer-3 final state from frozen
PW-0060 oracle run 002 and derive its own layer-4 tensors and routes. Neither
authority may consume the other's intermediates, routes, or selected experts.

Capture incoming state, input norm, QKV, Q/K/V, learned sinks, centered
attention scores, probabilities, attention output/projection, first residual,
post-normalization, F32 router logits/scores, exact selected expert order and
weights, expert-major gate/up/SwiGLU/down outputs, weighted scatter, and final
residual. Bind source-state hash, revision, prompt, checkpoint verification,
tensor layout, dynamic FP8 scheme, BF16 staging, and output schema.

Use the existing gates unchanged: BF16 relative L2 at most `5e-4`, maximum
absolute error at most `2e-2`, equality at least 99%; exact expert sets;
route-weight maximum error `5e-7`; final-state relative L2 `4e-5` and maximum
absolute error `3e-6`. Identify the first failing substage and make no
arithmetic change until source/runtime evidence explains it.

Retain the shared-host stops after each causal layer, target attention, every
completed expert, final residual, and capture writes. Fail closed below 20%
system-free memory, above 8 GiB current/peak or 4 GiB post-phase footprint,
above 512 MiB swap growth, on new throttled pages, or protected-service loss.
Release mapped pages and allocator transients at matrix/expert boundaries.
Generated tensors remain external and content-addressed. This diagnostic has
accepted tokens zero and cannot change hosted thresholds or throughput
constants.

## Result

Unexecuted.

## Decision

Unexecuted.
