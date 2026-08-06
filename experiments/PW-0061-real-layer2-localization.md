# PW-0061 — Real layer-2 substage localization

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean `6b31f86f45b7a930d62df45484999e1b2e872b3c`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0060 comparison
  `2b74e0c622f1e44947307820067d8633cd972096af558376b9dd4d71edf1bbdf`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, PW-0050 safety contract
- Related records: PW-0058 through PW-0060

## Hypothesis and mechanism

PW-0060 proves layer 2 is the first non-exact causal boundary, but layer-final
captures cannot distinguish attention accumulation from router sigmoid,
normalized weights, expert execution, scatter, or residual effects. Layer 1's
final state is bit-exact and supplies a clean source boundary.

## Contract

Run one Rust trace that causally recomputes layers 0–1 with production
functions, then executes production layer 2. Run one independent PyTorch trace
from PW-0060 oracle run 002's hash-verified layer-1 final state. Capture input
norm, QKV, Q/K/V, learned sinks, centered attention scores, probabilities,
attention output/projection, first residual, post-normalization, F32 router
logits/scores, exact selected expert sets and normalized weights, expert-major
gate/up/SwiGLU/down outputs, weighted scatter, and final residual.

The Python authority derives its own routes and resolves weight/scale tensors
independently across shards. The Rust authority cannot consume oracle routes
or intermediates. Source state, revision, prompt, checkpoint verification,
tensor index, dtype, shape, schedule, dynamic FP8 scheme, and BF16 staging are
exact gates.

Use PW-0058/PW-0059 limits: BF16 relative L2 at most `5e-4`, maximum absolute
error at most `2e-2`, equality at least 99%; exact expert sets; route-weight
maximum absolute error `5e-7`; and final-state relative L2 `4e-5` with maximum
absolute error `3e-6`. Identify the first failing substage and change no
arithmetic until it is explained by source/runtime evidence.

Retain phase stops after causal layer 1, layer-2 attention, routing, every
completed expert, residual, and capture writes. Release file pages and
allocator transients; fail closed below 20% system-free memory, above 8 GiB
current/peak or 4 GiB post-phase footprint, above 512 MiB swap growth, on new
throttled pages, or on protected-service disappearance. Generated arrays stay
external and content-addressed. This cannot change hosted thresholds or make a
throughput claim.

## Result

Comparison 001 is exact through centered attention scores. The first causal
difference is one BF16 probability value among 25,920, at position 17, head
29, in a 19-value sink-inclusive row. Probability equality is 99.9961%, with
maximum error `0.000244140625` and relative L2 `1.72e-5`. That perturbation
amplifies through attention/projection and reaches router logits at relative
L2 `5.60e-5`, then route weights at `1.7813905143770903e-6`; expert sets remain
exact. Final state matches PW-0060's localized error.

Replaying the exact score corpus separates mechanisms. PyTorch exponentials
with forward sequential F32 denominator accumulation reproduce the one
mismatch; PyTorch's own sum and reverse sequential F32 accumulation reproduce
the oracle exactly. Reverse accumulation also preserves zero mismatches across
all PW-0056 layer-0 and PW-0058 layer-1 probability corpora. Thus exponential
evaluation is cleared; denominator reduction order is the first discrepancy.

Evidence hashes: oracle manifest
`4f92e14a9d9a9c4a3ea137bb9f5e671814f6d047798bbbf1af40a9e95c93f91a`;
Rust manifest
`d0a75b07ecc95f0cf5579885555f88860955b9097d8b5c261d2a7af2b0674190`;
comparison
`848965d901ba9d540204446c7658985110628df5eee88469015732463018e103`.
Rust completed in 27.741 seconds and the oracle in 13.392 seconds. Rust peaked
at 727,023,616 bytes, retained at least 77% system-free memory, grew no swap,
and observed no throttling. No throughput constants change.

## Decision

Promote the layer-2 trace as a correctness diagnostic. Open a separately gated
softmax denominator-order repair using the exact failing 19-value row and all
three real layer corpora. Do not change sigmoid, routing, experts, or scatter:
their divergence is downstream of the first probability mismatch.
