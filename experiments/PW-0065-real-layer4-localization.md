# PW-0065 — Real layer-4 substage localization

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean implementation
  `b86340d3e5959a83c84751a6795c42411cb0c45b`
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

The routed-layer trace was generalized without duplicating its authority. The
default layer-2 comparison remains byte-identical to PW-0061 comparison 004,
and a clean production regression is exact for every layer-2 capture. A first
regression run is preserved but excluded because its supplied commit string
was mistyped; run 002 binds the exact implementation commit.

For layer 4, incoming state, input norm, QKV, Q/K/V, learned sinks, and every
centered attention score are bit-exact. The first numerical difference is two
of 25,920 BF16 attention probabilities: position 0/head 26 in a two-value row,
and position 23/head 40 in a 25-value row. Probability equality is 99.9923%,
maximum error `0.001953125`, and relative L2 `1.0445e-4`. Amplification first
breaches a general BF16 gate at post-attention RMSNorm (`moe_input`), whose
maximum error is `0.046875`. Expert sets and output order remain exact; route
weights later differ by at most `5.9301374015809094e-5`. The final metrics
exactly reproduce PW-0064's layer-4 boundary.

Source replay explains the two probability values. PyTorch's CPU F32 softmax
uses SLEEF vector exponentials, four-lane ARM accumulation, horizontal
reduction, reciprocal, and multiplication. Rust still uses vForce
exponentials, reverse scalar accumulation, and division. The old path happened
to round exactly on the complete layer-0 through layer-3 corpora, but not these
two rows. Replaying the PyTorch operation order produces zero mismatches on all
four complete real probability corpora, including layer 4.

The oracle completed in 15.482 seconds, peaked at 968,441,856 bytes, retained
at least 79% system-free memory, and ended at 513,277,952 bytes. Rust completed
in 58.645 seconds, peaked at 717,946,880 bytes, retained 81% free memory, and
ended at 121,097,856 bytes. Both grew no swap, observed no throttling, and
retained every protected service.

Oracle manifest hash:
`ebe394a2caae8ec72ef190ae0ba743c8c6d14b7d1fd57380f36f493470dac076`.
Rust manifest hash:
`b29b07314232c40412597eb2add3dc414a77d83fb177e78029ddfcca3dee532c`.
Comparison hash:
`4ee90dfffe0523382e5155bfe2eb27be359a8a544c3345a0ec4e46f5dc8ab54a`.

## Decision

Promote the generalized routed-layer trace as a correctness diagnostic and
localize layer 4's first difference to CPU softmax operation order. Open a
separate fixture-gated PyTorch softmax repair using both failing rows and all
four real corpora. Do not change routing, experts, hosted gates, or throughput
constants.
