# PW-0058 — Real layer-1 attention-to-routing localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: diagnostic contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0056 cleared comparison
  `a741cc0a3686926ff2d4c880b08c3ab4ee046b4912f58a3b9738d2952ebbcb78`;
  frozen chat fixture
  `56dac58d602ab7bd567e9875282bf2f13ad2c4338e23f65b82affc8ec8bec9a1`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, PW-0050 safety contract
- Related records: PW-0039, PW-0049, PW-0056, PW-0057

## Hypothesis and mechanism

Dense layer 0 is cleared, so its bit-exact final state is an authoritative
real input to routed layer 1. The first remaining structural boundary combines
SWA QKV layout, theta 10,000 RoPE, eight KV heads, learned sink logits, causal
attention, post-attention normalization, F32 router projection/sigmoid,
noaux-tc corrected selection, and uncorrected normalized route weights.

## Contract

Extend the independent PyTorch and Rust trace paths to consume the exact
PW-0056 layer-0 final state and execute layer 1 causally. Capture and compare:
incoming hidden state, input norm, QKV, post-RoPE Q/K, scaled V, learned sinks,
centered attention scores, probabilities, attention output/projection, first
residual, post-attention norm, F32 router logits and scores, selected expert
IDs, and normalized route weights.

The Rust trace must execute the production endpoint functions and derive every
downstream value itself. The oracle may consume only the independently cleared
layer-0 final capture, never a Rust layer-1 intermediate. Tensor names, shard
identities, checkpoint verification, prompt IDs, dtype policy, dynamic FP8
scheme, capture schema, and source input hash are exact gates.

Use PW-0056 numerical limits for tensor captures. Selected expert sets must be
exact at every position; route-weight maximum absolute error must be at most
`5e-7`. Stop before expert tensor reads. If routing clears, open the bounded
selected-expert execution rung; if it fails, localize the first attention or
router substage before any full walk.

All phases retain PW-0050 host stops. The Rust walk checkpoints after causal
layer 0, layer-1 attention, layer-1 routing, and capture writes, releasing
mapped file pages and malloc transients before measuring each completed phase.
It fails closed below 20% system-free memory, above 8 GiB current or peak
process footprint, above 4 GiB post-phase footprint, above 512 MiB swap growth,
on any new throttled page, or if a protected service that was resident at start
disappears. The independent oracle applies the same limits after every capture.
Generated arrays remain external and content-addressed. This experiment cannot
change hosted thresholds or make a throughput claim.

## Result

Unexecuted.

## Decision

Unexecuted.
