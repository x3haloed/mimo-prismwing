# PW-0058 — Real layer-1 attention-to-routing localization

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean `9d2f70cd34415612055d8e9c65073e750c3cdb5a`
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

Rust run 001 causally executed the production dense layer 0 and routed layer-1
attention/routing path, stopping with zero routed expert executions. The
independent oracle consumed only PW-0056 oracle run 004's hash-verified final
state. Comparison 001 clears every contracted boundary:

- incoming state through post-attention normalization is BF16 bit-exact;
- router logits are F32 bit-exact;
- router scores have relative L2 `1.4122187251662206e-8` and maximum absolute
  error `1.1920928955078125e-7`;
- all eight selected-expert sets match at all 27 positions; and
- route weights compared by expert identity have maximum absolute error
  `2.5432357775301284e-8`, below the `5e-7` limit.

Evidence hashes:

- independent oracle manifest:
  `a348b9f521fc6b46fd13ad95f3165c8f1cbfff5f53c3d342915dfa275206668e`;
- Rust manifest:
  `b946ed6a268be454247a7aa04888f2a00e52f36c9dab7b898e53287326c12f39`;
- comparison 001:
  `1cea804681d175b0fe4c359aafe120a6659f3dc45d5c125a3f5aad5ca36880d2`.

Rust completed in 4.245 seconds with 456,463,488 logical source bytes,
458,186,752 measured process disk-read bytes, zero expert executions, and
686,292,992 bytes peak resident memory. The oracle completed in 1.918 seconds
and peaked at 938,049,536 bytes. Both retained 76% system-free memory, grew no
swap, observed no throttled pages, and preserved every protected service.
All 29 Rust tests, 39 Python tests, strict Clippy, Python compilation, and the
release build pass. These are diagnostic timings, not throughput claims, so no
throughput-model constant changes.

## Decision

Promote the layer-1 attention/routing trace as a correctness diagnostic and
provisionally clear the first routed layer through exact expert selection.
The belief that SWA attention, learned sinks, or noaux-tc routing is the first
structural cause of hosted divergence is superseded. Open the bounded selected
expert execution rung for these exact routes; do not repeat another whole-model
walk or infer hosted parity until that expert path clears.
