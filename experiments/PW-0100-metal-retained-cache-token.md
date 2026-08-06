# PW-0100 — Bounded Metal retained-cache token

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0095 cached oracle manifest
  `75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8`;
  PW-0099 repaired routed-row output
  `77436d4ffc8a112d96f18275fbcc47097a67f2ca18a937c06726b736edc0d2a1`
- Hardware/runtime: Apple M1 shared 16 GiB host; verified SSD checkpoint;
  source Rust endpoint with a bounded source-FP8 Metal expert candidate
- Related records: PW-0092, PW-0095, PW-0096, PW-0097, PW-0098, PW-0099

## Shared construction contract

Capability: the existing target-faithful endpoint must prefill the frozen
27-token chat prompt, retain authoritative K/V in all 48 layers, accept token
264, then propagate only that one token through the same endpoint while using
PW-0099's bounded Metal executor for all 376 routed-expert executions. The
second step must causally produce its routes, layer states, final logits, token,
text, timing, byte ledger, and safety evidence from the real checkpoint.

Topology: tokenizer, checkpoint, attention, K/V caches, routing, residuals,
LM head, greedy selection, and accepted-token state remain in the existing
single Rust authority. Metal replaces only the routed expert projection
embodiment on a one-row incremental step. The CPU prefill and explicit Metal
incremental mode remain separately named; there is no hidden fallback,
alternate sampler, fixture-derived routing, or second output authority.

Embodiment depth: compile one Metal pipeline per process, map verified source
tensors, install at most one expert projection tensor at a time, perform exact
dynamic-FP8/BF16 staging and value-derived sparse midpoint repair, release its
buffers, and continue. Do not retain a model-wide expert bank, expand weights
to a persistent F32 representation, change weights or routing, or introduce a
modified mode. This is an L3 bounded arithmetic approximation until the full
project near-equivalence gates are passed.

## Hypothesis and gates

PW-0099 reduced a real routed row from PW-0096's 3,180 ms CPU attribution to
about 55.4 ms. Applying the same bounded mechanism to the 47 routed layers
should reduce the PW-0092/PW-0095 retained-cache token from about 158.5 seconds
to at most 20 seconds without crossing a source-derived correctness boundary.

Add deterministic tiny fixtures before integration for one-row expert schedule,
checkpoint tensor-view validation, weighted scatter order, sparse-repair
accounting, Metal/CPU mode separation, non-one-row rejection, and resource
release. The candidate must fail closed on revision, verification, fixture,
kernel, dtype, tensor name, shape, scale grid, route, non-finite, cache length,
oracle schema/hash, output, or commit mismatch.

The CPU prefill and accepted token 264 must remain semantically identical to
PW-0092. Compare the incremental step with every independent PW-0095 capture.
All 48 selected expert sets and their source order must match exactly; maximum
route-weight error must be at most `5e-4`. At every layer final, relative L2
must be at most `5e-4`, maximum absolute error at most `2e-2`, and BF16 identity
at least 99%. Apply the same three gates to final RMSNorm and all 152,576
logits. The greedy output must remain token 13. Report top-20 identity and
projected Jensen-Shannon divergence, but do not call a single position a
distributional acceptance result.

Time the complete second decode step from input token through available greedy
output, including attention, expert mapping/installation, dynamic staging,
sparse correction, Metal synchronization/readback, routing, residuals, LM head,
and sampling. Success requires at most 20,000 ms and at least 5x speedup versus
PW-0092's 158,521/158,615 ms controls. Record prefill separately; logical and
actual disk bytes; Metal-installed, sparse-decoded, and released bytes; batch
one; concurrency one; accepted tokens one in the timed interval; `A=1`; every
layer's `U`; cold/warm state; compiler; hardware; commit; and power as unknown
unless measured. This is walking-slice token latency, not the 8K/512-token
accepted-TPS gate.

Enforce normative Gate 8 at process start, checkpoint open, Metal compile,
every prefill and incremental layer, and each routed-layer buffer-release phase
after all eight sequential expert buffers have been released, plus LM head
completion, accepted-token commit, and final release. Assert and account for
each individual expert release in-process; perform the full host/service scan
at the layer phase so safety instrumentation does not dominate the token timer.
Stop if free memory
falls below 20%, process current or peak exceeds 8 GiB, post-release footprint
exceeds 4 GiB, swap grows more than 512 MiB, any new throttled page appears, or
ChatGPT, WindowServer, nxnode, or syncthing disappears. Record allocator relief
and prove both phase buffers and the resident service remain healthy.

Run one clean process first. Any correctness, causal, safety, or timing failure
kills the candidate and prevents an expensive repeat. Only a fully passing
first process authorizes a second clean process; the two candidate semantic
projections and accepted output must then be byte-identical. Passing authorizes
broader endpoint slices and repeated decode work, not a production default.

## Result

Unexecuted.

## Decision

Unexecuted.
