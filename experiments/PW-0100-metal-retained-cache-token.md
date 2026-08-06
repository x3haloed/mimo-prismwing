# PW-0100 — Bounded Metal retained-cache token

- Status: complete
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation
  `36bb2928530281e69d46f93382728610542dd45a`; clean tree
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

The implementation introduced an explicitly named candidate endpoint rather
than changing the CPU default. It compiled one process-owned Metal pipeline,
validated real checkpoint tensor views, rejected non-one-row use, preserved
the source router and BTree expert scatter authority, and accounted for every
projection installation, release, dispatch, and sparse repair. Forty-seven
Rust tests, 50 Python tests, formatting, and strict Clippy passed before the
walk.

The first invocation used the stale pre-PW-0100 release executable and stopped
at argument parsing before opening the model. It is preserved as
`run-001/error.txt`, SHA-256
`5308e89d653b5d4a1c334c32f6b1a30a73b8083e1a1049e350cafd6b5ca94d9d`.
After rebuilding the standalone release binary from the pinned clean commit,
`run-001a` completed the full CPU prefill and bounded Metal incremental step.

The candidate retained the cache and still chose greedy token 13. Its first
failed layer was layer 4; selected experts remained exact and maximum route-
weight error was zero, but the layer-final state reached `0.00163510` relative
L2, `1.0` maximum absolute error, and 97.8760% BF16 identity. These fail the
unchanged `5e-4`, `2e-2`, and 99% gates respectively. Final RMSNorm and logits
also failed their unchanged gates. Because the candidate failed closed, it did
not emit an accepted report or claim that token identity proves distributional
parity.

The complete retained-cache token took 75,725.919 ms, a diagnostic 0.01321
token/s and only 2.0934x/2.0946x faster than PW-0092's 158,521/158,615 ms CPU
controls. It therefore also fails the 20,000 ms and 5x gates independently of
correctness. The failure record hashes to
`7e76c0bcabb445ded01f547ce56f096f2a6c9474a1fccee74761293cfd29df74`.
No second expensive run was authorized.

Gate 8 passed through the failed-candidate release boundary: minimum system
free memory was 79%, peak RSS was 4,311,433,216 bytes, post-release physical
footprint was 3,062,211,200 bytes, swap growth was zero, new throttled pages
were zero, and all protected services remained resident. The run demonstrates
real phase-scoped release behavior, but safety does not rescue either failed
promotion gate. Power was not measured and remains unknown.
The updated throughput model hashes to
`0698c43507dcba0ce4464af352c3aaf3061707a26213baf055917d24130af9c0`.

## Decision

Reject the PW-0099 sparse-repaired Metal executor as a complete retained-cache
token embodiment. Its layer-43 component result does not generalize through
accumulated real layer states, and warm repeated component timing overstated
the gain when 376 source experts had to be installed across a complete token.
Do not enable this endpoint as a default, do not repeat the full walk, and do
not report its component rate or 0.01321 token/s diagnostic as accepted TPS.

Preserve two separate next questions. Correctness needs a cheap layer-4 cached
oracle localization to determine whether another BF16 midpoint amplification
escaped the fixed four-ULP predicate. Embodiment needs a cold expert-working-
set and buffer-install attribution; the 75.7-second token shows that kernel
arithmetic alone is not the endpoint bottleneck after integration. Neither
question authorizes a broader threshold, a persistent model-wide expansion, or
another full walk before a bounded falsification experiment passes.
