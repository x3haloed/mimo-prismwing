# PW-0060 — Full-prefix layer-final localization

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: Rust trace clean
  `14ae03afce14dcd1ae6d348f017c97092656490d`; oracle split-shard repair clean
  `f2fbba1`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0059 comparison
  `72d0b7156984c41f418b736e18dcbccef0d05f95060beccc62d54f40797ed209`;
  frozen hosted chat reference from PW-0052
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, PW-0050 safety contract
- Related records: PW-0050, PW-0052, PW-0056 through PW-0059

## Hypothesis and mechanism

Dense layer 0 and complete routed layer 1 independently clear every distinct
model semantic category, yet the 48-layer local endpoint remains far from the
hosted result. The remaining cheap discriminator is accumulated execution:
compare only each causal layer's final state across the complete frozen
27-token prefix, avoiding another speculative arithmetic change.

## Contract

Run one production Rust prefill and one independent readable PyTorch prefill,
serially. Both must derive the exact frozen embedding and all 48 layers from
the verified checkpoint. Capture only embedding, each layer-final BF16 state,
final normalization, and last-position F32 logits. The oracle derives every
route and executes only its own selected experts; it cannot consume Rust
intermediates or routes. The Rust path uses the endpoint's production
attention, router, expert, and residual authorities.

Bind revision, prompt IDs, fixture, verification, tensor index/shards, dynamic
FP8 policy, BF16 operation staging, capture schema, routes, and hashes. Compare
each BF16 boundary at relative L2 at most `5e-4`, maximum absolute error at
most `2e-2`, and equality at least 99%; preserve the PW-0049 stronger final
layer limit of relative L2 `4e-5` and maximum absolute error `3e-6`. Compare
final logits and selected hosted token logits explicitly. Stop and localize the
first failing layer; if all layers clear, classify the hosted/local mismatch as
external-reference identity or serving-semantics evidence before changing
local arithmetic again.

This is the first new full walk after the shared-host concern was raised. Both
authorities must release mapped pages, decoded matrices, and allocator
transients at least at every matrix/expert and completed layer. Check and fail
closed after every layer (and every routed expert where practical) below 20%
system-free memory, above 8 GiB current or peak process footprint, above 4 GiB
post-phase footprint, above 512 MiB swap growth, on any new throttled page, or
when a start-resident protected service disappears. Record current/peak RSS,
swap, pressure, relief, service PIDs, logical/actual bytes, union per layer,
cold/warm state, batch 1, concurrency 1, accepted tokens 0, hardware, commit,
and wall time.

Generated tensors remain external and content-addressed. This diagnostic may
not weaken hosted thresholds, become accepted TPS, or promote a performance
default.

## Result

Rust run 001 completed all 48 layers. Oracle run 001 failed closed after layer
42 when layer-43 expert 223's `up_proj` weight and scale resolved to different
verified shards. The oracle had incorrectly required co-location; the Rust
checkpoint authority already resolves every tensor independently. Run 001 is
preserved without a manifest. The repaired independent resolver remains
bit-exact on a co-located control and executes the real split tensor; oracle
run 002 then completed all layers.

Comparison 001 decisively localizes the first mismatch. Embedding, dense layer
0, and complete routed layer 1 are bit-exact. Layer 2 is the first failing
boundary: relative L2 `0.00041550844076489825`, maximum absolute error
`0.0625`, and BF16 equality 99.4520%. Its selected expert sets remain exact at
all positions, but maximum route-weight error reaches
`1.7813905143770903e-6`, exceeding the `5e-7` gate. Subsequent route weights,
sets, and states compound the divergence; layer 47 reaches relative L2
`0.02537`. This is localization evidence, not proof that route weights alone
cause every layer-2 state difference.

Evidence hashes:

- independent oracle run 002 manifest:
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
- Rust run 001 manifest:
  `3ea25262c1d03dec400d4686e9a31c877faee8c1049fa76a8baa8a09a468913b`;
- comparison 001:
  `2b74e0c622f1e44947307820067d8633cd972096af558376b9dd4d71edf1bbdf`.

Rust completed in 786.738 seconds, with minimum 81% system-free memory,
4,159,897,600 bytes peak RSS during the full 27-row LM head, 2,913,704,384
bytes post-capture footprint, no swap growth, no throttling, and all protected
services healthy. Oracle run 002 completed in 682.551 seconds, retained at
least 75% system-free memory, peaked at 3,872,161,792 bytes, ended phases below
508 MB current RSS, grew no swap, and observed no throttling. The host-safety
contract passed both full walks. These are cold correctness walls with accepted
tokens zero and change no throughput-model constant.

## Decision

Promote the full-prefix trace as a correctness localization tool. Preserve the
split-shard oracle failure and repair. Supersede the belief that an unobserved
late layer is needed to find the first accumulated divergence: it begins at
layer 2, immediately after the two cleared layers. Open a bounded layer-2
attention/router/expert trace from the bit-exact layer-1 final state before any
new full walk or arithmetic change.
