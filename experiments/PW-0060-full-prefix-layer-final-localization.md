# PW-0060 — Full-prefix layer-final localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: diagnostic contract precedes implementation
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

Unexecuted.

## Decision

Unexecuted.
