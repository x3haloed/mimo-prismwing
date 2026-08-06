# PW-0078 — Real layer-14 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0077 comparison
  `60b14d05e68f06c0fe4246cb856aec960f6382090f62affb0fdf1bdb7db518be`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0074 through PW-0077

## Hypothesis and contract

PW-0077 proves layer 13 is the last bit-exact accumulated state and layer 14
is both the first actual and formal divergence. Extend the generalized
routed-layer diagnostic selector to layer 14 without adding execution or
comparison authority.

Rust must causally recompute layers 0–13, then capture production layer 14.
PyTorch must load the hash-verified PW-0060 layer-13 final and independently
derive the same 21 attention, routing, selected-expert, scatter, and residual
substages. Bind all source, revision, checkpoint, prompt, numerical-policy,
shape, schedule, commit, and evidence hashes. Preserve every existing BF16,
final-state, exact-expert, and `5e-7` route-weight gate. Identify the first
actual differing substage before changing arithmetic.

Retain normative Gate 8 at every phase, including free-memory, current/peak
RSS, post-release footprint, swap-growth, throttling, buffer-release, allocator,
and protected-service stops. Record batch 1, concurrency 1, accepted tokens 0,
and wall time. This diagnostic cannot alter any acceptance threshold or count
as TPS.

## Result

Unexecuted.

## Decision

Unexecuted.
