# PW-0075 — Real layer-13 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0074 comparison
  `0dd64f521715c86fea52557168a5101cdaef76421269b6ed6c1b46b964c9ced6`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0071 through PW-0074

## Hypothesis and contract

PW-0074 proves layer 12 is the last bit-exact accumulated state and layer 13
is the first actual divergence. Extend the generalized routed-layer diagnostic
selector to layer 13 without introducing a second execution or comparison
authority.

Rust must causally recompute layers 0–12, then capture production layer 13.
PyTorch must load the hash-verified PW-0060 layer-12 final and independently
derive the same 21 attention, routing, selected-expert, scatter, and residual
substages. Bind source input, revision, checkpoint, prompt, numerical policy,
shapes, schedules, commit, and hashes. Preserve the BF16, final-state,
exact-expert, and `5e-7` route-weight gates. Name the first actual differing
substage before changing arithmetic.

Retain normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after release, above 512 MiB swap
growth, on new throttled pages, or on protected-service loss. Record buffer
release, allocator relief, hardware, batch 1, concurrency 1, accepted tokens
0, and wall time. Generated tensors stay external. This diagnostic cannot
alter hosted, fidelity, capability, cost, power, safety, or TPS thresholds.

## Result

Unexecuted.

## Decision

Unexecuted.
