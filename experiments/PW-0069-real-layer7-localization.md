# PW-0069 — Real layer-7 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0068 comparison
  `bc380e725d358594d6f73b8ec4e2b87371017eb4e1b7af47d2071ce985363799`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0064 through PW-0068

## Hypothesis and contract

PW-0068 proves layer 6 is an exact accumulated source boundary and layer 7 is
the first failure. Extend the already-regressed routed-layer diagnostic to
target layer 7 without duplicating its execution or comparison authority.

Rust must causally recompute layers 0–6, then capture production layer 7.
PyTorch must load the hash-verified PW-0060 layer-6 final and independently
derive layer-7 attention, routes, selected experts, expert tensors, scatter,
and residual. Capture the same 21 substages and bind the same source, revision,
checkpoint, prompt, numerical policy, shapes, schedules, and hashes as PW-0065.

Keep all BF16, final-state, exact-expert, and `5e-7` route-weight gates. Name
the first differing substage before changing arithmetic. Retain every
phase-level RSS/free-memory/swap/throttling/release/protected-service stop.
Generated tensors stay external; accepted tokens are zero and no throughput
or hosted threshold can change.

## Result

Unexecuted.

## Decision

Unexecuted.
