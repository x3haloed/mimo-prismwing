# PW-0072 — Real layer-11 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0071 comparison
  `744fad6a7ba4b9ea883c5f53eda2f4fafa67569e82718a65ec9cbdaac526a9c4`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0065, PW-0069 through PW-0071

## Hypothesis and contract

PW-0071 proves layer 10 is the last bit-exact accumulated state and layer 11 is
the first actual divergence. Extend the already-regressed routed-layer
diagnostic to target layer 11 without duplicating execution or comparison
authority.

Rust must causally recompute layers 0–10, then capture production layer 11.
PyTorch must load the hash-verified PW-0060 layer-10 final and independently
derive layer-11 attention, routes, selected experts, expert tensors, scatter,
and residual. Capture the same 21 substages and bind the same source, revision,
checkpoint, prompt, numerical policy, shapes, schedules, and hashes as PW-0069.

Keep all BF16, final-state, exact-expert, and `5e-7` route-weight gates. Name
the first differing substage before changing arithmetic. Retain every normative
Gate 8 phase-level RSS/free-memory/swap/throttling/release/protected-service
stop. Generated tensors stay external; accepted tokens are zero and no
throughput or hosted threshold can change.

## Result

Unexecuted.

## Decision

Unexecuted.
