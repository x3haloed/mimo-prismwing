# PW-0084 — Real layer-29 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0083 comparison
  `c8c6b94313aa780fe1fb1d728529d8fa903e06c4182404c2e096247b2a40c75f`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0082, PW-0083

## Hypothesis and contract

PW-0083 proves layer 28 is the last bit-exact accumulated state and layer 29
is both the first actual and formal divergence. Extend the generalized
routed-layer selector to layer 29 without adding execution or comparison
authority.

Rust must causally recompute layers 0–28, then capture production layer 29.
PyTorch must load the hash-verified PW-0060 layer-28 final and independently
derive the same 21 substages. Bind source, revision, checkpoint, prompt,
numerical policy, shapes, schedule, commit, and evidence hashes. Preserve all
BF16, final-state, exact-expert, and `5e-7` route-weight gates. Identify the
first actual differing substage before changing arithmetic.

Retain normative Gate 8 at every phase, record batch 1, concurrency 1,
accepted tokens 0, buffer release, allocator relief, and complete wall time,
and preserve stopped evidence. This diagnostic cannot count as TPS or alter
any threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
