# PW-0081 — Real layer-19 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0080 comparison
  `eb2f578b983a6be8befc29dc2724607d33fa81ec6cc4a77311dda1ad8a7d02c2`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0079, PW-0080

## Hypothesis and contract

PW-0080 proves layer 18 is the last bit-exact accumulated state and layer 19
is both the first actual and formal divergence. Extend the generalized
routed-layer selector to layer 19 without adding execution or comparison
authority.

Rust must causally recompute layers 0–18, then capture production layer 19.
PyTorch must load the hash-verified PW-0060 layer-18 final and independently
derive the same 21 substages. Bind source, revision, checkpoint, prompt,
numerical policy, shapes, schedule, commit, and evidence hashes. Preserve all
BF16, final-state, exact-expert, and `5e-7` route-weight gates. Identify the
first actual differing substage before changing arithmetic.

Retain normative Gate 8 at every phase, record batch 1, concurrency 1,
accepted tokens 0, and wall time, and preserve stopped evidence. This
diagnostic cannot count as TPS or alter any threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
