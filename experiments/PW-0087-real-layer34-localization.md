# PW-0087 — Real layer-34 substage localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0086 comparison
  `d23e411ab91712636d45553463ef162652403a60d3ee76f9bb835c007dce001f`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0085, PW-0086

## Hypothesis and contract

PW-0086 proves layer 33 is the last bit-exact accumulated state and layer 34
is the first actual divergence, even though the six-value delta does not
formally fail the layer-final gate until layer 36. Extend the generalized
routed-layer selector to layer 34 without adding execution or comparison
authority.

Rust must causally recompute layers 0–33, then capture production layer 34.
PyTorch must load the hash-verified PW-0060 layer-33 final and independently
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
