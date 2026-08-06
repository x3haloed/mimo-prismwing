# PW-0077 — Full-prefix frontier replay after layer 13

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0076 comparison
  `e260595915439eaef8edd2ee0cc4f07950295a3fe209a0acb8909e710ce2f279`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0074 through PW-0076

## Hypothesis and contract

PW-0076 makes layer 13 exact from the frozen exact layer-12 state. One
production Rust replay against the immutable PW-0060 oracle is the cheapest
way to prove the accumulated frontier and locate the next causal boundary.

Repeat the frozen 27-token prefill through all 48 layers with identical
embedding, layer-final, final-norm, logit, route, and weight captures. Bind the
verified checkpoint, revision, fixture, numerical policy, schema, hashes, and
clean commit. Preserve all existing correctness thresholds and distinguish the
last bit-exact layer, first actual divergence, and first formal gate failure.

Enforce normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after release, above 512 MiB swap
growth, on new throttled pages, or on protected-service loss. Record buffer
release, allocator relief, hardware, commit, cache state, batch 1, concurrency
1, accepted tokens 0, and complete wall time. Preserve stopped evidence.

This cannot count as TPS or alter any hosted, capability, fidelity, cost,
power, safety, or performance threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
