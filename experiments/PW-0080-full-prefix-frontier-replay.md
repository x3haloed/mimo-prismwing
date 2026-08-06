# PW-0080 — Full-prefix frontier replay after layer 14

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0079 comparison
  `a37e5af67dc8cb4f95ee4ca30cd8af30e62ec3266771c6454fed27be95844ece`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0077 through PW-0079

## Hypothesis and contract

PW-0079 makes layer 14 exact from the frozen exact layer-13 state. Repeat the
frozen production 27-token prefill through all 48 layers against the immutable
PW-0060 oracle to prove the accumulated frontier and identify the next causal
boundary.

Capture identical embedding, layer-final, final-norm, logit, route, and weight
artifacts. Bind the verified checkpoint, revision, fixture, numerical policy,
schema, hashes, and clean commit. Preserve every existing correctness threshold
and distinguish the last bit-exact layer, first actual divergence, and first
formal gate failure.

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
