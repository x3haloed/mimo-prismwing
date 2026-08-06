# PW-0086 — Full-prefix frontier replay after layer 29

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0085 comparison
  `716fa337cde3e90de10342f46afafd802d5b78f5b73a2e82e7c90ef9462da5b3`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0083 through PW-0085

## Hypothesis and contract

PW-0085 makes layer 29 exact from frozen exact layer 28. Repeat the frozen
production 27-token prefill through all 48 layers against the immutable oracle
to prove the accumulated frontier and identify the next causal boundary.

Capture identical embedding, layer-final, final-norm, logit, route, and weight
artifacts. Bind checkpoint, revision, fixture, numerical policy, schema,
hashes, and clean commit. Preserve every correctness threshold and distinguish
the last bit-exact layer, first actual divergence, and first formal failure.

Enforce normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after release, above 512 MiB swap
growth, on new throttled pages, or on protected-service loss. Record release,
allocator relief, hardware, commit, cache state, batch 1, concurrency 1,
accepted tokens 0, and wall time. Preserve stopped evidence.

This cannot count as TPS or alter any hosted, capability, fidelity, cost,
power, safety, or performance threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
