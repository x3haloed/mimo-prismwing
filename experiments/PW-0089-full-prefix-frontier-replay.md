# PW-0089 — Full-prefix frontier replay after layer 34

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0088 comparison
  `967a7f9d0ee0c0b004c8b1b365b68cd1ff2c4cca2c280d93318de4950d1274aa`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0086 through PW-0088

## Hypothesis and contract

PW-0088 makes layer 34 exact from frozen exact layer 33. Repeat the frozen
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
