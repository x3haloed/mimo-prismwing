# PW-0067 — Full-prefix frontier replay after layer 4

- Status: running
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0066 comparison
  `432ac5db9ab02ff5031442424e7a0c72b0f0b9319ce07e952c3d951534f6e759`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0064 through PW-0066

## Hypothesis and contract

PW-0066 restores exact accumulated layer-4 state. One production Rust replay
against the immutable PW-0060 oracle is the cheapest way to advance the exact
frontier and find the next causal boundary without regenerating an oracle.

Repeat the frozen 27-token prefill through all 48 layers with identical
embedding, layer-final, final-norm, logit, route, and weight captures. Bind the
verified checkpoint, revision, fixture, numerical policy, schema, hashes, and
clean commit. Keep the BF16 `5e-4` relative-L2, `2e-2` maximum-error, 99%
equality gates; final `4e-5`/`3e-6` gates; exact expert sets; and `5e-7`
route-weight gate. Stop semantic speculation at the first failing layer.

Retain phase-level current/peak footprint, system-free memory, swap growth,
throttling, allocator relief, buffer/page release, and protected-service
checks. Fail closed below 20% free memory, above 8 GiB current/peak, above
4 GiB post-phase, above 512 MiB swap growth, on throttling, or service loss.
Record warm/uncontrolled cache state, batch 1, concurrency 1, accepted tokens
0, wall time, bytes, hardware, and commit. This cannot count as TPS or alter
hosted thresholds.

## Result

Unexecuted.

## Decision

Unexecuted.
