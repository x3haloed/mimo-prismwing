# PW-0071 — Full-prefix frontier replay after layer 7

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0070 comparison
  `62ea2df8ba4494959e5fdb9544af7ac758b31e54a73b7508bdc17212dd14d472`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0064, PW-0067 through PW-0070

## Hypothesis and contract

PW-0070 restores exact accumulated layer-7 state. One production Rust replay
against the immutable PW-0060 oracle is now the cheapest way to advance the
exact frontier and find the next causal boundary without regenerating an
oracle or speculating about downstream arithmetic.

Repeat the frozen 27-token prefill through all 48 layers with identical
embedding, layer-final, final-norm, logit, route, and weight captures. Bind the
verified checkpoint, revision, fixture, numerical policy, schema, hashes, and
clean commit. Keep the BF16 `5e-4` relative-L2, `2e-2` maximum-error, 99%
equality gates; final `4e-5`/`3e-6` gates; exact expert sets; and `5e-7`
route-weight gate. Stop semantic speculation at the first failing layer.

This is the first full walk under the normative Gate 8 shared-host policy.
At every phase, record and enforce current footprint, peak RSS, system-free
memory, swap growth, throttling, allocator relief, buffer/page release, and
protected-service health. Fail closed below 20% free memory, above 8 GiB
current/peak, above 4 GiB after declared release, above 512 MiB swap growth, on
any new throttled page, or on start-resident service loss. Preserve a stopped
run as failed evidence.

Record cold/warm state as observed, batch 1, concurrency 1, accepted tokens 0,
wall time, bytes, hardware, and commit. This cannot count as TPS or alter any
hosted, capability, fidelity, cost, power, or performance threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
