# PW-0067 — Full-prefix frontier replay after layer 4

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean
  `d96c5bc324a45913413d4b6b0b9f574d89539770`
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

Run 001 stopped before final-layer capture with exit code 134. A later router
sigmoid called the scalar SLEEF exponential with a negative argument whose
rounded base-2 exponent is below the normal F32 exponent range. The port used
one direct exponent-bit construction and panicked when `q + 127` was negative.
Pinned SLEEF instead uses `vldexp2`: it splits `q` into two representable
powers of two and multiplies twice, preserving subnormal results. This is an
implementation defect in the recently introduced SLEEF port, not a model
capacity or memory stop.

No trace manifest was written and no comparison was attempted. The preserved
failure artifact hashes to
`60a39dbd4437502ba930bc88bcdb51554c657fc89439111c9a8fadc672bf9470`.
The last external observation at 12:23 showed about 428 MB resident, 81%
system-free memory, and zero swap growth; the LM head had not begun. No host
safety threshold was approached.

## Decision

Reject this run as correctness evidence while preserving its failure. Add a
source-exact `vldexp2` subnormal fixture and repair in a separate experiment,
then repeat the full-prefix replay without changing any acceptance or memory
threshold.
