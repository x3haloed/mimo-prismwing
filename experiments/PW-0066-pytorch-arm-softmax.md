# PW-0066 — PyTorch ARM softmax operation order

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PyTorch
  `cf30153c4c131c8164ee7798e5022d810682e2cb`; PW-0065 comparison
  `4ee90dfffe0523382e5155bfe2eb27be359a8a544c3345a0ec4e46f5dc8ab54a`
- Related records: PW-0057, PW-0061, PW-0062, PW-0065

## Hypothesis and contract

Rust's vForce exponential, reverse scalar denominator, and per-value division
do not reproduce the installed PyTorch aarch64 F32 softmax kernel universally.
The pinned PyTorch source uses four-lane SLEEF exponentials, lane-wise
accumulation, ARM horizontal reduction, one reciprocal, and multiplication.
That operation order should remove both first layer-4 differences while
preserving all earlier real probability corpora.

Freeze the exact two-value and 25-value centered-score rows from PW-0065 with
their PyTorch F32 and BF16 probability payloads. Implement the pinned
four-lane algorithm using the already-gated SLEEF U10 exponential. Preserve a
separate scalar diagnostic path. Fail closed on empty, non-finite, or invalid
denominator inputs.

Pass only if the new fixtures are bit-exact, every existing fixture and test
passes, and replay over the complete real layer-0, layer-1, layer-2, and
layer-4 score corpora produces zero BF16 mismatches. Then repeat PW-0065's
layer-4 Rust trace against the frozen oracle. Incoming through final residual,
routes, and weights must meet the original gates; no threshold may move.

Retain all shared-host stops. This is a correctness repair with accepted
tokens zero, not a performance result or hosted-threshold change.

## Result

Unexecuted.

## Decision

Unexecuted.
