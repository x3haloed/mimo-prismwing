# PW-0063 — PyTorch-compatible router sigmoid

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation commit
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0061 comparison 002
  `dc0aee3b534f8ccc6ea37c1e8cf47215a9c848571dc5dc400abffbcaafa88a09`
- Related records: PW-0058, PW-0061, PW-0062

## Hypothesis and contract

PyTorch CPU vector sigmoid uses its SLEEF U10 vector exponential, while Rust's
scalar standard-library exponential differs by one F32 ULP on 165 of 6,912
real layer-2 logits. The tiny score error survives normalized routing and
changes nine final BF16 values.

Freeze at least 32 real mismatch logits and exact PyTorch F32 score bits. Port
the SLEEF U10 float exponential operation order and constants without adding a
framework dependency. Use the resulting sigmoid scores for both corrected
top-k selection and uncorrected normalized weights. Preserve the existing
generic scalar router fixture as a distinct diagnostic path.

Pass only if every frozen bit pattern, all existing tests, and a repeated
PW-0061 layer-2 trace pass; probabilities through final residual, routes, and
weights must satisfy the original gates. Retain all shared-host stops. This is
a correctness repair, not a performance or hosted-threshold change.

## Result

Unexecuted.

## Decision

Unexecuted.
