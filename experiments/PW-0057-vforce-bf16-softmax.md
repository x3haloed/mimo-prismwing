# PW-0057 — vForce BF16 attention softmax

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract and fixture precede runtime change
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0056 comparison
  `c45a03b379673884e726fd65175a41f343ae69fac7b8c8c0265fe0debbf0662c`
- Hardware/runtime: Apple M1, Accelerate vForce, PyTorch 2.13.0 CPU oracle,
  shared-host PW-0050 safety contract
- Related records: PW-0054, PW-0056

## Hypothesis and mechanism

PW-0056 is bit-exact through scaled QK scores. Its first divergence is the
F32 softmax: Rust scalar `exp` yields 99.797% BF16 probability equality and the
resulting attention output falls to 98.793%. On the complete captured layer-0
score corpus, Accelerate `vvexpf` followed by the same F32 normalization and
BF16 cast matches PyTorch 2.13.0 probabilities bit-for-bit.

## Contract

Before runtime change, commit seeded PyTorch BF16 score/probability cases of
length 2, 7, and 27, including large offsets and near ties. Rust vForce softmax
must match every expected BF16 probability payload. Preserve the existing F32
scalar attention oracle as a distinct passing path. Reject empty, non-finite,
or zero-denominator inputs.

Change only BF16 attention exponential evaluation on Apple to `vvexpf`.
Retain stable max subtraction, F32 normalization, BF16 cast, sink semantics,
all other numerics, and every hosted threshold.

Success requires all component gates and a repeated PW-0056 layer-0 trace
whose attention probabilities and every downstream capture satisfy the
PW-0056 limits. Only then may a frozen hosted-prefix walk be considered.

## Result

Unexecuted.

## Decision

Unexecuted.
