# PW-0057 — vForce BF16 attention softmax

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean `8f4e8b611f50e6acf86a701636040628615d1e76`
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

The source-ordered fixture hashes to
`e4826e16debf06430741d67ac235f2f2c5ef7517caf49f31b80a9f1e99e3e0c8`.

Change only BF16 attention exponential evaluation on Apple to `vvexpf`.
Retain stable max subtraction, F32 normalization, BF16 cast, sink semantics,
all other numerics, and every hosted threshold.

Success requires all component gates and a repeated PW-0056 layer-0 trace
whose attention probabilities and every downstream capture satisfy the
PW-0056 limits. Only then may a frozen hosted-prefix walk be considered.

## Result

All length-2, length-7, and length-27 fixture probabilities match PyTorch BF16
payloads exactly. The pre-existing F32 scalar attention fixture remains
separate and passes. All 29 Rust tests, 37 Python tests, strict Clippy, and the
release build pass.

The investigation also corrected the independent oracle: the pinned model
performs BF16 max-subtraction before F32 softmax. With that source ordering,
vForce probabilities, attention output, output projection, residual/norm,
dense SwiGLU, down projection, and final layer state all match the PW-0056
oracle bit-for-bit. The only non-exact final capture is centered QK scores at
`2.85e-6` relative L2, 99.9959% BF16 equality, and `7.63e-6` maximum error.

## Decision

Promote vForce exponential evaluation for the Apple BF16 attention path as a
correctness repair. It realizes the selected executable PyTorch CPU oracle and
clears the complete real layer-0 gate. This is not yet a hosted-parity result;
the next diagnostic is routed layer 1 before any new whole-model walk.
