# PW-0066 — PyTorch ARM softmax operation order

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean implementation
  `ab2a9407979e65e7116628ecc2a107953ad25060`
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

Both frozen PW-0065 rows pass bit-exactly at F32 and BF16. All 31 Rust tests,
42 Python tests, strict Clippy, and the release build pass. A content-addressed
replay over 101,952 real probability values from layers 0, 1, 2, and 4 reports
zero BF16 mismatches; its evidence hash is
`16becf506663acc1dd823ee69e0e5ca44d745288277b5ceb06eef4c3a4b1d516`.

PW-0065 Rust run 002 makes all 25,920 layer-4 probabilities bit-exact, and the
post-attention residual onward is bit-exact through all router, expert,
scatter, and final captures. Every selected expert and all 216 route-weight
F32 values are bit-exact. One attention-output value differs by
`0.000244140625`, and five projection values differ by at most
`0.0009765625`; both are far inside their gates and disappear at the BF16
residual boundary. They are a separate attention-accumulation-order diagnostic,
not evidence against this softmax repair.

The Rust replay completed in 58.675 seconds, peaked at 720,568,320 bytes,
retained at least 80% system-free memory, ended at 127,454,848 bytes, grew no
swap, observed no throttling, and retained every protected service. Rust
manifest hash:
`2b1ec903c27241c096b78199930e36e6837f906f3b1e9a413513a9780411c224`.
Comparison hash:
`432ac5db9ab02ff5031442424e7a0c72b0f0b9319ce07e952c3d951534f6e759`.

## Decision

Promote the PyTorch ARM softmax operation order as a correctness repair. It
removes the first PW-0065 difference and restores exact accumulated layer-4
state without weakening any gate. The tiny pre-residual attention accumulation
delta is preserved but does not justify blocking the next full-prefix
localization replay. No throughput or hosted threshold changes.
