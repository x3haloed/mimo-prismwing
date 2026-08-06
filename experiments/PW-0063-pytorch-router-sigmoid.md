# PW-0063 — PyTorch-compatible router sigmoid

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation; clean implementation
  `a3b100dd254f84b142209e9e2974cfe25c79a733`
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

The SLEEF U10 scalar port reproduces every frozen PyTorch F32 sigmoid bit. The
32-value fixture is drawn from the 165 mismatches among 6,912 real layer-2
router logits. All 31 Rust tests, 42 Python tests, strict Clippy, and the
release build pass.

The first repeated trace made all router scores exact, but did not remove the
route-weight or final-state difference. This falsified the narrower assumption
that exact sigmoid scores alone were sufficient. Source inspection and replay
then identified a second coupled semantic: PyTorch CPU `topk(sorted=False)`
returns libc++ `std::nth_element` order, and its eight-value sum reduces four
vector lanes in that order. A frozen 256-score real route now gates that exact
unsorted order. The small C++ bridge catches all exceptions, validates the
selection boundary in Rust, and fails closed on invalid or tied inputs.

PW-0061 Rust run 004 is bit-exact against the oracle for all 21 captures from
incoming state through final residual, including router scores, all 216 expert
placements, and routed output. Selected experts are exact in order. The
comparison's `7.23e-9` route-weight delta is decimal JSON spelling only: all
216 values have identical F32 bits. The run completed in 26.685 seconds,
peaked at 728,858,624 bytes, retained 81% system-free memory, ended at
399,634,112 bytes, grew no swap, observed no throttling, and retained every
protected service.

Rust manifest hash:
`311bbf4df8b927915d48c2a26100ab71f08a14571bdf667e53e69603f6b223a6`.
Comparison hash:
`8aaeeeb9c9fb5698867dc7da68932a7d3b75679d7023f4e0cfb6fdad996ba832`.

## Decision

Promote the SLEEF sigmoid, libc++ unsorted top-k order, and PyTorch eight-value
reduction as one correctness repair. Layer 2 is cleared without relaxing any
gate. Repeat the full-prefix Rust trace against the frozen PW-0060 oracle to
locate the next accumulated boundary; make no throughput claim.
