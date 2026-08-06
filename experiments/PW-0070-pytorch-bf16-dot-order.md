# PW-0070 — PyTorch aarch64 BF16 dot-product order

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0069 comparison
  `01f212e3f16e0f3d5f03899a9608272c770691a1d0ae4777d186635fa792ba98`
- Hardware/runtime: Apple M1 shared 16 GiB host; installed PyTorch 2.13.0
  commit `cf30153c4c131c8164ee7798e5022d810682e2cb`; Rust native runtime
- Related records: PW-0065, PW-0066, PW-0068, PW-0069

## Hypothesis and contract

PW-0069's only first-stage mismatch is a length-192 BF16 query/key dot. The
pinned PyTorch fallback GEMM source computes reduced-precision dots as four
independent F32 partial sums, then adds partials 1, 2, and 3 into partial 0.
Rust currently performs one forward F32 sum. On the failing pair, the two sums
straddle a BF16 rounding boundary before attention scaling.

Create a small deterministic fixture from the hash-bound PW-0069 oracle query,
key, and sink captures. It must contain the exact 192 BF16 input bit patterns,
the pinned-source four-lane and old forward F32 replay bits, the PyTorch
post-dot BF16 bits, the scaled BF16 score bits, the row maximum, and the
centered-score bits. Record source manifest and capture hashes; do not commit
full real-model captures.

First prove that a source-derived four-lane replay matches the installed
PyTorch bits and the failing centered score while forward accumulation does
not. Then implement one named four-lane dot helper, gate it with the fixture,
and replay the complete layer-7 trace. The repair is promoted only if all
preexisting Rust/Python tests and strict Clippy pass, the complete layer-7
comparison is exact or advances the first boundary, and every shared-host
safety stop remains enforced.

Generated evidence stays external. This experiment changes no model identity,
hosted threshold, acceptance threshold, or throughput constant.

## Result

Unexecuted.

## Decision

Unexecuted.
