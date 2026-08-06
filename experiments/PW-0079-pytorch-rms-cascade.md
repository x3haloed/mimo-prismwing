# PW-0079 — PyTorch F32 RMS cascade reduction

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0078 oracle
  `53e6b5db1d63128fddc2d3d6a8445424021f89f6f20131c98a42ab857f819e1f`
- Hardware/runtime: Apple M1 shared 16 GiB host, PyTorch 2.13.0 CPU oracle,
  production Rust trace
- Related records: PW-0077, PW-0078

## Hypothesis and contract

PW-0078 proves layer 14 first diverges because the prior F64 RMS square sum
does not reproduce PyTorch's F32 contiguous-inner cascade. Extract the exact
4,096-value BF16 position-1 post-attention row from the hash-verified oracle.
Record the input payload, PyTorch variance and inverse bits, and the prior
high-precision variance in a deterministic fixture.

Implement the pinned `SumKernel.cpp` topology: four interleaved vector rows,
four hierarchy levels, a minimum 16-vector chunk, source-order row and vector
lane reduction, F32 division, epsilon addition, square root, and reciprocal.
The fixture must distinguish it from the prior reduction. Run all Rust/Python
tests, strict lint, deterministic regeneration, and the real layer-14 replay.
Promote only if all 21 captures and routing pass existing gates.

Retain normative Gate 8. This is correctness work, with accepted tokens zero;
it cannot count as TPS or alter any acceptance threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
