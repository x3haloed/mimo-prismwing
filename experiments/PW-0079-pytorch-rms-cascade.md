# PW-0079 — PyTorch F32 RMS cascade reduction

- Status: complete
- Disposition: promoted correctness repair
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

The deterministic real-row fixture hashes to
`08f53fe2c837af5008bea88a3191d70b4458f81aecd73d66eba53753e89cdc35`
and regenerates byte-for-byte from the immutable PW-0078 oracle. It records
PyTorch variance raw `0x41913476`, inverse raw `0x3e705b06`, and the prior
high-precision variance raw `0x41913477`.

The pinned cascade implementation reproduces the fixture exactly, including
the four interleaved vectors, 16-vector hierarchy, level carry order, vector
tail, scalar tail, lane reduction, mean, epsilon, square root, and reciprocal.
It passes 36 Rust tests, 42 Python tests, strict clippy, and deterministic
fixture regeneration. A narrow documented lint exception keeps the indexed
level/register/lane topology auditable.

The production layer-14 replay completed in 229.884 seconds. All 21 captures
are bit-exact against the immutable oracle, including both RMSNorms, attention,
router logits/scores, every selected-expert tensor, scatter, and final state.
Expert sets and order are exact; route-weight serialization differs by only
`1.7046356215466574e-8`, below `5e-7`.

Gate 8 passed: peak RSS was 714,653,696 bytes, final footprint was 137,173,056
bytes, system-free memory stayed at or above 82%, swap growth and new throttled
pages were zero, and all protected services remained healthy. Evidence hashes:

- Rust manifest:
  `92d8667ecf78d5251693bf279400bcadbc496fe1d1af3b90ef83990936d5f429`
- Comparison:
  `a37e5af67dc8cb4f95ee4ca30cd8af30e62ec3266771c6454fed27be95844ece`

## Decision

Promote the source-pinned F32 RMS cascade as a correctness repair. Layer 14 is
exact from the frozen layer-13 state, so one frozen full-prefix replay may now
advance the accumulated frontier. No throughput, hosted, fidelity, or
acceptance threshold changes.
