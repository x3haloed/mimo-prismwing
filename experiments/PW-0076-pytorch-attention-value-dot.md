# PW-0076 — PyTorch BF16 attention-value dot order

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0075 oracle
  `294e25355d4cb6ca3dcdcb060e131e7599b6603987eaaf0664a39f95ff0ddf74`
- Hardware/runtime: Apple M1 shared 16 GiB host, PyTorch 2.13.0 CPU oracle,
  production Rust trace
- Related records: PW-0072, PW-0073, PW-0075

## Hypothesis and contract

PW-0075 identifies one 25-element BF16 probability/value dot where forward
F32 accumulation lands on a BF16 tie and PyTorch does not. The pinned PyTorch
specialized reduced-precision GEMV source applies the same eight-vector,
pairwise-reduction, vector-tail, and scalar-tail topology already gated for
global-attention query/key dots.

Extract the exact position-24/head-4/value-dimension-52 pair from the
hash-verified PW-0075 oracle. Record both operand payloads, source capture
hashes, PyTorch result, forward result, and source-topology result in a small
deterministic fixture. The fixture must prove that the source topology rounds
to PyTorch while forward accumulation does not.

Only then use the existing specialized helper for BF16 attention-value dots.
Run all Rust/Python correctness tests, strict lint, deterministic fixture
regeneration, and a real layer-13 replay against the immutable oracle. Every
one of the 21 captures, expert sets/order, and route weights must pass existing
thresholds before promotion. Preserve normative Gate 8. This is correctness
work and cannot count as TPS or alter any acceptance threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
