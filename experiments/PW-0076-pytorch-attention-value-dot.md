# PW-0076 — PyTorch BF16 attention-value dot order

- Status: complete
- Disposition: promoted correctness repair
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

The deterministic 25-element fixture hashes to
`e059be02f3c9b8ea2e710dabc2d57b89c93669d47bd23df606c4c204b815b7f3`
and regenerates byte-for-byte from the immutable PW-0075 oracle. Forward F32
accumulation produces raw `0xbcaa8000` and rounds to BF16
`-0.020751953125`; the pinned specialized vector-tail topology produces raw
`0xbcaa8002` and rounds to PyTorch's `-0.0208740234375`.

Production BF16 attention now uses that topology independently for each value
dimension. The initial full suite correctly caught that SWA probability rows
include a learned sink with no corresponding value row; the implementation
was withheld until it explicitly excluded the sink from the value dot while
preserving its normalization effect. The corrected implementation passes 35
Rust tests, 42 Python tests, strict clippy, and deterministic regeneration.

The production layer-13 replay completed in 213.851 seconds. All 21 captures
are bit-exact against the immutable PyTorch oracle, including attention,
projection, residuals, router logits/scores, every selected-expert tensor,
scatter, and final state. Expert sets and order are exact; maximum route-weight
serialization error is `2.6036071743007483e-8`, below `5e-7`.

Gate 8 passed: peak RSS was 753,500,160 bytes, maximum physical footprint was
663,265,536 bytes, final footprint was 133,203,712 bytes, system-free memory
stayed at or above 82%, swap growth and new throttled pages were zero, and all
protected services remained healthy. Evidence hashes:

- Rust manifest:
  `4df0abc60c7cd942382fd90903c392ead131b49353ba501d7df1e96437966d55`
- Comparison:
  `e260595915439eaef8edd2ee0cc4f07950295a3fe209a0acb8909e710ce2f279`

## Decision

Promote the source-pinned attention-value reduction as a correctness repair.
Layer 13 is exact from the frozen layer-12 state, so the accumulated exact
frontier can advance through layer 13 after one frozen full-prefix replay.
The missing-parent preflight attempt produced no model execution or evidence
manifest and is not a semantic or safety result. No throughput, hosted,
fidelity, or acceptance threshold changes.
