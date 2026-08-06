# PW-0082 — Specialized BF16 SWA score dot

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0081 oracle
  `5bf6ed69aa01293e8020e3d4b2dc3a34dd087672901f59e321258f8ab1c0313b`
- Hardware/runtime: Apple M1 shared 16 GiB host, PyTorch 2.13.0 CPU oracle,
  production Rust trace
- Related records: PW-0070, PW-0072, PW-0073, PW-0081

## Hypothesis and contract

PW-0081 supplies a real SWA query/key pair that discriminates PyTorch's
specialized eight-vector BF16 dot topology from both forward and four-lane
reduction. Add it to the existing hash-bound dot-fixture generator and record
all operands, source hashes, source-topology result, competing result, scale,
row maximum, and centered score.

Only after the fixture proves the distinction, use the already pinned
specialized topology for every BF16 attention score dot. Preserve the PW-0070
four-lane fixture as historical evidence and prove its final BF16 payload still
matches under the specialized topology. Run all Rust/Python tests, strict lint,
deterministic fixture regeneration, and a real layer-19 replay. Promote only if
all 21 captures and routing pass existing gates.

Retain normative Gate 8. This is correctness work with zero accepted tokens;
it cannot count as TPS or alter any threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
