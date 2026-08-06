# PW-0082 — Specialized BF16 SWA score dot

- Status: complete
- Disposition: promoted correctness repair
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

The discriminating real fixture hashes to
`b109634299b88146687d18b1c266b0313fa7beeaabaed81c940f3659029230be`
and regenerates byte-for-byte from the immutable PW-0081 oracle. It records
the specialized raw result `0x41368001`, which rounds to PyTorch's BF16
`11.4375`, and the four-lane raw result `0x41368000`, which rounds to the
adjacent `11.375`. The historical PW-0070 fixture also remains byte-identical,
and its specialized result rounds to the same BF16 oracle value as four-lane.

Production BF16 score dots now consistently use the pinned specialized
eight-vector topology. The implementation passes 37 Rust tests, 42 Python
tests, strict clippy, and deterministic regeneration of both the new
discriminator and PW-0070 historical fixture.

The production layer-19 replay completed in 304.222 seconds. All 21 captures
are bit-exact against the immutable oracle, including scores, probabilities,
attention, projections, both RMSNorms, routing, every selected-expert tensor,
scatter, and final state. Expert sets/order are exact; route-weight
serialization error is `2.2118377684954282e-8`, below `5e-7`.

Gate 8 passed: peak RSS was 746,504,192 bytes, final footprint was 137,637,440
bytes, system-free memory stayed at or above 82% and ended at 83%, swap used
decreased with zero measured growth, new throttled pages were zero, and all
protected services remained healthy. Evidence hashes:

- Rust manifest:
  `018b9d080d9160c404819a2cd1c5fd3e893e592fa687de252a8e4e55437e35ec`
- Comparison:
  `90858655eda93256a8bb5abae28acf9aafe1faa8a2da457445f42d313b832bc5`

## Decision

Promote the specialized score-dot topology as the unified production
correctness path. Preserve PW-0070's four-lane fixture and historical inference
as superseded evidence. Layer 19 is exact from frozen layer 18; one full-prefix
replay may advance the accumulated frontier. No threshold or throughput change.
