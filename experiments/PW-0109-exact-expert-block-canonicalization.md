# PW-0109 — Exact expert block canonicalization

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: pending
- Artifact authority: PW-0106 layer-4 selected-expert artifact
  `fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21`;
  manifest
  `40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8`
- Hardware/runtime: Apple M1 shared 16 GiB host; internal SSD
- Related records: PW-0013, PW-0045, PW-0106 through PW-0108

## Hypothesis and representation boundary

PW-0108 proves that exact acquisition of the unchanged 201,375,744 selected
tensor bytes is at least 10.334 ms too slow for the frozen continuation bound.
SwiGLU hidden neurons have an exact permutation symmetry: a common permutation
of gate/up output rows and the corresponding down input columns preserves the
expert function. If independently trained experts contain alignable structure,
canonicalizing that symmetry may lower cross-expert residual entropy and the
physical bytes that must be acquired.

The source format quantizes weights with 128-by-128 scale blocks. Arbitrary
single-neuron permutations would mix scale blocks and require a different or
expanded scale representation. Phase A therefore tests the deepest symmetry
that preserves the existing source-FP8 scale topology exactly: the 16 groups
of 128 intermediate neurons. For each group, move gate and up weight rows and
scale rows together with down weight columns and scale columns. No weight or
scale value changes.

## Construction contract

Build a deterministic 16-block record for each of the eight authenticated
selected experts. A block contains, in a fixed schema, the corresponding gate
rows/scales, up rows/scales, and down columns/scales. The concatenation must
cover every tensor byte exactly once. Reconstructing the original artifact
records through inverse permutations must reproduce every original tensor hash.

Choose one reference expert deterministically. Derive fixed-size byte sketches
for all 16 blocks of every expert, compute the complete 16-by-16 pairwise XOR
distance matrix against the reference, and solve the minimum-cost one-to-one
assignment with deterministic tie-breaking. Preserve the reference identity
ordering. Report permutations and exact full-block distances; the sketch may
select an assignment but may not stand in for final compression evidence.

Compare three equal-logical-byte streams:

1. unmodified expert-major block records;
2. identity-order reference plus exact XOR residuals, block-major; and
3. aligned reference plus exact XOR residuals, block-major.

Compress each stream with the same pinned lossless codec/settings at a fast
runtime level and a high-analysis level. Measure compression and decompression
wall, verify decompressed bytes, invert XOR and permutations, and reproduce all
source tensor hashes. Temporary streams remain external and are removed after
their hashes, sizes, and codec evidence are persisted. No generated payload is
committed.

## Verification and measurement

Before the real artifact, add tiny fixtures proving gate/up-row and down-column
block extraction, scale-row/column movement, assignment bijection, exact XOR
reversal, inverse-permutation reconstruction, complete byte coverage, and
fail-closed shape/layout handling.

Run the real selected-route analysis once under Gate 8; it is deterministic,
not a throughput distribution. Record source and stream hashes, byte counts,
codec binary/version/settings, compressed sizes, compression/decompression
wall, peak RSS, physical reads, swap, throttling, release, and service health.
Codec wall is diagnostic and cannot become endpoint TPS.

The PW-0108 best cold acquisition median is 58.033833 ms. For each candidate,
compute the optimistic transformed bound

`58.033833 * compressed_ratio + measured_decompression_ms`.

This deliberately assumes storage time scales perfectly with bytes and charges
no layout or dispatch overhead, so failure is decisive while success only
authorizes a deeper executable-codec experiment.

## Gates

- **Exactness/accounting:** all six source tensors per expert are covered once;
  every assignment is bijective; XOR reversal and inverse permutation recreate
  every original byte and tensor hash.
- **Canonicalization signal:** aligned residual compression must beat both the
  unmodified and identity-delta controls by at least 10% at the same codec
  setting. A generic codec gain without an alignment gain does not validate the
  symmetry mechanism.
- **Physical continuation:** the aligned fast-codec result must reduce bytes by
  at least 25% and its optimistic transformed bound must be at or below 47.7 ms.
- **Safety:** Gate 8 passes with no swap growth, throttling, service loss, or
  unexplained resident retention.

If any exactness or safety gate fails, reject the implementation. If the
canonicalization or physical bound fails, reject 128-neuron block symmetry as
the executable-byte mechanism on this selected real route. Do not escalate to
all 256 experts, individual-neuron scale expansion, a learned basis, or a
runtime decoder. If both pass, freeze a separate full-layer/holdout contract
before claiming generality or building an executable codec.

## Result

Not yet executed.
