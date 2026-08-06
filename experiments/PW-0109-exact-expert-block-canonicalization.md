# PW-0109 — Exact expert block canonicalization

- Status: completed
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: `e3d47f1cdbc866cf70a056ba0dfe87b643ee4e82`
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

The implementation at
`f91121f0a3491ab41733a1e3dddc6f82e18538ee` extracted 16 exact
1,573,248-byte neuron-block records from each of the eight authenticated
experts. The 201,375,744-byte ledger closes exactly. Every expert assignment is
a bijection, XOR reversal passes, and inverse block scatter reproduces all 48
source tensor hashes. Five focused Python tests cover exact XOR, component
lengths, tiny gate/up row movement, down column/scale movement, nonaliasing,
bounds, and inverse-count failures.

Expert 9 is the deterministic reference. Every other expert selects a
nonidentity permutation, but exact assigned XOR popcounts remain approximately
93.0--93.7 million bits per expert. That apparent assignment activity does not
translate into compressible shared structure. All three compared streams have
the same 201,375,744 logical bytes:

| Stream | zstd 1 bytes / ratio | zstd 19 bytes / ratio |
| --- | ---: | ---: |
| Unmodified expert-major | 177,024,596 / 87.908% | 176,833,045 / 87.812% |
| Identity reference/XOR | 191,564,246 / 95.128% | 191,719,443 / 95.205% |
| Aligned reference/XOR | 191,481,312 / 95.087% | 191,648,055 / 95.169% |

At the fast setting, alignment improves identity-delta by only 0.0433%, far
below the 10% mechanism gate, and is 8.167% larger than the unmodified control.
It reduces source bytes by only 4.913%, far below the 25% physical gate.
Single-thread zstd decompression to `/dev/null` takes 190.621 ms, yielding an
optimistic transformed acquisition-plus-decode bound of 245.804 ms rather than
47.7 ms. High analysis compression is no better and is not a runtime proposal.

Gate 8 passes with 78% minimum free memory, 508,133,376-byte peak RSS,
82,200,640-byte final physical footprint, zero swap growth, zero new throttled
pages, and stable protected services. Temporary 604,127,232-byte streams were
released and never entered Git. The immutable run manifest at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0109/run-001/manifest.json`
hashes to
`9e0f15f65269d1b5c53536f18cda62df039d13ed19f48242f3eef91966b43bab`.
The clean analyzer at
`4dd03eae8c01715b3e2f373354d7b6ac82b214f7` emitted `analysis-001.json`, hash
`a298ae0b3022fa5f22e06a573af9d1bfdc9471eb33c8fafcd7e664cf26d0b12d`.
The updated throughput model hashes to
`0b8d0db57e0b4869517c620c2292433eb0508ec03bfdc10901e2d9500a44a38f`.

## Decision

Reject 128-neuron block canonicalization as an executable-byte mechanism for
this selected real route. Do not expand it to all 256 experts, build a runtime
decoder, or infer that arbitrary-neuron canonicalization will succeed: that
deeper representation would expand or replace source scale topology and needs
independent evidence. The unmodified codec control shows a modest generic
12.1% high-level compression opportunity, but its measured CPU decode cost and
insufficient byte reduction fail the physical bound. No endpoint, artifact
default, or L3 arithmetic change is authorized.
