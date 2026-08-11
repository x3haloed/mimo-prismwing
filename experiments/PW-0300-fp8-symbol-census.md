# PW-0300 — FP8 symbol census and executable-subset preflight

- Status: completed
- Disposition: conditional
- Date: 2026-08-11
- Owner: Thimble with project-owner authorization
- Contract commit: `f677893`
- Checkpoint: XiaomiMiMo/MiMo-V2.5 revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Hardware/runtime: resident Linux host; sequential pinned HTTP range reads;
  Python standard-library analysis
- Exactness: L1 for palette/escape simulations; L3 for 6-bit FP8-subset
  simulations
- Related records: PW-0108, PW-0109, PW-0113, PW-0147, PW-0148

## Question and bounded acquisition

The measured M1 path spends approximately 58 ms acquiring a selected layer's
source FP8 expert bytes and approximately 8 ms executing them. Test whether the
actual learned FP8 symbols admit an executable representation materially below
one byte per weight before downloading complete 34-GiB shards or implementing
a decoder.

Use pinned safetensors headers and sequential HTTP range reads only. Sample one
128-row tile from every gate, up, and down projection for four depth/expert
points plus three layer-24 expert controls. A row tile contains 16 or 32 exact
128-by-128 source quantization blocks and costs only 256 or 512 KiB. Acquire a
second disjoint row tile for every tensor as holdout. Total network traffic is
20,828,528 bytes including the shard header in both passes. No complete tensor,
expert, or shard is downloaded.

The discovery run preceded this permanent record during a live, owner-requested
cheap-test investigation. This record consolidates that evidence; it does not
retroactively turn the simulations into routed-output or endpoint evidence.

## Measurements

Across 21 training tiles and 21 disjoint holdout tiles, each pass contains 560
quantization blocks and 9,175,040 sampled FP8 weight bytes.

For the training pass:

- every block contains 182–239 distinct FP8 byte values;
- zero of 560 blocks fit an exact 6-bit local palette;
- zero of 560 blocks fit an exact 7-bit local palette;
- block entropy ranges from 6.4121 to 7.0994 bits/weight, with a 6.4844-bit
  median;
- a top-seven exponent code plus exact exponent escapes occupies an idealized
  88.495–93.359% of source bytes, with an 88.959% median, before independent
  chunk offsets or alignment; and
- sign and mantissa are effectively fully populated while most statistical
  redundancy resides in exponent frequency, not a small exact exponent
  alphabet.

A greedy global 64-value codebook drawn only from legal E4M3FN values gives
sampled weight relative L2 of 1.479% median, 2.936% p95, and 6.929% maximum.
Per-tensor codebooks fit on one row tile and scored on the disjoint tile give
1.502% median, 2.577% p95, and 4.016% maximum. Representation-local codebooks
fit separately to each holdout row tile give 1.479% median, 2.476% p95, and
3.952% maximum.

A 64-byte codebook per row tile plus packed 6-bit indices occupies 75.0122% of
gate/up source bytes or 75.0244% of down source bytes. Existing FP8 block-scale
bytes remain required. These are weight-only errors and arithmetic ledgers;
they do not establish routed expert output, route stability, accumulated
logits, cold decode speed, or endpoint behavior.

## Evidence

Evidence is stored outside Git under
`/srv/residents/thimble/root/evidence/mimo-prismwing/PW-0300/`:

- `train-census.json` —
  `abc2ef97a15a286c8ddd71dac0773c5d43eb0e308b8791b9d87b4fee74835c37`
- `holdout-census.json` —
  `4229b4f893cba004035956f89b92c390502f7464c4c0c3be29799395ad8727a6`
- `analysis.json` —
  `dd603e2fbd03cdf6600d7a81f716becd2ea809eeaf3b0b46d53a71f613258552`

The committed tools fail closed on checkpoint identity, tensor dtype/layout,
row-block bounds, malformed symbol counts, and observed E4M3FN NaN codes.

## Decision

Reject exact local FP8 palettes and the tested top-K/exponent escape forms as
the required 25% executable-byte reduction. The observed entropy floor means a
typical lossless block cannot reach 75% even with an ideal entropy coder, and
the practical exact escape simulation remains near the earlier zstd result.
Do not build an exact Metal decoder for these forms.

Retain one conditional L3 candidate: 6-bit indices into a representation-local
64-value FP8 subset. It reaches almost exactly 75% physical bytes and preserves
the source's nonuniform FP8 value geometry, but its sampled weight-error tail
is too large to authorize a bank or kernel. The next experiment, if pursued,
must download only named representative full experts and compare routed expert
outputs against immutable source-FP8 activations. It must beat the existing
PW-0148 six-bit affine control under a frozen fidelity gate before any decoder
or broader download.
