# PW-0212 — Full-expert FP8-subset six-bit control

- Status: complete
- Disposition: rejected
- Date: 2026-08-11
- Owner: Thimble with project-owner authorization
- Contract commit: uncommitted pre-execution record
- Checkpoint: XiaomiMiMo/MiMo-V2.5 revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Exactness: L3 modified weight representation
- Related records: PW-0129, PW-0138, PW-0148, PW-0211

## Question

PW-0211 rejects the proposed exact palettes but leaves a 6-bit FP8-subset
representation at almost exactly 75% of source weight bytes. Determine whether
its sampled reconstruction results survive complete representative experts and
whether it improves on a same-weight affine six-bit round-to-nearest control
before acquiring routed activations or implementing a decoder.

## Bounded acquisition

Use the three experts already selected as early/middle/deep controls by
PW-0138/PW-0148:

- layer 4 / expert 96;
- layer 24 / expert 22; and
- layer 46 / expert 28.

Read all source FP8 weight bytes and source F32 scale bytes for those experts
through ordered pinned HTTP ranges with one active request by default and never
more than two. Do not download complete shards. The expected source payload is
approximately 75.5 MB for all three experts. Preserve tensor identities,
offsets, hashes, requested bytes, and total network bytes.

## Controls and measurements

For every 128-row gate/up/down tile:

1. Fit a deterministic 64-value codebook constrained to legal source E4M3FN
   values and pack indices conceptually at six bits.
2. Reconstruct through the retained source block scales.
3. Compare weight relative L2 by tensor, tile, and source quantization block.
4. Build a same-group six-bit affine round-to-nearest control with its complete
   code, scale, and bias byte ledger.
5. Report candidate/control error ratios and tails; do not substitute
   parameter-space error for routed-output evidence.

Fail closed on unknown tensor layout, missing scale topology, NaN codes,
checkpoint drift, overlapping/missing ranges, or byte-ledger mismatch.

## Continuation gate

Authorize a separately frozen routed-output experiment only if the FP8-subset
candidate:

- is no larger than 75.1% of source weight bytes including codebooks and index
  packing, with source scale bytes charged separately and unchanged;
- improves complete-expert weight relative L2 over affine six-bit
  round-to-nearest for all nine projection tensors;
- has no quantization block above 5% relative L2 and no complete projection
  above 2%; and
- completes within the declared network and memory bounds.

Failure rejects this codebook granularity and fitting rule. A pass authorizes
only acquisition of the existing routed-activation corpus and an immutable
source-output comparison. It does not authorize a packed bank, Metal decoder,
holdout access, endpoint claim, or performance promotion.

## Execution

The census read 192 complete 128-row tiles and 4,608 source quantization
blocks through ordered pinned HTTP ranges. Total source weight payload was
75,497,472 bytes. Including Safetensors headers and the small paired source
scale reads, the authoritative single-run ledger was 77,799,800 network bytes.
Its process returned without a visible completion line, so the nine projections
were conservatively reacquired one at a time for 82,367,592 additional bytes.
Keyed comparison confirmed every sample payload byte-for-byte identical. Thus
the actual investigation transferred 160,167,392 bytes; the evidence retains
the lower single-run ledger rather than hiding or charging duplicated operator
verification as model payload. Only one range request was active at a time.

Each tile received its own deterministic greedy 64-entry source-E4M3FN
codebook. Reconstruction errors were accumulated after applying every source
block's recorded `weight_scale_inv` value. The affine control used independent
128-value row groups, six-bit round-to-nearest codes, and one F16 scale and
bias per group.

| expert | projection | subset relative L2 | affine6 relative L2 | worst subset block | subset bytes |
|---|---:|---:|---:|---:|---:|
| L4 E96 | gate | 4.0199% | 2.5978% | 5.7293% | 75.0122% |
| L4 E96 | up | 3.8445% | 2.4521% | 5.4436% | 75.0122% |
| L4 E96 | down | 2.3970% | 2.9270% | 3.5659% | 75.0244% |
| L24 E22 | gate | 1.5388% | 2.3873% | 2.9630% | 75.0122% |
| L24 E22 | up | 1.4956% | 2.3961% | 2.3015% | 75.0122% |
| L24 E22 | down | 1.6033% | 2.4201% | 7.6234% | 75.0244% |
| L46 E28 | gate | 2.7355% | 2.5263% | 7.1681% | 75.0122% |
| L46 E28 | up | 2.2699% | 2.4452% | 6.6783% | 75.0122% |
| L46 E28 | down | 2.4165% | 2.6198% | 6.8382% | 75.0244% |

Complete-expert subset/control relative L2 was 3.5055%/2.6607% for L4 E96,
1.5459%/2.4012% for L24 E22, and 2.4653%/2.5240% for L46 E28. The affine
control occupies 78.125% of weight bytes under the declared scale/bias ledger.

Evidence is preserved outside Git at:

- `evidence/mimo-prismwing/PW-0212/full-expert-census.json`, SHA-256
  `1c7530c51b763a6e45510533c281a9a2c1506014b1fb45ce5c701a495a61bb03`;
- `evidence/mimo-prismwing/PW-0212/analysis.json`, SHA-256
  `57011d3e4e5c6d09a70ed7c4bf988b516e97f749e4bdf698ff1c556461017282`.

## Decision

Reject this tile-local 64-value subset and greedy fitting rule. It satisfies
the physical-byte gate but improves on affine six-bit RTN in only six of nine
projections, exceeds 2% projection error in six projections, and exceeds the
5% block ceiling in six projections. The failure is strongly nonuniform: the
middle expert benefits while the early expert's gate and up projections lose
badly. Do not acquire routed activations or build a decoder for this branch.

This rejection is deliberately narrow. It does not test covariance-trained
row transforms, unbiased inner-product sketches, activation-conditioned
representations, or whole-expert reparameterization. Those require separate
contracts and equal-byte/equal-MAC controls rather than reinterpretation of
this failed gate.
