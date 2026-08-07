# PW-0121 — Rank-768 activation-weighted expert pilot

- Status: complete
- Disposition: conditional
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0119 analysis
  `166f56b0b56c82099520acd6696647d8bc350b52d5b33d8649d51a7971cf7a34`;
  PW-0120 analysis
  `4fce122f9887f7c103c635337c235767fe66de63372c80219f8b745a191c4a50`
- Hardware/runtime: Apple M1 shared 16 GiB; NumPy Accelerate SVD, PyTorch
  2.13 MPS optimization and CPU source-FP8 evaluation; internal SSD
- Exactness: L4 fitted diagnostic; unchanged source control
- Related records: PW-0045, PW-0116 through PW-0120; E5

## Question and mechanism

PW-0119 finds that independent matrix-SVD rank 768 leaves 70.98% routed-output
relative L2 on layer-24 hot expert 23, while PW-0120 rejects only the full-bank
Adam allocation topology. Test the narrower causal question: can the same
rank-768 per-projection capacity fit the routed activation distribution
materially better than global matrix Frobenius SVD when optimized one
projection at a time?

This is an independent per-expert control, not yet a shared-basis fit. It is
strictly cheaper and stronger as a fidelity target: if activation-weighted
rank 768 cannot improve one well-covered middle-layer expert, a shared rank-768
basis has no reason to be trained. If it can, the result establishes the
matched control a later shared candidate must approach.

## Frozen data and parameterization

Use only layer 24 expert 23 from PW-0116: 65 train placements at positions
`0..111`, 46 validation placements at `112..167`, and 56 untouched pilot-
holdout placements at `168..223`. Authenticate every schedule, capture, and
source tensor. Derive source gate, up, BF16 SwiGLU hidden, and down targets with
the already bit-exact PW-0119 PyTorch source-FP8 oracle.

For gate, up, and canonical transposed down independently, initialize balanced
rank-768 SVD factors `L = U[:, :768] * sqrt(S[:768])` and
`R = sqrt(S[:768]) * Vt[:768, :]`, so the candidate matrix is `L @ R` while
neither factor absorbs the entire singular scale. Optimize only one
projection's `L` and `R` at a time on MPS; inactive projections remain on CPU
and carry no gradient or optimizer state. Each active projection has 4,718,592
F32 values (18,874,368 bytes), with a 75,497,472-byte
parameter/gradient/Adam semantic live set.

Train gate/up against their source BF16 projection outputs using frozen
source-quantized `moe_input`. Train down against source BF16 down outputs using
frozen source-quantized source SwiGLU hidden. The differentiable training
surrogate is F32 factor multiplication and normalized MSE. Evaluation restores
the true dynamic-FP8 input quantization, BF16 projection boundaries, BF16 SiLU
and product, and the complete three-projection expert composition. Thus the
optimizer need not differentiate through FP8 casting, but no evaluation gate
is computed on the surrogate.

## Frozen optimizer and model selection

Use seed `260121`, Adam learning rate `0.001`, at most 100 full-train steps per
projection, and validation every five steps including step zero. Select each
projection checkpoint by minimum validation normalized MSE; the pilot holdout
must not be evaluated until all three projection checkpoints are frozen. Stop
early after four consecutive validation checks without improvement. Preserve
step/validation losses, selected step, factor hashes, source target hashes, and
phase wall times. Do not persist the approximately 226 MB fitted factors after
their hashes and final metrics are written.

Add a deterministic tiny CPU fixture proving SVD orientation, normalized loss,
one-projection update, checkpoint selection, and the complete candidate expert
equation. Apply Gate 8 before and after each SVD, parameter migration, backward,
optimizer step group, checkpoint copy, projection release, full expert
evaluation, and final service health. Set the MPS memory fraction to 0.60 and
record current/driver memory at every phase.

## Gates and interpretation

1. The source oracle must remain bit-exact to PW-0116 expert-down. The
   authoritative three-factor `U,S,Vt` control must reproduce PW-0119 rank-768
   metrics within `1e-6`. Separately report the balanced two-factor
   initialization and require its relative-L2 metrics to remain within `5e-6`
   of that authority; acceptance thresholds remain bound to PW-0119, not the
   re-associated initialization arithmetic.
2. Every training and validation value must be finite. The selected validation
   normalized MSE must be below step zero for all three projections; selection
   uses no holdout result.
3. After selection, the complete candidate expert must reduce validation
   relative L2 by at least 25% from PW-0119's `0.7103805967306607` (candidate
   at most `0.5327854475479955`) and untouched holdout relative L2 by at least
   25% from `0.6849577905886747` (candidate at most
   `0.5137183429415060`). Report train and overall metrics without using them
   as promotion gates.
4. Gate 8 retains the 20% free, 8-GiB live, 4-GiB release, 512-MiB swap-growth,
   zero-throttling, and protected-service requirements. MPS current allocation
   must return to zero after every projection and at final release.
5. Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS.

Passing authorizes the same experiment on layer-46 hot expert 28 and then a
shared-basis pilot. It does not authorize fitting all experts, persisting an L4
bank, building a kernel, or changing the runtime. Failure kills this
activation-weighted rank-768 factor fit as the current identity-basis fidelity
path; do not respond by tuning on the pilot holdout or increasing rank without
a new contract.

## Result

The first implementation attempt at
`854d78f2f649faa37c40f85ec2a405005ad80eee` stopped after fitting but before
candidate evaluation or evidence emission. No candidate holdout metric was
computed or observed. Balanced `sqrt(S)` factor materialization changed
the overall BF16 expert-output relative L2 from `0.709771747` to `0.709773524`,
a `1.777e-6` association delta that missed the original `1e-6` control check.
This is not representation evidence. The measurement repair retains PW-0119's
exact `U,S,Vt` arithmetic as the acceptance authority and adds a separate
`5e-6` bound for the balanced optimizer initialization. No quality threshold
or holdout result changed.

The repaired clean implementation at
`45111335cabed14bf49a14ea80a2deff51c81286` passed in 24,153.190 ms. The
source oracle remained bit-exact and the authoritative SVD control reproduced
PW-0119 exactly. Balanced initialization differed by only `1.777e-6` overall
relative L2 and passed its separate association bound.

All projection validation objectives improved under selection that never read
the holdout:

| Projection | Initial validation NMSE | Selected NMSE | Reduction | Step |
| --- | ---: | ---: | ---: | ---: |
| gate | 0.112674 | 0.011874 | 89.46% | 50 |
| up | 0.257866 | 0.021014 | 91.85% | 100 |
| down | 0.227509 | 0.033148 | 85.43% | 60 |

After freezing all three selections, the complete source-FP8/BF16 expert
comparison was:

| Partition | SVD relative L2 | Activation-weighted relative L2 | Reduction |
| --- | ---: | ---: | ---: |
| train | 0.731278 | 0.108358 | 85.18% |
| validation | 0.710381 | 0.251869 | 64.54% |
| pilot holdout | 0.684958 | 0.378045 | 44.81% |
| overall | 0.709772 | 0.267000 | 62.38% |

Both frozen 25% continuation gates pass. The result is not close to source
identity, but it decisively shows that routed-activation fitting is much more
informative than global matrix SVD at the same rank on this expert.

Gate 8 passed with 68% minimum free memory, 1,301,970,944-byte peak RSS,
1,684,082,816-byte maximum physical footprint, zero swap growth/throttling,
stable services, zero MPS current allocation after every projection, and a
307,990,080-byte final footprint. The raw report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0121/run-001.json` hashes to
`04388f2704607657fecd5304d2533585e7ee6389080f3e77e5658a9875da05fb`.
Independent analysis at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0121/analysis-001/manifest.json`
hashes to
`6f3c7e8d9ddd25db65dc35cb888a98349bfa89b538cc33be7a2e0ffe5e3c6d17`.
There are zero accepted tokens and no TPS claim.

## Decision

Authorize the identical activation-weighted rank-768 pilot on layer-46 hot
expert 28. Preserve its 100 train, 56 validation, and 56 holdout rows and set
its continuation threshold against PW-0119 before execution.

Do not train shared bases yet. A late-layer replication is required because
PW-0119 showed substantial depth dependence. Even a second pass would
authorize only a small shared-basis pilot; this corpus is English, sequential,
and sparse in expert coverage, and the selected factors were deliberately not
persisted as a runtime artifact.
