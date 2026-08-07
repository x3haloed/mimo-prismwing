# PW-0131 — INT4 input-conditioned low-rank repair oracle

- Status: completed
- Disposition: capacity pass; train-only validation authorized
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0129 raw
  `1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
  PW-0130 raw
  `b011bd5ced8787df62f4380aeeccab9a35aef8b8ab15541207bcd99e35727994`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX affine INT4 plus analytical/F16 repair factors
- Exactness: explicitly modified L3 input-conditioned repair oracle;
  unchanged source-FP8 control
- Related records: PW-0045, PW-0116, PW-0129, PW-0130

## Question and changed premise

PW-0130 shows that static diagonal output repair removes most INT4 error but
cannot reach the frozen gate even when fitted and evaluated on validation.
The remaining error is input-dependent and/or cross-channel. Test the smallest
such executable family before changing quantization or training weights:

`y = affine_repair(y_int4) + (f16(x) @ A_e) @ B_e`,

where `x` is the expert's real 4,096-wide MoE input and `A_e [4096,r]`,
`B_e [r,4096]` are expert-specific F16 factors. The correction stays inside
the routed-layer transaction before source route weighting.

## Frozen same-validation capacity oracle

Authenticate and exactly reproduce PW-0129's INT4 outputs and PW-0130's
same-validation affine repair at layers 4, 24, and 46. Preserve positions
`168..223` sealed.

For ranks `8, 16, 32, 56`, fit and evaluate on the same validation rows:

1. compute each expert's residual after its PW-0130 F16 affine repair;
2. take the best rank-`r` SVD approximation of that validation residual
   matrix;
3. solve least squares from the expert's F16-staged real input rows to the
   residual's left coordinates;
4. store both factors as F16, execute the two factor matmuls, add the repair,
   route-weight and BF16-reduce exactly as the layer does; and
5. account a fixed full-bank rank for all 256 experts, even when a validation
   expert has fewer rows and its fitted factors are zero-padded.

This is deliberately noncausal and can memorize validation. It is only a
capacity and physical-fitness oracle. Add deterministic fixtures for
rank-limited residual reconstruction, F16 factor staging, underdetermined
least squares, zero padding, rank monotonicity, routed reduction, invalid
shapes, and non-finite factors.

## Physical envelope and gates

For each rank, report:

- full-layer factor bytes `256 * 2 * 4096 * r * 2`;
- combined INT4 + PW-0130 affine + low-rank bytes relative to the source bank;
- repair MACs per eight-expert mixture `8 * 2 * 4096 * r`;
- repair/source-expert MAC ratio; and
- complete validation routed-output metrics.

The branch passes capacity only if at least one rank satisfies all conditions:

1. aggregate routed-output relative L2 at most 1%;
2. every layer at most 2%;
3. no row above 5%;
4. errors are monotonically non-increasing with rank at every layer;
5. combined executable bytes at most 60% of the source routed bank; and
6. repair MACs at most 5% of source selected-expert MACs per mixture.

Select the smallest passing rank without reading holdout. A pass authorizes a
separately frozen train-only fit evaluated on validation; it does not authorize
a bank, accumulated model, recovery training, or endpoint. If rank 56 fails
despite same-validation fitting, reject this two-factor input-conditioned
output repair family and move to weight-domain mixed precision/calibration or
a structurally different program.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 at source authentication, every expert release, every
rank fit, layer release, checkpoint release, and final service-health readback.

## Result

The clean run completed in 69,259.843 ms. It reproduced PW-0129's packed INT4
artifacts and PW-0130's affine parameter hashes and validation metrics exactly.
All rank errors are monotonic at every layer, and the holdout remained sealed.

| Rank | Aggregate relative L2 | Worst layer | Worst row | Combined source-byte ratio | Repair/source MAC ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.9075% | 3.1675% | 3.9249% | 53.6978% | 0.2604% |
| 16 | 1.5148% | 2.5183% | 3.2738% | 54.2185% | 0.5208% |
| 32 | 0.9493% | 1.5825% | 2.1885% | 55.2599% | 1.0417% |
| 56 | 0.0485% | 0.0572% | 0.0911% | 56.8221% | 1.8229% |

Rank 32 is the smallest passing capacity. It uses 134,217,728 low-rank-factor
bytes per complete layer in addition to the INT4 and affine artifacts. Its
complete representation is 55.2599% of the source layer bank and its repair
adds 2,097,152 MACs per eight-expert mixture, 1.0417% of source expert MACs.
The same-validation numerical result passes every frozen gate.

This is a capacity result, not generalization. The fit sees the validation
targets it scores, and rank 56's near-perfect result illustrates the available
memorization capacity. A train-only rank-32 fit must now reproduce the gain on
validation before holdout can be considered.

Gate 8 passes at 78% minimum free memory, 763,248,640-byte maximum peak RSS,
221,907,904-byte maximum physical footprint, zero swap growth or new
throttled pages, and stable protected services. Raw evidence hashes to
`e0cf60d13b3e55fd805b480bf834baa55e87f7cf5de6b49623f722c094c0d876`;
independent analysis hashes to
`754285ca807cde425f5742dfb3ffc1014d2a99be9cf7f188eb16307fb3f90042`.

## Decision

Authorize a separately frozen train-only rank-32 affine-plus-low-rank repair
fit, evaluated on validation with holdout sealed. Do not build a bank, execute
an accumulated model, or claim endpoint fidelity or TPS. The result proves
only that an input-conditioned cross-channel repair family has sufficient
same-slice capacity inside the initial byte and compute envelope. Train-only
failure would reject generalization from this pilot; success would authorize
one holdout read and then broader corpus acquisition.
