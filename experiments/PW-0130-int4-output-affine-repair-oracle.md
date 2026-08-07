# PW-0130 — INT4 expert-output affine repair oracle

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0129 raw report
  `1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX 0.31.2 affine INT4
- Exactness: explicitly modified L3 INT4 plus expert-output repair oracle;
  unchanged source-FP8 control
- Related records: PW-0011 through PW-0019, PW-0116, PW-0129

## Question and causal mechanism

PW-0129 proves that naive affine INT4 has the desired 53.112% expert-byte
ratio but produces 4.19--15.46% validation routed-output error. Before paying
for mixed-precision blocks, GPTQ/AWQ-style calibration, or recovery training,
test whether the dominant complete-expert error is merely a systematic
per-output-channel distortion.

Grant each expert a diagonal output repair after its INT4 down projection:

`y_repaired[j] = f16(scale[e,j]) * y_int4[j] + f16(bias[e,j])`.

This transform remains inside the routed-layer transaction before source route
weighting and reduction. A full 256-expert layer needs 4,194,304 bytes of F16
scale/bias parameters, negligible relative to its source bank. It does not
repair input-conditioned cross-channel errors.

## Frozen oracle and authority

Authenticate PW-0116 and PW-0129 exactly. Recompute INT4 expert outputs for
positions `0..167` one expert at a time and require every packed artifact hash,
byte count, source route, source capture, train metric, and validation baseline
to reproduce PW-0129. Preserve positions `168..223` untouched.

Evaluate two nested same-validation capacity oracles at layers 4, 24, and 46:

1. **bias-only:** for every validation-touched expert and output channel, use
   the validation mean source-minus-INT4 error;
2. **affine:** solve the validation least-squares scale and bias independently
   for every expert/output channel, with constant candidate columns falling
   back to unit scale plus mean bias.

Quantize all fitted parameters to F16 before applying them. Experts absent from
validation retain identity repair. Fitting and evaluating on the same
validation rows is intentionally noncausal and cannot authorize deployment;
it is an upper-bound capacity test. Add deterministic fixtures for exact
single-placement bias repair, multi-placement least squares, constant columns,
F16 parameter staging, nested oracle ordering, routed reduction, invalid
shapes, and non-finite fits.

## Gates and dispositions

The affine oracle passes capacity only if all of the following hold on the
unchanged validation positions `112..167`:

1. aggregate routed-output relative L2 across all layers is at most 1%;
2. every layer is at most 2%;
3. no row exceeds 5% relative L2;
4. affine repair is no worse than bias-only, and bias-only is no worse than
   uncorrected INT4 at every layer;
5. F16 scale/bias are finite and add no more than 0.2% of source layer-bank
   bytes; and
6. every PW-0129 baseline and Gate 8 authority reproduces.

If this same-validation oracle fails, reject diagonal expert-output repair and
do not implement train-only calibration, a full repaired bank, or accumulated
execution. If it passes, freeze a separate train-only fit and evaluate
validation before unsealing holdout. This experiment never reads holdout and
reports zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.

Failure does not reject weight-domain activation-aware quantization,
mixed-precision outlier blocks, low-rank cross-channel repair, recovery
training, or structurally different representations. It rejects only a
diagonal affine correction applied after the complete INT4 expert.

