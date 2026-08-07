# PW-0132 — Train-only INT4 rank-32 repair generalization

- Status: completed
- Disposition: rejected; weight-domain calibration or mixed precision next
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0129 raw
  `1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
  PW-0131 raw
  `e0cf60d13b3e55fd805b480bf834baa55e87f7cf5de6b49623f722c094c0d876`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX affine INT4 plus F16 rank-32 repair factors
- Exactness: explicitly modified L3 train-fitted repair; unchanged source-FP8
  control
- Related records: PW-0116, PW-0129 through PW-0131

## Question and causal mechanism

PW-0131 proves same-validation capacity at rank 32 but grants direct access to
the targets being scored. Determine whether the same compact program learns a
rule rather than merely memorizing a residual matrix.

For each expert at layers 4, 24, and 46, fit only positions `0..111`:

1. fit F16 per-channel affine output repair on that expert's train placements;
2. compute the remaining train residual;
3. fit the exact PW-0131 rank-32 mapping from F16-staged real MoE input to that
   residual;
4. freeze all parameters; and
5. apply them without refitting to validation positions `112..167`.

Validation experts absent from train receive identity repair and remain visible
in metrics. Positions `168..223` remain sealed regardless of this result; a
validation pass authorizes a separate one-time holdout record.

## Frozen authority and implementation

Authenticate and reproduce PW-0129's INT4 packed artifacts and baseline
metrics exactly. Reuse PW-0131's F16 factor staging, rank-32 physical ledger,
least-squares convention, zero padding, and routed BF16 reduction. Add fixtures
for disjoint train/validation fitting, frozen-parameter application, unseen-
expert identity fallback, coverage accounting, no target leakage, and corrupted
partition boundaries.

Record per layer:

- train and validation baseline, affine-only, and affine-plus-rank-32 metrics;
- train/validation experts and placements covered by fitted parameters;
- parameter hashes and finite-value checks;
- rank-32 combined byte and MAC ratios from PW-0131; and
- fit wall, application wall, memory, and Gate 8 evidence.

Fit/application timings are component diagnostics, not endpoint TPS.

## Gates and dispositions

Pass generalization only if frozen validation reaches:

1. aggregate routed-output relative L2 at most 1%;
2. every layer at most 2%;
3. no row above 5%;
4. rank-32 is no worse than train-fitted affine-only, which is no worse than
   uncorrected INT4, at every layer;
5. all validation placements are covered by a train-fitted expert, with no
   identity fallback hiding errors; and
6. the unchanged physical envelope remains at most 60% source bytes and 5% of
   source expert MACs.

If the strict gate fails but aggregate error is at most 2%, every layer at most
4%, every row at most 8%, and coverage is complete, classify it as a near miss
that authorizes broader **training** corpus acquisition while keeping the
existing holdout sealed. Otherwise reject this pilot rank-32 train-only repair
and move to weight-domain calibration/mixed precision or another structure.

A pass does not authorize a full bank or accumulated model. It authorizes only
a new frozen holdout evaluation. Report zero accepted tokens, `A=0`, no
endpoint timing, and no TPS claim. Apply normative Gate 8 at every expert
release, layer fit/application boundary, corpus release, and final service
health readback.

## Result

The clean run completed in 68,146.898 ms and kept positions `168..223` sealed.
The frozen rank-32 repair failed decisively rather than narrowly:

| Layer | INT4 validation L2 | Affine validation L2 | Rank-32 train L2 | Rank-32 validation L2 | Worst validation row |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4.1919% | 5.5296% | 0.0519% | 17.3981% | 57.4205% |
| 24 | 11.9174% | 6.7939% | 0.3862% | 9.7538% | 33.4253% |
| 46 | 15.4606% | 10.2856% | 2.3368% | 9.2928% | 12.3385% |

Aggregate frozen-validation relative L2 is 15.0331%, the worst layer is
17.3981%, and the worst row is 57.4205%. The nested monotonic gate fails:
layer 4's train-fitted affine and rank-32 repairs both make validation worse
than uncorrected INT4. Coverage also fails because 15 of layer 24's 448
validation placements use one expert absent from training and therefore take
the declared identity fallback. This is not a threshold-edge or coverage-only
failure; even the completely covered layers miss the near-miss gate by large
margins.

Gate 8 passes across 216 snapshots at 78% minimum free memory,
731,004,928-byte maximum peak RSS, 224,037,568-byte maximum physical footprint,
zero swap growth or new throttled pages, and stable protected services. Raw
evidence hashes to
`0499a40645452eab646276e1619fb2e94b74439ef4263a71f036fae61fd8a9fe`;
independent analysis hashes to
`c098eb01547d211de5f3bf7fa545b599701616b8142c5689559bcda73e808557`.

## Decision

Reject this pilot train-only affine-plus-rank-32 activation-repair mechanism.
Do not read holdout, acquire a broader corpus to rescue it, build a repair
bank, or compose it with the endpoint. PW-0131's same-validation pass is now
classified as memorization capacity rather than evidence of a general repair
rule. Return to weight-domain calibration, outlier-aware mixed precision, or a
structurally different executable representation. No endpoint performance or
TPS claim changes.
