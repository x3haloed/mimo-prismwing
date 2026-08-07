# PW-0132 — Train-only INT4 rank-32 repair generalization

- Status: planned
- Disposition: unexecuted
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

