# PW-0135 — Group-local GPTQ three-expert control

- Status: completed
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0129 raw
  `1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
  PW-0134 raw
  `7d470bd5fa5541424c2b619afb49a2ebf493ce7a11b2498cf281b3d1c6f34490`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  block-local NumPy GPTQ assignment oracle and dense-F16 execution oracle
- Exactness: explicitly modified L3 weight-only quantization; unchanged
  source-FP8 and affine-INT4 controls
- Related records: PW-0116, PW-0129, PW-0133, PW-0134

## Question and scope

PW-0133's diagonal group selection and PW-0134's channel rescaling improve
affine INT4 but cannot approach the validation gate. Test whether correlated
second-order error propagation has enough local capacity before applying it to
every validation expert.

Use the deterministic highest-validation-coverage expert that also appears in
train at each representative layer: layer 4/expert 96, layer 24/expert 22, and
layer 46/expert 28. Each occupies all 56 validation positions and has 109, 26,
and 100 train placements respectively.

## Frozen group-local GPTQ mechanism

For each gate, up, and down projection:

1. retain MLX affine group-128 INT4's original per-row scale and bias;
2. form a 128-by-128 activation Hessian independently inside each quantization
   group from train positions `0..111` only;
3. add diagonal damping at `0.1%`, `1%`, and `10%` of mean Hessian diagonal;
4. test natural column order and descending Hessian-diagonal activation order;
5. quantize one column at a time to the fixed affine grid and propagate its
   normalized error through the remaining Cholesky inverse columns; and
6. select the damping/order pair minimizing that projection's train output
   squared error.

Gate/up use the real expert MoE inputs. Down uses source-derived BF16 SwiGLU
inputs. Compose the three independently selected assignments and score the
complete expert only on positions `112..167`; positions `168..223` remain
sealed.

The capacity oracle may execute unpacked F16 grid values, but must prove every
value belongs to the original affine grid and charge the unchanged packed
INT4 bytes. It may not report dense-oracle memory or timing as the executable
artifact or endpoint performance. Add deterministic fixtures for fixed-grid
quantization, error propagation, damping, activation-order permutation,
grid-membership validation, singular/zero activation groups, and partition
leakage.

## Continuation gate

Authorize an all-validation-expert GPTQ audit only if every one of the three
complete experts:

1. reduces validation relative L2 by at least 50% versus an unpacked affine
   INT4 control executed through the identical dense oracle;
2. reaches validation relative L2 at most 8%;
3. has no validation row above 12%;
4. improves rather than regresses on train; and
5. remains exactly representable by the original 13,369,344-byte affine INT4
   payload with no extra runtime MACs.

Also compare the unpacked control with PW-0129's MLX packed control so a dense
oracle arithmetic difference cannot masquerade as GPTQ gain. Failure rejects
this group-local fixed-grid GPTQ form, not global-Hessian GPTQ, rotations, or
recovery training. A pass authorizes only a new full-layer validation contract,
not holdout, a bank, kernel, accumulated model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 at every projection-candidate and expert release.

## Result

The frozen continuation gate fails by one criterion on one expert. All three
experts reduce validation relative L2 by at least 50% and improve train. Layer
4/expert 96 reaches `0.033047` validation relative L2 with a `0.059566`
maximum row; layer 24/expert 22 reaches `0.066439` with a `0.091837` maximum
row. Both pass.

Layer 46/expert 28 reduces validation error by 50.60%, from `0.163279` to
`0.080659`, and its `0.107637` maximum row passes. It exceeds the frozen
`0.080000` absolute validation ceiling by `0.000659`, so the all-expert gate
fails and the decision is `reject_group_local_fixed_grid_gptq`. The selected
setting is 0.1% damping with activation order for all nine projections.

The unpacked RTN oracle remains within `0.000913` relative L2 of the MLX
packed control, ruling out dense-oracle arithmetic as the gain. The candidate
retains the exact 13,369,344-byte INT4 physical charge (`0.531120` of source)
and adds no runtime MACs. Holdout remains sealed. Do not start the contracted
all-validation-expert audit. This rejects the fixed-grid, group-local form as
frozen, not global-Hessian GPTQ, function-preserving rotations, or recovery
training.

Gate 8 passes across 60 snapshots: minimum system memory free is 78%, maximum
peak RSS is 1,063,256,064 bytes, maximum physical footprint and release-boundary
footprint are 353,652,480 bytes, swap growth and new throttled pages are zero,
and protected service PID sets remain stable.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0135/run-001.json`, SHA-256
`56b9d38c3c630359b8d5b1a911627882df06a2e2fc374751fde2fddaeb3888db`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0135/analysis-001/manifest.json`,
SHA-256
`63565129c4f47cff5ab274b687a27bf9c64131ab83e86fab6b3ee4cb98a24bf6`.
No endpoint TPS or measured throughput constant changes.
