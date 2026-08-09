# PW-0144 — Train-only code-changing INT4 QAT

- Status: completed
- Disposition: rejected
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0139 raw
  `83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5`;
  PW-0142 raw
  `0c2095a2068ccf347ab86beccb41e8d303444ce371b52d8475a37b26c29e9cc7`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX full-batch straight-through optimizer and dense-F16 execution oracle
- Exactness: explicitly modified L3 train-fitted quantization; source routes,
  route weights, source-FP8 control, and validation targets remain unchanged
- Related records: PW-0129, PW-0137 through PW-0142

## Question and causal mechanism

PW-0142 rejects metadata-only recovery because its frozen scale/bias schedule
does not improve even train. Test the smallest materially different recovery
mechanism: keep PW-0139's group-128 affine grids fixed but permit their four-bit
assignments to change under end-to-end expert loss.

Use layer 4/expert 96, layer 24/expert 200, and layer 46/expert 249. Reconstruct
PW-0139's initial grids and codes exactly from routed positions `0..111`. Fit
only those routed train positions and score `112..167` without refitting.
Positions `168..223` remain sealed.

## Frozen QAT form

For every weight, initialize a dimensionless latent offset at zero around its
PW-0139 code. The executable forward code is:

```text
continuous_code = initial_code + latent_offset
integer_code = clamp(round(continuous_code), 0, 15)
weight = integer_code * fixed_scale + fixed_bias
```

Use the straight-through gradient of `continuous_code` for the rounded/clamped
forward value. Jointly optimize gate, up, and down latent offsets against
normalized complete-expert output MSE with full-batch Adam for 128 steps,
learning rate `0.05`, betas `(0.9, 0.999)`, epsilon `1e-8`, bias correction
enabled, and `1e-6` mean-square latent regularization. The fixed scales and
biases are staged through F16 before both training and final validation.

No learning-rate, step-count, optimizer, regularization, expert, loss, or
initialization search is allowed. Add fixtures for straight-through forward
identity, nonzero gradients, code-bound clamping, initial PW-0139 reproduction,
train/validation isolation, changed-code accounting, deterministic final code
hashes, and corrupt authority rejection. Record code-change counts by
projection, loss history, fit wall time, train/validation metrics, memory, and
Gate 8 evidence. F32 latent offsets and Adam state are training-only machinery.

## Continuation gate

Authorize only a separately frozen all-validation-expert QAT audit if:

1. layer 24/expert 200 and layer 46/expert 249 each improve PW-0139 validation
   relative L2 by at least 25% and reach at most 5%;
2. their maximum validation row is at most 8%;
3. layer 4/expert 96 does not regress by more than 10% from PW-0139;
4. every expert improves its train output, has finite decreasing loss, and
   changes at least one but not all codes;
5. every final assignment is an integer in `[0, 15]`, scales/biases remain
   bit-identical to the F16-staged initial grids, and the source controls
   reproduce; and
6. the artifact remains 13,369,344 bytes per expert (`0.531120` of source)
   with zero additional runtime MACs.

Failure rejects this fixed-grid, code-changing, straight-through schedule. It
does not reject grid-changing QAT, broader recovery training, distillation, or
a different executable representation. A pass authorizes only a full
validation audit—not holdout, a bank, kernel, accumulated model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before training, during training at least every 16
steps, and after every expert release.

## Result

The frozen straight-through schedule fails its train prerequisite. Layer
4/expert 96 receives no effective latent update or code change and remains
exactly at `0.005445` train and `0.022155` validation relative L2. Layer
24/expert 200 changes 5,398,042 of 25,165,824 codes (21.45%) but worsens from
`0.030712` to `0.158823` on train and from `0.065851` to `0.684947` on
validation. Layer 46/expert 249 changes 5,870,177 codes (23.33%) and worsens
from `0.039279` to `0.170402` on train and from `0.067440` to `0.444887` on
validation. Their worst validation rows reach `0.817322` and `0.613438`.

All PW-0139 controls reproduce before training. Every final code remains in
`[0,15]`, fixed F16 scale/bias metadata remains bit-identical, and the
partitions and sealed holdout remain intact. The branch therefore rejects this
learning-rate/straight-through/fixed-grid schedule rather than exposing an
authority or artifact defect. Because the permitted train objective itself
worsens, do not tune this schedule on visible validation or expand it to all
experts. Grid-changing QAT, broader recovery training, distillation, and other
executable representations remain distinct.

The runtime ledger remains 13,369,344 bytes per expert (`0.531120` of source)
with zero additional MACs; latent offsets and Adam state are training-only.
Gate 8 passes across 51 snapshots: minimum free memory is 58%, maximum peak RSS
is 1,632,075,776 bytes, maximum physical footprint is 1,430,818,176 bytes,
maximum release-boundary footprint is 285,494,144 bytes, swap growth and new
throttled pages are zero, and protected services remain resident.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0144/run-001.json`, SHA-256
`8828db18f3d9471aa9abd2110b78994b86803ff393e4ec0d9fe81e10cef5d00c`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0144/analysis-001/manifest.json`,
SHA-256
`93871a88c85a883ced215a233e25791b1f39861a86332e60fd1e2a9cfc64db28`.
No endpoint TPS or measured throughput-model constant changes.
