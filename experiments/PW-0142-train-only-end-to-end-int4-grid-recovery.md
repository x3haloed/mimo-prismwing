# PW-0142 — Train-only end-to-end INT4 grid recovery

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0139 raw
  `83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5`;
  PW-0141 raw
  `4dae2abe2a59457a77e09bd4d1328b7b6dce8f0e41e3ac115fd27645c93e56a9`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX full-batch recovery optimizer and dense-F16 execution oracle
- Exactness: explicitly modified L3 train-fitted quantization; source routes,
  route weights, source-FP8 control, and validation targets remain unchanged
- Related records: PW-0129, PW-0132, PW-0137 through PW-0141

## Question and causal mechanism

PW-0139 shows that projection-local global-Hessian assignment improves every
calibration projection but does not preserve the complete routed expert well
enough at deeper layers. PW-0141 then rejects one fixed rotation as the missing
weight geometry. Test the cheapest genuinely different recovery-training
mechanism: optimize the existing group-128 affine grids jointly through the
complete SwiGLU expert while preserving their four-bit codes.

Use the same well-covered experts as PW-0141: layer 4/expert 96, layer
24/expert 200, and layer 46/expert 249. Reconstruct PW-0139's global-Hessian
codes exactly from positions `0..111`. Fit only those experts' routed train
positions and score positions `112..167` without refitting. Positions
`168..223` remain sealed.

## Frozen training form

For every gate/up/down row-group, retain the fixed PW-0139 four-bit code and
parameterize its executable grid as:

```text
scale = initial_scale * exp(log_scale_multiplier)
bias  = initial_bias + initial_scale * bias_delta
weight = code * scale + bias
```

Initialize both trainable dimensionless parameters to zero. Optimize all three
projections jointly against normalized complete-expert output MSE using
full-batch Adam for 128 steps, learning rate `0.01`, betas `(0.9, 0.999)`,
epsilon `1e-8`, and `1e-4` mean-square regularization on both parameter sets.
No learning-rate, step-count, seed, loss, expert, or initialization search is
allowed. Stage the final scale and bias arrays through F16 before validation,
matching the prospective packed artifact.

Add deterministic fixtures for grid reconstruction, zero-delta identity,
gradient movement, fixed-code preservation, train/validation disjointness,
F16 final staging, and corrupted authority rejection. Record the train loss
curve, parameter hashes, parameter displacement, fit wall time, validation
metrics, memory, and Gate 8 evidence. Optimizer state and F32 master parameters
are training machinery, not runtime artifacts.

## Continuation gate

Authorize only a separately frozen all-validation-expert recovery audit if:

1. layer 24/expert 200 and layer 46/expert 249 each improve their reproduced
   PW-0139 validation relative L2 by at least 25% and reach at most 5%;
2. their maximum validation row is at most 8%;
3. layer 4/expert 96 does not regress by more than 10% from PW-0139;
4. every trained expert improves its train output from the reproduced PW-0139
   initialization and has a finite, decreasing loss;
5. all four-bit codes reproduce PW-0139 and remain unchanged; and
6. final F16 scales and biases keep the packed payload at 13,369,344 bytes per
   expert (`0.531120` of source) with zero additional runtime MACs.

Failure rejects this fixed-code, group-parameter, end-to-end recovery form. It
does not reject code-changing QAT, broader recovery training, learned
rotations, or a different executable representation. A pass authorizes only a
full-validation audit, not holdout, a bank, packed kernel, accumulated model,
or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before training, during training at least every 16
steps, and after every expert release.

