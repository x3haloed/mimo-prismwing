# PW-0145 — Train-only code-QAT optimizer preflight

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
  PW-0144 raw
  `8828db18f3d9471aa9abd2110b78994b86803ff393e4ec0d9fe81e10cef5d00c`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX full-batch straight-through optimizer and dense-F16 execution oracle
- Exactness: explicitly modified L3 training preflight; no validation or
  holdout target is loaded or scored
- Related records: PW-0139, PW-0142, PW-0144

## Question and scope

PW-0144 proves that its `0.05` Adam schedule can change millions of codes but
worsens even train. Before another representation or validation experiment,
determine whether the same fixed-grid straight-through parameterization has a
stable descending train regime.

Use only layer 46/expert 249 and its 90 routed positions below 112. Do not load
or score validation expert outputs or inputs. Reconstruct the exact PW-0139
initial grid and train metric, then run independent 32-step full-batch Adam
trials from the same zero-offset state at learning rates
`[0.0001, 0.0005, 0.001, 0.005]`. Keep betas `(0.9,0.999)`, epsilon `1e-8`,
bias correction enabled, and `1e-6` latent regularization unchanged. No trial
may warm-start another.

Record the complete train loss at steps 0/4/8/.../32, final source-output train
error, changed-code count/fraction, code domain, F16 metadata identity, latent
maximum, wall time, memory, and Gate 8 evidence. Select the lowest final train
relative L2, breaking exact ties toward the smaller learning rate. The
selection rule and candidate set are frozen before execution.

## Continuation gate

Authorize one separately frozen train/validation confirmation only if the
selected trial:

1. reduces PW-0139 train relative L2 by at least 25%;
2. ends with finite loss below its initial loss;
3. changes at least one but no more than 5% of codes;
4. keeps every code in `[0,15]` and F16 grid metadata bit-identical; and
5. retains the 13,369,344-byte (`0.531120`) zero-extra-MAC runtime ledger.

Failure rejects this fixed-grid straight-through optimizer family over the
tested train-only schedule envelope. A pass authorizes only a new experiment
with the selected learning rate frozen—no validation conclusion, holdout,
bank, kernel, accumulated model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after every trial and at least every eight
training steps.

