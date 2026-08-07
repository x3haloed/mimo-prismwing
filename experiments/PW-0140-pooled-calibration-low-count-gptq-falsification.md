# PW-0140 — Pooled-calibration low-count GPTQ falsification

- Status: planned
- Disposition: pending
- Date: 2026-08-07
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0139 raw
  `83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5`;
  PW-0139 analysis
  `9aecfdcd32e535b4b9d27fcac075dfd1c9014080d624aa3f4af2c678be3f3b6c`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  NumPy global-Hessian assignment oracle and dense-F16 execution oracle
- Exactness: explicitly modified L3 weight-only quantization; unchanged
  source-FP8 and affine-INT4 controls
- Related records: PW-0135, PW-0137, PW-0138, PW-0139

## Question and scope

PW-0139's worst consequential expert errors include layer 24/expert 39 (6
routed train placements, `0.124820` validation L2), layer 24/expert 128 (19,
`0.108000`), and layer 46/expert 140 (10, `0.100973`). Test whether using the
full layer train-input distribution, rather than each sparse routed subset,
removes this failure mode.

This is a validation-visible capacity falsification on three already-unsealed
experts. It cannot select or promote a final policy. Positions `168..223`
remain sealed.

## Frozen mechanism

For each expert, calibrate gate/up on all layer MoE inputs at positions
`0..111`. Compute that expert's source BF16 gate/up/SwiGLU values on the same
112 inputs and calibrate down on those hidden values. Apply PW-0139's
global-Hessian mechanism unchanged: original affine group-128 grids, 0.1%
damping, descending activation order, static original-group lookup, and
128-column cross-block propagation. Do not tune or blend pooled and routed
Hessians.

Compare against each expert's immutable PW-0139 routed-calibration candidate
on its validation placements. Prove grid membership, projection calibration
improvement over pooled round-to-nearest controls, exact sample identities,
and the unchanged physical ledger.

## Continuation gate

Authorize only a separately frozen train-only shrinkage-policy experiment if
all three pooled candidates:

1. improve validation relative L2 versus their PW-0139 routed-calibration
   candidate;
2. reach validation relative L2 at most 8%;
3. have no validation row above 12%;
4. improve every pooled projection calibration output versus the identical
   affine-INT4 round-to-nearest control; and
5. remain exactly representable by 13,369,344 bytes with no added runtime MACs.

Failure rejects pooled-only calibration as the explanation for the observed
deep-layer failures and moves to function-preserving rotation or recovery
training. A pass does not authorize a bank, holdout, kernel, accumulated model,
or endpoint; it only justifies deriving a policy without validation labels.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after every projection and expert release.
