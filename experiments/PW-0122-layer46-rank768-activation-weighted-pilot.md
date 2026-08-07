# PW-0122 — Layer-46 rank-768 activation-weighted pilot

- Status: proposed
- Disposition: unexecuted
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
  PW-0121 analysis
  `6f3c7e8d9ddd25db65dc35cb888a98349bfa89b538cc33be7a2e0ffe5e3c6d17`
- Hardware/runtime: Apple M1 shared 16 GiB; NumPy Accelerate SVD, PyTorch
  2.13 MPS optimization and CPU source-FP8 evaluation; internal SSD
- Exactness: L4 fitted diagnostic; unchanged source control
- Related records: PW-0045, PW-0116 through PW-0121; E5

## Question and changed premise

PW-0121 shows that per-projection activation-weighted rank-768 fitting can
generalize to an untouched middle-layer holdout despite poor global-SVD error.
Replicate the identical mechanism at the late-layer PW-0116 control before
spending on shared bases. Layer 46 has materially different singular spectra,
activation scales, and downstream sensitivity; a layer-24 pass alone cannot
establish that the mechanism survives depth.

## Frozen execution

Use layer 46 hot expert 28 only: 100 train placements at positions `0..111`,
56 validation placements at `112..167`, and 56 untouched pilot-holdout
placements at `168..223`. Use seed `260122`. All factor dimensions, balanced
rank-768 SVD initialization, source-target derivation, F32 normalized-MSE
surrogate, Adam `0.001`, 100-step maximum, five-step validation interval,
four-check patience, source-FP8/BF16 complete-expert evaluation, factor hashing,
non-persistence, MPS 0.60 cap, and Gate 8 observations remain identical to
PW-0121.

Refactor the PW-0121 executor around one immutable experiment specification;
do not fork a second training implementation. Preserve PW-0121's defaults and
tests, and add a fixture proving that layer/expert/count/baseline/threshold
authority is injected explicitly and emitted unchanged.

## Gates

1. The source oracle must reproduce all captured expert-down BF16 values
   bit-for-bit. The authoritative `U,S,Vt` rank-768 control must reproduce
   PW-0119 within `1e-6`; balanced initialization must remain within `5e-6` in
   relative L2.
2. Every projection's selected validation normalized MSE must be finite and
   below step zero. Selection must not inspect the pilot holdout.
3. The complete candidate must reduce validation relative L2 by at least 25%
   from `0.572330134931118` (candidate at most
   `0.4292476011983385`) and untouched holdout relative L2 by at least 25%
   from `0.5458150398186078` (candidate at most
   `0.4093612798639559`). Train and overall metrics remain diagnostics.
4. Gate 8 and zero-current-MPS release requirements are unchanged from
   PW-0121. Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS.

Passing authorizes a separately contracted multi-expert shared-basis pilot
within one layer. It does not authorize a full layer or bank, a persisted L4
artifact, kernel work, or runtime changes. Failure rejects depth-general use of
this fitting recipe and requires a different objective or representation; do
not tune on the holdout.

## Result

Unexecuted.

## Decision

Unexecuted.
