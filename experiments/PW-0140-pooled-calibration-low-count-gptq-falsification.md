# PW-0140 — Pooled-calibration low-count GPTQ falsification

- Status: completed
- Disposition: rejected
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

## Result

Pooled calibration materially improves all three experts but fails the frozen
uniform gate. Layer 24/expert 39 improves from `0.124820` to `0.063960`, and
expert 128 from `0.108000` to `0.068350`; both pass. Layer 46/expert 140
improves from `0.100973` to `0.083363`, with a passing `0.100029` maximum row,
but misses the `0.080000` validation ceiling by `0.003363`.

Every projection improves its pooled round-to-nearest calibration control and
the physical ledger remains 13,369,344 bytes per expert (`0.531120` of source)
with no extra runtime MACs. The decision is
`reject_pooled_only_low_count_gptq`. The positive signal does not justify a
validation-derived hybrid policy, particularly because PW-0139's deeper layer
misses also include better-covered experts. Keep holdout sealed and move to a
different geometry or learned recovery mechanism.

Gate 8 passes across 24 snapshots: minimum system memory free is 79%, maximum
peak RSS is 1,643,970,560 bytes, maximum physical footprint is 370,298,304
bytes, maximum release-boundary footprint is 333,581,696 bytes, swap growth
and new throttled pages are zero, and protected services remain resident.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0140/run-001.json`, SHA-256
`824d66549da7833d855f430a60b761f145a98757fa191bb41db6cf6e56f78b9f`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0140/analysis-001/manifest.json`,
SHA-256
`1efbd70bba8c5a3a1a7ade6668ff76d90b2bcbde7a931988fae47db0a1a7ebe9`.
No endpoint TPS or measured throughput-model constant changes.
