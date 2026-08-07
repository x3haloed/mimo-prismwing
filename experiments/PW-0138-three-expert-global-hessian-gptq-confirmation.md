# PW-0138 — Three-expert global-Hessian GPTQ confirmation

- Status: completed
- Disposition: promoted to all-validation-expert audit
- Date: 2026-08-07
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0129 raw
  `1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
  PW-0135 raw
  `56b9d38c3c630359b8d5b1a911627882df06a2e2fc374751fde2fddaeb3888db`;
  PW-0137 raw
  `95fee340bb676ac7c9486ea713da9c461ca6fb62441b41b32ff988e97ed1502e`;
  PW-0137 analysis
  `7a741514aad2f4ec783cd95b1283ae5b98afbcdad17cd64e8a7759c12f3b5d67`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  NumPy global-Hessian assignment oracle and dense-F16 execution oracle
- Exactness: explicitly modified L3 weight-only quantization; unchanged
  source-FP8 and affine-INT4 controls
- Related records: PW-0116, PW-0129, PW-0135, PW-0137

## Question and scope

PW-0137 rescues layer 46/expert 28. Confirm that the same mechanism retains
PW-0135's passing layer 4/expert 96 and layer 24/expert 22 results while exactly
reproducing the layer-46 rescue before expanding to all validation experts.

Use the same expert placements and sealed partitions as PW-0135: 109/56 train
and validation placements at layer 4, 26/56 at layer 24, and 100/56 at layer
46. Positions `168..223` remain excluded from all numerical inputs.

## Frozen mechanism

Apply PW-0137 unchanged to gate, up, and down for all three experts:

- original MLX affine group-128 INT4 scale and bias;
- one full train-only activation Hessian per projection;
- 0.1% full-Hessian mean-diagonal damping;
- descending full-Hessian-diagonal activation order;
- static grid lookup by original pre-permutation group;
- 128-column blocks with error propagation into all later columns; and
- exact original-order grid reconstruction before expert composition.

Do not tune by layer or projection. Gate/up use real MoE input; down uses
source-derived BF16 SwiGLU input. Reuse PW-0137's deterministic fixtures and
add coverage for three-expert identity and exact PW-0137 reproduction.

## Continuation gate

Authorize a separately frozen all-validation-expert audit only if every expert:

1. reduces validation relative L2 by at least 50% versus its identical
   unpacked affine-INT4 control;
2. reaches validation relative L2 at most 8%;
3. has no validation row above 12%;
4. improves rather than regresses on train;
5. is no worse than its PW-0135 group-local candidate; and
6. remains exactly representable by the original 13,369,344-byte affine-INT4
   payload with no extra runtime MACs.

Layer 46/expert 28 must reproduce PW-0137's train and validation metrics,
projection grid hashes, and activation-order hashes exactly. Failure rejects
this frozen global-Hessian mechanism as the current bank candidate. A pass
authorizes only a new all-validation-expert contract—not holdout, a packed
runtime, kernel, full bank, accumulated model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after each projection and at every expert
release, including conservative projected-workspace preflight. Stop on any
memory-pressure, swap, throttling, service-health, or release-boundary failure.

## Result

All three experts pass and layer 46/expert 28 exactly reproduces PW-0137's
metrics, grid hashes, and activation-order hashes. Validation relative L2 is
`0.022155`, `0.059604`, and `0.059227` at layers 4/24/46, reductions of
79.77%, 66.87%, and 63.73% from their identical affine-INT4 controls. Maximum
row errors are `0.036150`, `0.079795`, and `0.077608`; train errors are
`0.005445`, `0.041274`, and `0.033130`. Every frozen condition passes.

The physical ledger remains 13,369,344 bytes per expert (`0.531120` of source)
with no additional runtime MACs. Holdout remains sealed. The decision is
`authorize_all_validation_expert_global_hessian_audit`; no runtime artifact,
kernel, bank, accumulated model, or endpoint is authorized yet.

Gate 8 passes across 24 snapshots: minimum system memory free is 79%, maximum
peak RSS is 1,648,082,944 bytes, maximum physical footprint is 417,795,456
bytes, maximum release-boundary footprint is 395,726,080 bytes, swap growth
and new throttled pages are zero, and protected services remain resident.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0138/run-001.json`, SHA-256
`37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0138/analysis-001/manifest.json`,
SHA-256
`7ed32546bfb042d5b863c23d812eeada89cafb7d65b9c1d86c30c7483022e14b`.
No endpoint TPS or measured throughput-model constant changes.
