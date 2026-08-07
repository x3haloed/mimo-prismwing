# PW-0141 — Fixed residual-Hadamard rotation control

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
  PW-0140 raw
  `824d66549da7833d855f430a60b761f145a98757fa191bb41db6cf6e56f78b9f`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  NumPy randomized-Hadamard transform, global-Hessian assignment oracle, and
  dense-F16 execution oracle
- Exactness: explicitly modified L3 weight-only quantization; algebraically
  exact residual-basis change before quantization
- Related records: PW-0129, PW-0135, PW-0137 through PW-0140

## Question and scope

PW-0139 rejects the fixed-grid bank even for several well-covered deep experts,
so calibration density is not the entire problem. Test whether a single
function-preserving randomized-Hadamard residual basis changes weight outlier
geometry enough to improve global-Hessian INT4 on well-covered experts.

Use layer 4/expert 96 as an early control, layer 24/expert 200 (71 routed train,
37 validation placements), and layer 46/expert 249 (90/56). These inputs avoid
PW-0140's sparse-calibration confound. Validation is already unsealed for these
experts; positions `168..223` remain sealed.

## Frozen rotation and quantization

Derive one 4,096-element Rademacher sign vector from SHA-256 of
`PW-0141|63651580ca774f8504f676040460aed3e1244ac1` and use it at every layer.
Let `Q = D H`, where `D` is that sign diagonal and `H` is the normalized
4,096-point Walsh-Hadamard matrix. For row-vector execution:

- rotate input `z = x Q`;
- rotate gate/up weights `Wg' = Wg Q`, `Wu' = Wu Q`;
- rotate down output rows `Wd' = Q^T Wd`; and
- compare `f'(z) Q^T` with the source-frame expert output.

Before quantization, require float64 forward parity and orthogonal round-trip
error at most `1e-10` relative L2. This local oracle explicitly unrotates the
output; a future whole-model embodiment would keep the residual stream in the
rotated basis and fold `Q` into neighboring weights, so this experiment adds no
online transform to its prospective expert ledger.

For each rotated projection, create a new MLX affine group-128 INT4 grid and
apply PW-0139's full-Hessian, 0.1%-damped, activation-ordered, 128-column GPTQ
using only that expert's routed train positions. No rotation seed, grid,
damping, or ordering search is allowed. Add fixtures for normalized FWHT,
sign order, right and left rotations, exact round trip, unquantized SwiGLU
parity, grid membership, and sealed partitions.

## Continuation gate

Authorize only a separately frozen broader rotation confirmation if:

1. layer 24/expert 200 and layer 46/expert 249 each improve PW-0139 validation
   relative L2 by at least 25% and reach at most 5%;
2. their maximum validation row is at most 8%;
3. layer 4/expert 96 does not regress by more than 10% from PW-0139;
4. every rotated projection improves its train output over rotated affine-INT4
   round-to-nearest;
5. all unquantized algebra and transform fixtures pass; and
6. the packed payload remains 13,369,344 bytes per expert with no online
   residual-transform operation in the prospective globally rotated model.

Failure rejects this fixed residual-Hadamard seed/form, not learned rotations
or recovery training. A pass authorizes only a broader validation confirmation,
not holdout, a whole-model rotation, bank, runtime, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after every projection and expert release.

## Result

The fixed rotation is algebraically correct but numerically ineffective for
weight-only affine INT4 plus GPTQ. Unquantized forward parity is between
`1.93e-15` and `3.52e-15`, and every round trip is exact at reported precision.
Every rotated GPTQ projection improves its rotated round-to-nearest train
control.

Validation remains essentially unchanged: layer 4/expert 96 moves from
`0.022155` to `0.022241`; layer 24/expert 200 from `0.065851` to `0.065040`;
and layer 46/expert 249 from `0.067440` to `0.066235`. The deep improvements
are only 1.23% and 1.79%, far below the required 25%, and neither reaches 5%.
Rotated round-to-nearest is substantially worse, at `0.095517`, `0.150095`,
and `0.185599`, so GPTQ—not the rotation—is doing nearly all useful work.

The decision is `reject_fixed_residual_hadamard_rotation`. The payload remains
13,369,344 bytes per expert (`0.531120` of source) with no prospective online
residual transform inside a globally rotated model. Keep holdout sealed and do
not rotate the checkpoint or build the runtime. Learned rotations remain
logically distinct, but the next bounded branch is recovery training.

Gate 8 passes across 24 snapshots: minimum system memory free is 79%, maximum
peak RSS is 1,789,853,696 bytes, maximum physical footprint is 373,395,072
bytes, maximum release-boundary footprint is 346,541,696 bytes, swap growth
and new throttled pages are zero, and protected services remain resident.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0141/run-001.json`, SHA-256
`4dae2abe2a59457a77e09bd4d1328b7b6dce8f0e41e3ac115fd27645c93e56a9`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0141/analysis-001/manifest.json`,
SHA-256
`0cd0f7d5cd9d8fd563a1c35888a16e8452f8f9128d26c36ce7e02d646cc3bf26`.
No endpoint TPS or measured throughput-model constant changes.
