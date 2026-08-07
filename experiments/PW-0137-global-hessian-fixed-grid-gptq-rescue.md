# PW-0137 — Global-Hessian fixed-grid GPTQ rescue

- Status: completed
- Disposition: promoted to three-expert confirmation
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
  `56b9d38c3c630359b8d5b1a911627882df06a2e2fc374751fde2fddaeb3888db`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  NumPy global-Hessian assignment oracle and dense-F16 execution oracle
- Exactness: explicitly modified L3 weight-only quantization; unchanged
  source-FP8 and affine-INT4 controls
- Related records: PW-0116, PW-0129, PW-0135, PW-0136

## Question and scope

PW-0135 missed its complete-expert validation ceiling only at layer 46/expert
28: `0.080659` versus `0.080000`. Its group-local implementation discarded
all Hessian coupling across the fixed 128-column INT4 groups. Test whether the
standard GPTQ cross-block error update rescues that expert before repeating a
three-expert or full-layer audit.

This is a one-expert capacity falsification, not a quantization sweep. It uses
the same 100 train placements and 56 validation placements as PW-0135 and does
not read positions `168..223`.

## Frozen global-Hessian mechanism

For each gate, up, and down projection of layer 46/expert 28:

1. retain MLX affine group-128 INT4's original per-row scale and bias;
2. form one full input-channel activation Hessian from train positions
   `0..111` only;
3. add the already selected PW-0135 damping of `0.1%` of the full Hessian's
   mean diagonal;
4. use the already selected descending Hessian-diagonal activation order;
5. process 128 columns at a time, quantizing each column to the fixed affine
   grid associated with its original input-channel group;
6. propagate normalized error inside the current block and, at every block
   boundary, update all remaining columns through the full inverse-Cholesky
   factor; and
7. undo the activation-order permutation before composing the expert.

Gate/up use the real expert MoE inputs. Down uses source-derived BF16 SwiGLU
inputs. The implementation must add deterministic fixtures proving cross-block
propagation, original-group grid lookup under activation order, grid
membership, blocked-versus-unblocked identity, singular/dead activation
handling, and partition isolation.

The capacity oracle may execute unpacked F16 grid values, but must prove every
value belongs to the original affine grid and charge the unchanged packed
13,369,344-byte INT4 payload. It may not report dense-oracle memory or timing
as executable-artifact or endpoint performance.

## Continuation gate

Authorize a separately frozen three-expert confirmation only if the composed
layer 46/expert 28 candidate:

1. reduces validation relative L2 by at least 50% versus the identical
   unpacked affine-INT4 control;
2. reaches validation relative L2 at most 8%;
3. has no validation row above 12%;
4. improves rather than regresses on train;
5. is no worse than PW-0135's `0.080659` validation result; and
6. remains exactly representable by the original 13,369,344-byte affine-INT4
   payload with no extra runtime MACs.

Failure rejects this full-Hessian, fixed-grid, 0.1%-damped, activation-ordered
GPTQ form. It does not reject function-preserving rotations, learned recovery,
different grids, or mixed precision. A pass authorizes only a new three-expert
confirmation contract—not holdout, a runtime artifact, a kernel, accumulated
model fidelity, or endpoint performance.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 after every projection and every full-Hessian workspace
release. Stop before allocation if projected resident use plus current physical
footprint exceeds the Gate 8 headroom; stop on any memory-pressure, swap,
throttling, protected-service-health, or release-boundary failure.

## Result

The frozen one-expert gate passes. Layer 46/expert 28 reaches `0.059227`
validation relative L2 and a `0.077608` maximum row, versus `0.163279` for the
identical affine-INT4 control and `0.080659` for PW-0135's group-local GPTQ.
That is a 63.73% reduction from the control and a 26.57% improvement over the
group-local candidate. Train relative L2 is `0.033130`, so every continuation
condition passes.

All three projections improve their train control and record nonzero
cross-block updates. The candidate remains exactly on the original affine grid,
costs 13,369,344 bytes (`0.531120` of source), and adds no runtime MACs.
Holdout remains sealed. The decision is
`authorize_three_expert_global_hessian_confirmation`; this is not authorization
for a full layer, runtime artifact, kernel, accumulated model, or endpoint.

Gate 8 passes across nine snapshots: minimum system memory free is 78%, maximum
peak RSS is 1,576,271,872 bytes, maximum physical footprint is 384,945,408
bytes, maximum release-boundary footprint is 365,005,952 bytes, swap growth
and new throttled pages are zero, and protected services remain resident.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0137/run-001.json`, SHA-256
`95fee340bb676ac7c9486ea713da9c461ca6fb62441b41b32ff988e97ed1502e`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0137/analysis-001/manifest.json`,
SHA-256
`7a741514aad2f4ec783cd95b1283ae5b98afbcdad17cd64e8a7759c12f3b5d67`.
No endpoint TPS or measured throughput-model constant changes.
