# PW-0139 — All-validation-expert global-Hessian audit

- Status: completed
- Disposition: rejected
- Date: 2026-08-07
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0129 raw
  `1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
  PW-0138 raw
  `37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49`;
  PW-0138 analysis
  `7ed32546bfb042d5b863c23d812eeada89cafb7d65b9c1d86c30c7483022e14b`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  NumPy global-Hessian assignment oracle and dense-F16 execution oracle
- Exactness: explicitly modified L3 weight-only quantization; source routes,
  route weights, reduction order, and BF16 routed-output boundary unchanged
- Related records: PW-0116, PW-0129, PW-0135, PW-0137, PW-0138

## Question and scope

Test the frozen PW-0138 mechanism on every expert selected by validation
positions `112..167` at layers 4, 24, and 46: 10, 16, and 15 unique experts,
covering all 448 routed placements per layer. Keep positions `168..223`
sealed. Reconstruct the complete route-weighted layer output, rather than
promoting from isolated expert errors.

## Frozen calibration and execution

For each expert with train placements, calibrate each projection only on that
expert's routed positions in `0..111`. Layer 24 experts 25 and 251 have no
train placements; for those two declared fallbacks only, use all 112 layer
MoE inputs for gate/up and the corresponding source-expert BF16 SwiGLU values
for down. Record the fallback identity and count; do not use validation values
for calibration or selection.

Apply PW-0138 unchanged: original affine group-128 grids, one full Hessian,
0.1% damping, descending activation order, static original-group lookup, and
128-column blocked error propagation. No per-expert tuning is permitted.

Execute every candidate through the dense-F16 grid oracle, apply the source
route weights in the source schedule, accumulate in F32, and cast the final
routed result through BF16 exactly as PW-0129. Reconstruct the source prefix
and independent source replay before candidate evaluation. Prove every grid
assignment, placement, route, and physical ledger entry. The three PW-0138
experts must reproduce their metrics and assignment hashes exactly.

## Continuation gate

Authorize a separately frozen holdout audit only if:

1. aggregate routed-output validation relative L2 across all three layers is
   at most 1%;
2. every layer is at most 2% relative L2;
3. no validation row exceeds 5% relative L2;
4. every output is finite and all routes and placements are accounted;
5. every projection improves its calibration output over the identical
   round-to-nearest affine-INT4 control;
6. all three PW-0138 controls reproduce exactly; and
7. the artifact remains 13,369,344 bytes per expert (`<=60%` of source) with
   no additional runtime MACs.

Failure rejects this frozen global-Hessian fixed-grid form as a complete routed
layer candidate on the current corpus. It does not reject rotations, recovery
training, mixed precision, or different grids. A pass authorizes only a new
holdout contract—not a bank, packed kernel, accumulated model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after every projection and expert release.
Stop on projected-headroom, memory-pressure, swap, throttling, protected-
service-health, or release-boundary failure.

## Result

The full-validation gate fails despite exact authority, coverage, physical, and
calibration checks. Aggregate routed-output relative L2 is `0.035040` versus
the `0.010000` ceiling. Layer errors are `0.010130`, `0.040686`, and
`0.057541` at layers 4/24/46; the maximum row is `0.082790` versus `0.050000`.
Only layer 4 clears the per-layer and row gates.

All 41 experts, 1,344 routed placements, and three source replays are
accounted. Every projection improves its calibration control, both declared
train-absent fallbacks execute, and all three PW-0138 controls reproduce
exactly. This therefore rejects the frozen expert-specific/global-Hessian
fixed-grid form on broad validation rather than exposing an authority or
implementation failure. Holdout remains sealed.

The deeper layers show a calibration-coverage problem but not an exclusively
low-count failure. Layer 24 experts 39 and 128 have only 6 and 19 routed train
placements and reach `0.124820` and `0.108000` validation error; layer 46
expert 140 has 10 and reaches `0.100973`. However, several better-covered
experts remain near 6--7%, so pooled-Hessian shrinkage is only a bounded next
falsification, not a presumed rescue.

The physical ledger remains 13,369,344 bytes per expert (`0.531120` of source)
with no additional runtime MACs. Gate 8 passes across 296 snapshots: minimum
system memory free is 78%, maximum peak RSS is 1,765,031,936 bytes, maximum
physical footprint is 394,153,664 bytes, maximum release-boundary footprint is
348,393,152 bytes, swap growth and new throttled pages are zero, and protected
services remain resident.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0139/run-001.json`, SHA-256
`83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0139/analysis-001/manifest.json`,
SHA-256
`9aecfdcd32e535b4b9d27fcac075dfd1c9014080d624aa3f4af2c678be3f3b6c`.
No endpoint TPS or measured throughput-model constant changes.
