# PW-0146 — Train-only threshold-crossing code QAT

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
  PW-0145 raw
  `1d5f4f4bf9dacc39114d483f90e3e61590f847aa24a31a1c6d48dbb077deafa4`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX full-batch straight-through optimizer and dense-F16 execution oracle
- Exactness: explicitly modified L3 training preflight; no validation or
  holdout target is loaded or scored
- Related records: PW-0144, PW-0145

## Question and frozen schedule

PW-0145's largest low-rate trial reaches only 0.159808 latent displacement and
changes no code; PW-0144's much larger schedule reaches 7.25 and diverges after
changing 23.33% of codes. Test one predicted first-bin schedule before leaving
this parameterization: layer 46/expert 249, its same 90 train placements,
learning rate `0.02`, 32 full-batch steps, Adam betas `(0.9,0.999)`, epsilon
`1e-8`, bias correction enabled, and `1e-6` latent regularization.

The schedule, expert, and duration are fixed from the train-only displacement
bracket. Do not load validation inputs or expert outputs. Start from the exact
PW-0139 grid and zero latent offsets. Record train loss every four steps,
source-output train error, changed codes by projection, code domain, F16 grid
identity, maximum latent displacement, wall time, and Gate 8 evidence.

## Continuation gate

Authorize one separately frozen train/validation confirmation only if:

1. train relative L2 falls at least 25% from PW-0139;
2. final loss is finite and below initial loss;
3. at least one and no more than 5% of codes change;
4. all codes remain in `[0,15]` and F16 grid metadata is bit-identical; and
5. the 13,369,344-byte (`0.531120`) zero-extra-MAC ledger remains exact.

Failure rejects further schedule search for this fixed-grid, straight-through
latent parameterization. A pass authorizes only a new validation experiment
with this exact schedule—not holdout, bank, kernel, accumulated model, or
endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 at least every eight steps and after release.

## Result

The schedule crosses the intended first-bin regime but is immediately
destructive. Maximum latent displacement reaches `0.628309`; 590,345 of
25,165,824 codes change (2.3458%), inside the frozen 5% bound. Train relative
L2 nevertheless explodes from `0.039279` to `1.062956`, and normalized loss
rises from `0.001542818` to `1.129876494`.

The code-domain, changed-code, grid-metadata, physical, and safety conditions
all pass. Validation values were never loaded and holdout remains sealed. The
result joins PW-0145's inert low-rate family and PW-0144's destructive high
schedule: within this fixed-grid, independent latent-offset, straight-through
form, schedules below the boundary cannot change the artifact and the first
bounded boundary crossing destroys the train objective. End schedule search
for this parameterization. Grid-changing training or a structurally different
executable representation is required.

The runtime ledger remains 13,369,344 bytes per expert (`0.531120` of source)
with zero additional MACs. Gate 8 passes across 15 snapshots: minimum free
memory is 59%, maximum peak RSS is 1,533,788,160 bytes, maximum physical
footprint is 1,424,920,128 bytes, maximum release-boundary footprint is
302,648,448 bytes, swap growth and new throttled pages are zero, and protected
services remain resident.

Raw evidence:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0146/run-001.json`, SHA-256
`7bb795455927295c673bfe65d06ae6311dbdd97b9d3517caa357307d189bdcf3`.
Validated analysis:
`/Users/chad/Models/mimo-prismwing/evidence/PW-0146/analysis-001/manifest.json`,
SHA-256
`9ea3ba39fc7381389e168b09478cdf3789cbcce846ac82b096e392b5d42bc3f3`.
No endpoint TPS or measured throughput-model constant changes.
