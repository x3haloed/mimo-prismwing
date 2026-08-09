# PW-0147 — Five-bit global-Hessian three-expert control

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0138 raw
  `37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49`;
  PW-0146 raw
  `7bb795455927295c673bfe65d06ae6311dbdd97b9d3517caa357307d189bdcf3`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  NumPy global-Hessian assignment oracle and dense-F16 execution oracle
- Exactness: explicitly modified L3 five-bit weight quantization; source routes,
  source targets, and validation partitions unchanged
- Related records: PW-0129, PW-0137 through PW-0139, PW-0142 through PW-0146

## Question and physical premise

The fixed-grid four-bit branch fails broad validation and its cheap recovery
families are exhausted. Test whether adding one code bit supplies enough
assignment capacity while retaining a materially compressed executable bank.

At group size 128, three five-bit code matrices require 15,728,640 bytes per
expert. F16 affine scale and bias metadata require 786,432 bytes, for
16,515,072 bytes total (`0.656090` of the 25,171,968-byte source expert). A full
routed bank would be approximately 198.7 GB decimal before container padding,
small enough for a prospective 256 GiB companion-memory embodiment but not the
16 GiB M1. This is physical eligibility, not a hardware or TPS claim.

## Frozen numerical control

Use the original PW-0138 representative experts: layer 4/expert 96, layer
24/expert 22, and layer 46/expert 28. For every row-group, create one 32-level
affine grid from source min/max, stage scale and bias through F16, and apply
PW-0138's full-Hessian assignment with 0.1% damping, activation order, original
group lookup, and 128-column blocked error propagation. Codes are integers in
`[0,31]` and are packed conceptually at exactly five bits; dense unpacked F16
execution is an oracle only.

Calibrate only each expert's routed positions below 112. Score positions
`112..167`; keep `168..223` sealed. Reproduce source expert outputs and the
PW-0138 four-bit controls exactly. Add fixtures for 32-level affine endpoints,
five-bit packing byte arithmetic, code domain, fixed-grid reconstruction,
cross-block propagation, partition isolation, and corrupt input rejection.

## Continuation gate

Authorize a separately frozen all-validation-expert five-bit audit only if:

1. every expert reaches validation relative L2 at most 2%;
2. every maximum validation row is at most 5%;
3. every candidate improves its train output over five-bit round-to-nearest;
4. every candidate improves on its exact PW-0138 four-bit validation control;
5. all codes remain in `[0,31]`, source and prior controls reproduce, and
   holdout remains sealed; and
6. packed bytes remain exactly 16,515,072 per expert (`<=70%` of source) with
   zero additional runtime MACs.

Failure rejects this affine-group-128/global-Hessian five-bit form on the three
representative experts. A pass authorizes only an all-validation-expert audit,
not holdout, a bank, kernel, companion purchase, accumulated model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after every projection and expert release.
