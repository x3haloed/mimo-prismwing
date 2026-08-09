# PW-0148 — Six-bit global-Hessian three-expert control

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
  PW-0147 raw
  `a7706fce33dc716930d080988e197089bcf1ebb6fb5729adcdb3203a8cccd62e`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  NumPy global-Hessian assignment oracle and dense-F16 execution oracle
- Exactness: explicitly modified L3 six-bit weight quantization; source routes,
  source targets, and validation partitions unchanged
- Related records: PW-0129, PW-0137 through PW-0139, PW-0142 through PW-0147

## Question and physical premise

PW-0147 proves that five-bit assignment improves every four-bit control but
does not clear the deep-expert fidelity gate. Test the next distinct code
capacity point without changing calibration, assignment, validation, or error
thresholds.

At group size 128, three six-bit code matrices require 18,874,368 bytes per
expert. F16 affine scale and bias metadata require 786,432 bytes, for
19,660,800 bytes total (`0.781059` of the 25,171,968-byte source expert). A full
routed bank is 236,558,745,600 bytes decimal (220.3125 GiB) before container
padding. It is arithmetically eligible for a prospective 256 GiB companion,
but the roughly 35.6-GiB remaining capacity must also hold the spine, KV,
runtime, OS, and safety headroom. Eligibility is not a purchase, runtime, or
TPS claim.

## Frozen numerical control

Use PW-0138/PW-0147's representative experts: layer 4/expert 96, layer
24/expert 22, and layer 46/expert 28. For every row-group, create one 64-level
affine grid from source min/max, stage scale and bias through F16, and apply the
same full-Hessian assignment with 0.1% damping, activation order, original
group lookup, and 128-column blocked error propagation. Codes are integers in
`[0,63]` and are packed conceptually at exactly six bits; dense unpacked F16
execution is an oracle only.

Calibrate only each expert's routed positions below 112. Score positions
`112..167`; keep `168..223` sealed. Reproduce source expert outputs and the
PW-0138 four-bit controls exactly. Bind and compare the immutable PW-0147
five-bit result. Add fixtures for 64-level affine endpoints, six-bit byte
arithmetic, code domain, fixed-grid reconstruction, partition isolation, and
corrupt authority rejection.

## Continuation gate

Authorize a separately frozen all-validation-expert six-bit audit only if:

1. every expert reaches validation relative L2 at most 2%;
2. every maximum validation row is at most 5%;
3. every candidate improves its train output over six-bit round-to-nearest;
4. every candidate improves on its exact PW-0147 five-bit validation result;
5. four-bit controls reproduce, all codes remain in `[0,63]`, source and
   partitions remain authoritative, and holdout stays sealed; and
6. packed bytes remain exactly 19,660,800 per expert (`<=80%` of source) with
   zero additional runtime MACs.

Failure rejects this affine-group-128/global-Hessian six-bit form on the three
representative experts. A pass authorizes only an all-validation-expert audit,
not holdout, a bank, kernel, companion purchase, accumulated model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after every projection and expert release.
