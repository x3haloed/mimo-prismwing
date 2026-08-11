# PW-0183 — Projection-sensitive mixed precision

- Status: completed
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0116; PW-0182
- Execution mode: shadow L3 mixed affine precision beside source BF16
- Runtime: MLX 0.31.2 on the existing Apple M1
- Related records: PW-0147, PW-0148, PW-0182

## Contract

Test the current upstream MLX mixed-precision premise on layer-46 expert 28:
gate/up at 3 or 4 bits and down at 6 or 8 bits, all affine group-128 and directly
executed by `quantized_matmul`. The frozen candidates are `3/3/6`, `4/4/6`,
`3/3/8`, and `4/4/8` for gate/up/down. The first is the only primary physical
candidate; the others attribute capacity.

Authenticate PW-0182 and all source authorities. Add a deterministic fixture
proving per-projection configuration dispatch and exact packed-byte accounting.
Keep holdout sealed. Evaluate all 56 validation positions with the same source
control and record projection/complete errors, packed bytes, and batch-one warm
median/p95 after ten warmups and 50 trials.

Promote only a candidate with complete validation L2 at most 2%, maximum-row
error at most 5%, gate/up each at most 2%, no more than 1.05 times the
13,369,344-byte INT4 expert, and warm median at most 0.75 ms. A pass authorizes
an all-validation audit and fused eight-expert `gather_qmm` layer transaction.
No result here is endpoint TPS.

If `4/4/8` still misses 5% complete error, reject projection-only bit allocation
as the missing fidelity mechanism. If it passes but the primary `3/3/6` fails,
the bytes required by this family are physically ineligible for 1 TPS; do not
search intermediate widths without a new compression mechanism.

## Result

The authoritative report hashes to
`38a6ac68ce858bd0e7e06ffa8a31974ea625e6cf1648d554801d2dc289506fb0`.
No candidate passes. The only primary physical candidate, `3/3/6`, exactly
matches the 13,369,344-byte INT4 expert and runs in 0.5977 ms warm median, but
complete validation error is `0.255800`; gate/up are `0.163618/0.113376`.

Even the diagnostic `4/4/8` candidate remains at `0.123779` complete error
while growing to 17,563,648 bytes (`1.313725` times INT4). Raising only the
down-projection precision barely changes the corresponding gate/up-controlled
result. Reject projection-only mixed precision as the missing fidelity
mechanism. The holdout remains sealed; zero tokens are accepted and no
endpoint TPS or throughput constant changes.
