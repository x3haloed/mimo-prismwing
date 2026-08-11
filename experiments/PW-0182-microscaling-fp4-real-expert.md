# PW-0182 — Microscaling FP4 real-expert control

- Status: complete
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; authenticated PW-0116
  routed-activation corpus; layer-46 hot expert 28
- Execution mode: shadow L3 weight formats beside a target-faithful source
  expert; candidate state never enters the model
- Runtime: MLX 0.31.2 `quantized_matmul` on the existing Apple M1
- Related records: PW-0015, PW-0129, PW-0149, PW-0177 through PW-0181

## Changed premise

PW-0181 grouped all four-bit paths under affine/scalar or custom-vector
representations. Current MLX exposes directly executable MXFP4 (E2M1 values,
one E8M0 scale per 32 weights) and NVFP4 (E2M1 values, E4M3 scales per 16),
which preserve exponent structure differently from every rejected form.

## Contract

Authenticate the checkpoint and PW-0116 corpus, reconstruct layer-46 expert 28
from source FP8, and keep positions `168..223` sealed. Add a deterministic tiny
fixture proving quantize/dequantize determinism and equality between
`quantized_matmul` and multiplication by the explicitly dequantized matrix.

Quantize all three projections independently as MXFP4, NVFP4, and affine
INT4/group-32 matched control. Preserve packed arrays and scales outside Git
with shape, dtype, bytes, and SHA-256. Evaluate the complete expert on all 56
validation positions with MLX FP16 activations and direct quantized kernels.
Compare against the bit-exact source BF16 capture and report gate, up, complete
expert, weight-reconstruction, and maximum-row errors.

For each mode, measure one complete batch-one expert after ten warmups and 50
synchronized trials. Record median/p95, batch size, concurrency, accepted
tokens, `A`, `U`, hardware, OS, MLX version, and implementation identity.
Timings are component diagnostics only.

Promote a mode only if validation complete-expert relative L2 is at most 2%,
maximum-row relative L2 is at most 5%, gate/up relative L2 are each at most 2%,
packed expert bytes are no greater than the 13,369,344-byte affine-INT4
artifact, and warm median is at most 0.75 ms. A pass authorizes an all-
validation-expert audit and an eight-expert fused `gather_qmm` routed-layer
transaction—not a bank, endpoint, or TPS claim.

Reject each mode independently if it misses. A failure above 10% closes that
format on this deep control; a smaller miss may authorize mixed precision only
when its exact byte ledger can still beat affine INT4 traffic. Report zero
accepted tokens and leave throughput constants unchanged until an endpoint.

## Result

The valid report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0182/run-001/report.json`
rejects every tested mode. Its SHA-256 is
`db62501ba622bb09a18db327c06cc883ab51ec978836f6d0dc703ab72ebbf485`.

MXFP4 exactly matches the 13,369,344-byte affine-INT4 envelope and executes a
complete expert in 0.5875 ms warm median, but validation relative L2 is
`0.193978` with `0.211392` maximum-row error. NVFP4 uses 14,155,776 bytes and
reaches only `0.167407`. The group-32 affine control is better at `0.134476`
but costs 15,728,640 bytes. Gate/up errors remain far above 2% in every mode.

Kill direct MXFP4 and NVFP4 for this deep MiMo expert. Preserve the fast
direct-kernel result and advance only projection-sensitive mixed precision,
which changes the allocation of bits rather than the four-bit number system.
The holdout remains sealed; zero tokens are accepted and no endpoint TPS or
throughput constant changes.
