# PW-0189 — Direct-checkpoint FP8 projection

- Status: completed
- Disposition: rejected as a source-BF16 projection; physical binding remains live
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0101 authority
- Execution mode: target-faithful accelerated layer-local projection
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0033, PW-0101, PW-0187, PW-0188

## Contract and gate

Execute the real layer-4 expert-64 gate projection from its original checkpoint
shard with the promoted PW-0188 page-rounded Metal no-copy bindings for both
FP8 weights and source scales. Bind each logical tensor at its explicit
intra-page offset. Use PW-0101's source MoE input and gate output as the frozen
authority and the unchanged source-FP8 Metal kernel.

Promote a complete direct-checkpoint expert only if both source buffers copy
zero bytes; offsets and mapped-region sizes are recorded; complete projection
parity remains at relative L2 at most `2e-5` and maximum absolute error at most
`2e-4`; the real command completes; and the existing M1 timing gate remains
intact. Preserve the copied-buffer path as a named control. This is a
projection result, not accepted-token or endpoint TPS.

## Result

The direct-checkpoint command completes and produces finite output, but it
fails the frozen PW-0101 projection authority at `0.00617527` relative L2 and
`0.0678575` maximum absolute error. The cause is semantic rather than a hidden
offset relaxation: the existing Metal kernel consumes the BF16-widened input
as F32, whereas PW-0101's source path dynamically quantizes each 128-value
activation group to E4M3FN, dequantizes it, executes the full 2,048-row
projection through Accelerate, and rounds the result to BF16.

Reject this kernel as a target-faithful PW-0101 projection. Do not weaken the
predeclared thresholds or report its timing. The result does not reject
PW-0188's byte-exact mapping and is consistent with PW-0114's already named L3
repair-free arithmetic. Test direct-vs-copied equivalence under that L3
semantic separately before any wide integration. Zero tokens are accepted and
no throughput constant changes.
