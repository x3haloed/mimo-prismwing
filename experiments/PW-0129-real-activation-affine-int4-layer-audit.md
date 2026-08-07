# PW-0129 — Real-activation affine-INT4 routed-layer audit

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus manifest
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  MLX affine quantized matmul
- Exactness: explicitly modified L3 affine-INT4 candidate; source-FP8 route
  and captured output remain the target-faithful controls
- Related records: PW-0011 through PW-0019, PW-0045, PW-0116, PW-0128

## Question and changed premise

PW-0128 shows that bounded routed-layer arenas make wide execution physically
coherent, but the unchanged direct-FP32 legacy hardware misses the full-target
prefill gate. On the M1, affine group-128 INT4 is already a fast executable
representation near 53% of source expert bytes. PW-0016 measured 17.02%
routed-block relative L2, but used a synthetic eight-row activation fixture.
Before recovery training, speculation, cache composition, or a full INT4 bank,
test the unchanged quantizer on real source-routed activations.

Use PW-0116's early, middle, and late layer captures (`4`, `24`, `46`). Keep
the source selected IDs and route weights fixed. This is deliberately
optimistic: an accumulated quantized model can also change hidden states and
routes, so failure at this layer-local rung is decisive while a pass only
authorizes an accumulated probe.

## Frozen execution and authority

Authenticate the checkpoint and corpus hashes, every capture hash and shape,
all 224 positions, all 1,792 placements per layer, unique expert IDs, and route
weights. Authenticate PW-0116's committed full-corpus reconstruction hash, but
reconstruct only the in-scope `0..167` source prefix bit-for-bit before testing
a candidate. Independently replay one multi-placement source expert at each
layer through dynamic-FP8/BF16 source semantics and require the established
source tolerance. This prefix restriction resolves the preimplementation
conflict between full reconstruction and the holdout seal; it changes no
candidate threshold or observed result.

Evaluate affine group-128 INT4 as the candidate and affine group-128 INT8 as a
quality-oriented control. For every source-selected expert:

1. losslessly dequantize its three source-FP8 matrices;
2. quantize each matrix with MLX affine group-128 packing;
3. gather all in-scope real `moe_input` rows for the expert;
4. execute gate, up, SwiGLU, and down through MLX quantized matmul;
5. accumulate the unchanged source route weights in F32 and apply the source
   BF16 routed-output boundary; and
6. release that expert's source and candidate buffers before loading the next.

Add a deterministic tiny fixture for affine packing/execution, route-weighted
reduction, BF16 staging, invalid bit widths, invalid schedule cardinality, and
buffer-release accounting. Record hashes for packed values, scales, and biases; bytes,
quantization/setup wall, execution wall, MLX peak allocation, source bytes,
and Gate 8 boundaries. Candidate setup and component wall are diagnostics, not
endpoint TPS.

## Partitions and gates

Primary selection reads only train positions `0..111` and validation positions
`112..167`. Do not load or evaluate positions `168..223` unless **INT4** passes
every validation condition:

1. aggregate routed-output relative L2 across all three layers is at most 1%;
2. every layer's routed-output relative L2 is at most 2%;
3. no validation position exceeds 5% row-relative L2;
4. outputs are finite and every source route and placement remains accounted;
5. measured packed expert bytes are at most 60% of source-FP8 expert bytes;
   and
6. the INT8 control and all source-replay fixtures pass their authority gates.

If INT4 passes, unseal the final 56 positions once and apply the same numerical
thresholds without refitting or changing quantization. A validation failure
rejects naive affine INT4 for direct promotion and leaves this branch's holdout
sealed. It does not reject calibration, outlier-aware mixed precision, recovery
training, an exact codec, or other changed representations.

Do not build a full bank, train a proposer, or run an accumulated prefix from
this audit. A pass authorizes only a separately frozen accumulated
route/logit probe. Report zero accepted tokens, `A=0`, no endpoint timing, and
no performance claim. Apply normative Gate 8 at source authentication, every
layer, every expert release, corpus release, and final service-health readback.
