# PW-0129 — Real-activation affine-INT4 routed-layer audit

- Status: completed
- Disposition: rejected for naive affine INT4
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

## Result

The clean implementation run completed in 151,984.255 ms. Every capture and
route authority passed; the in-scope source prefix reconstructed bit-for-bit
at all three layers. Independent dynamic-FP8/BF16 source replays for experts
96, 23, and 28 also matched their captured outputs bit-for-bit.

INT4 achieves its physical objective: every streamed expert occupies
13,369,344 executable bytes, or 53.1120% of the 25,171,968-byte source-FP8
record. It misses every numerical gate by a wide margin:

| Layer | Train routed relative L2 | Validation routed relative L2 | Validation worst row |
| ---: | ---: | ---: | ---: |
| 4 | 1.5812% | 4.1919% | 7.7105% |
| 24 | 10.6954% | 11.9174% | 17.9215% |
| 46 | 14.6523% | 15.4606% | 17.2042% |

Validation aggregate relative L2 is 9.7661%, versus the 1% gate. Maximum
layer error is 15.4606% versus 2%, and maximum row error is 17.9215% versus
5%. The failure is depth-general and appears on both train and validation; it
is not explained by a single synthetic input or validation-only route novelty.
The holdout remained sealed.

INT8 provides a monotonic quality control rather than a compact candidate. Its
validation errors are 0.9703%, 2.4061%, and 3.5508% across layers 4, 24, and
46, materially better than INT4 but still above the per-layer gate at the two
deeper layers. Its 25,952,256-byte artifact is 103.100% of source-FP8 bytes and
therefore cannot change the traffic premise.

Candidate execution timings exclude source load, quantization, attention,
accumulated routing, and the endpoint; they remain diagnostic only. Gate 8
passes at 78% minimum free memory, 599,113,728-byte maximum peak RSS,
226,265,920-byte maximum physical footprint, zero swap growth or new
throttled pages, and stable protected services. Raw evidence hashes to
`1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
independent analysis hashes to
`6d7f75d8b65ccd0ba2fe5c3767e2f2e2a4841c4a859749dbcab8289c7c29b673`.

## Decision

Reject naive affine group-128 INT4 on real routed activations before a full
bank, accumulated prefix, cache composition, or proposer training. Its
physical compression is real, but its layer-local error is already roughly
5--15 times the allowed layer gate under source routes, an optimistic setting.
Do not read this branch's holdout. Preserve affine INT8 as a quality-oriented
diagnostic only; it is larger than source FP8 and also misses the deeper-layer
gate. Calibration, outlier-aware mixed precision, recovery training, exact
codecs, and structurally different representations remain separate candidates.
No endpoint TPS or measured throughput constant changes.
