# PW-0196 — Direct-checkpoint source-BF16 projection semantics

- Status: completed
- Disposition: promoted as the source-semantic projection premise
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0101 authority
- Execution mode: target-faithful layer-local semantic falsifier
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0101, PW-0188, PW-0189, PW-0195

## Hypothesis

PW-0189's projection error is explained by two omitted source semantics rather
than the Metal reduction topology: dynamic per-row, 128-value E4M3FN activation
quantize/dequantize before the projection and BF16 rounding after it. Applying
those exact readable semantics around the unchanged direct-checkpoint Metal
kernel will recover the frozen PW-0101 expert-64 gate output.

## Contract and gate

Use the original checkpoint shard and PW-0188 page-rounded no-copy bindings.
Apply the same readable dynamic-FP8 activation transformation used by the slow
source endpoint before dispatch and BF16 output staging after dispatch. Compare
against PW-0101's frozen source-BF16 gate projection with relative L2 at most
`2e-5` and maximum absolute error at most `2e-4`. Do not time or claim the CPU
semantic adapters; this is a cheap causal falsifier, not an endpoint or TPS
result. Record zero accepted tokens.

If it passes, promote the semantic premise only and next move those adapters
onto the resident GPU path. If it fails, reject the premise and identify Metal
reduction topology as a remaining source-fidelity difference. Do not weaken the
gate after observing the result.

## Result

The hypothesis passes exactly. The output is byte-identical to PW-0101's
source-BF16 expert-64 gate projection: relative L2 and maximum absolute error
are both zero, and the output SHA-256 equals the frozen reference SHA-256
`1b03251ce18f62483ac2e006c3c0d379f76aa8840f1cf748120bc88a143bf93e`.
Both source tensors remain direct original-shard mappings with zero copied
bytes. The runtime report hashes to
`f7cc290c12293de07f3061e15747054e7cdec75f96313b8465e5fcacf4352b6a`.

The 0.592-ms warm Metal median excludes the readable dynamic-FP8 and BF16
adapters and is therefore diagnostic only. Promote GPU-resident adapters and
wide complete-expert validation; accept zero tokens and change no endpoint
constant.
