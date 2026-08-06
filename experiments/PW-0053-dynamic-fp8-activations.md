# PW-0053 — Dynamic FP8 activation semantics

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean `05c2bed437bdf6aa570848cbdd92fb77c071c4ed`
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0052 hosted manifest
  `f9c5dd42a76e0eb87581fa427fe03c69ad32903c5711e5078a002ab7514732ea`
- Hardware, OS, compiler, storage, memory pressure: M1 host and PW-0050 safety
  contract; PyTorch 2.13.0 supplies the independent E4M3FN conversion fixture
- Related records: PW-0016, PW-0039, PW-0049, PW-0050, PW-0052

## Hypothesis and mechanism

PW-0052's large hosted mismatch is caused materially by omitting the
checkpoint's declared dynamic FP8 activation quantization. The config combines
E4M3, 128x128 weight blocks, and `activation_scheme: dynamic`. The published
DeepSeek FP8 format defines online per-token-per-128-channel quantization for
that combination, and compressed-tensors names it dynamic group-size-128 input
quantization.

For every FP8 linear input row and each contiguous 128-value K group, compute
`scale=max(abs(x),1e-10)/448`, clamp `x/scale` to the finite E4M3FN range, cast
round-to-nearest-even to E4M3FN, and use the dequantized `q*scale` values in the
readable matrix reference. BF16 output projections, F32 routers, normalization,
residuals, attention, weights, and routes are otherwise unchanged.

Primary authorities:

- <https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md>
- <https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py>
- <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/fp8_utils.py>

## Contract

Before the runtime change, generate a deterministic two-row, two-group fixture
with PyTorch 2.13.0 `float8_e4m3fn`, preserving input F32 bits, group scales,
encoded bytes, and dequantized F32 bits. Rust must match every encoded byte and
dequantized value. Negative tests reject non-128-aligned rows, non-finite input,
empty input, and scale underflow.

Apply the semantic only to matrices identified as FP8 by the exact checkpoint
metadata. Keep generic weight-block validation strict and preserve the named
full-QKV scale mapping. Record dynamic groups and activation bytes in the
endpoint ledger.

Success: byte-level fixture parity, all existing component gates, safe complete
raw and chat endpoints, and material movement toward the frozen hosted answer.
Promotion to target-faithful requires the PW-0052 chosen-token/top-20 limits;
mere output readability is not sufficient.

Kill: no meaningful hosted improvement rejects dynamic activation omission as
the primary mismatch. Preserve the result and next test BF16 boundary rounding
or another source-derived semantic; do not tune scales against `Hello!`.

## Baseline and candidate

Baseline `fe367df` produces `.3` at the frozen chat prefix with hosted chosen
token logprob errors 13.5370 and 8.0002 nats. Candidate adds only the declared
per-token-group activation representation to FP8 linears.

## Isolated attribution

The independent PyTorch 2.13.0 fixture contains 512 seeded F32 activations in
four token groups, including signed zero, sub-epsilon values, saturation, and
ordinary values. Its content hashes to
`b04d86b500afaaed48f36821c42a3d423f65a3a93ca30425f513645f832dcb5a`.
Rust matches every F32 scale bit, every encoded E4M3FN byte, and every
dequantized F32 bit. Production accounting records groups and values at each
FP8 matrix boundary.

## End-to-end result

The byte-level semantic gate passes. All 26 Rust tests, 35 Python tests, strict
Clippy, and the release build pass before endpoint execution.

The raw two-token walk completed in 291.025 seconds and changed the diagnostic
continuation from `[122046,13]` (`瀛.`) to `[122982,122046]` (`棪瀛`). Its report
hash is `df45aa60896d014318abeb7bfaa52552d14d34b460b69a789ce339d38bd1dbf2`.

The frozen 27-token chat walk completed in 924.155 seconds and generated
`[13,481]` (`. -`) rather than hosted `[9707,0]` (`Hello!`). It processed
890,624 dynamic activation groups containing 113,999,872 values. Logical
source traffic was 83,325,157,120 bytes and measured process disk reads were
84,586,336,256 bytes. The report is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0053/chat-001/endpoint.json`,
SHA-256 `6e716542f0790952add3efc24ff937c0e51f24ec73ca9119bc6678d0857599b7`.

The shared-host contract remained healthy: minimum system free memory was 82%,
peak process residency was 4,418,895,872 bytes, swap did not grow, no new
throttled pages appeared, and every protected service remained alive.

## Correctness result

The candidate preserved all local gates but did not approach hosted parity.
For hosted token 9707 at step one, local logprob moved from -13.598146 to
-13.854777 while hosted was -0.061158: error worsened from 13.5370 to 13.7936
nats. For hosted token 0 at step two, local logprob moved from -8.017008 to
-7.507362 while hosted was -0.016850: error improved from 8.0002 to 7.4905
nats. Top-1 agreement remained 0/2. The opposite movements and still-enormous
errors satisfy the kill criterion for activation-quantizer omission as the
primary accumulated mismatch.

## Decision

Keep the implementation as a correctness repair: it exactly realizes the
checkpoint's declared activation scheme and does not weaken another semantic.
Reject it as the primary explanation for PW-0052. Do not tune quantizer scales
against the hosted text. The next source-derived falsification is explicit
BF16 execution-boundary rounding, for which the current F32 component fixtures
provide no accumulated whole-model authority.
