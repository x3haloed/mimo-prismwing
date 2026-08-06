# PW-0053 — Dynamic FP8 activation semantics

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract and fixture generator precede runtime change
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

Unexecuted.

## End-to-end result

Unexecuted.

## Correctness result

Unexecuted.

## Decision

Unexecuted.
