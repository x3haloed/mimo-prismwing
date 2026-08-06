# PW-0054 — BF16 execution-boundary semantics

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean `6bd74ebccddb0b314338adceabccdbaf04a04282`
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0052 hosted manifest
  `f9c5dd42a76e0eb87581fa427fe03c69ad32903c5711e5078a002ab7514732ea`
- Hardware, OS, compiler, storage, memory pressure: M1 shared 16 GiB host and
  PW-0050 safety contract; PyTorch 2.13.0 supplies the independent BF16 fixture
- Related records: PW-0049, PW-0050, PW-0052, PW-0053

## Hypothesis and mechanism

PW-0053 still executes almost every model tensor operation in F32 even though
the pinned checkpoint declares `dtype: bfloat16`. Across 48 layers, omitting
the official implementation's BF16 result boundaries can alter attention,
routes, residual streams, and logits materially.

The pinned `modeling_mimo_v2.py` is the semantic authority. It explicitly
computes RMS normalization in F32 and returns to the input dtype; computes RoPE
in F32 and returns Q/K to their original dtype; computes attention softmax in
F32 and returns probabilities to query dtype; accumulates router logits and MoE
weights in F32, then returns the expert aggregate to hidden-state dtype. Normal
BF16 tensor projections, elementwise operations, matmuls, residual additions,
and the LM head retain BF16 outputs under the checkpoint dtype.

## Contract

Before changing the endpoint, generate a deterministic PyTorch 2.13.0 fixture
covering finite F32-to-BF16 round-to-nearest-even conversion, including signed
zero, normals, subnormals, infinities, NaNs, exact halfway cases, and adjacent
bit patterns. Rust must reproduce every BF16 payload bit and widened F32 bit
for finite values, and preserve IEEE class/sign for non-finite values. Runtime
model tensors must fail closed on non-finite values.

The resulting conversion and two-case causal-attention fixture hashes to
`43e40b3d129394a49a254abdfe264f91afd3ad1eb8dd94695f353d7719e3f6fc`.

Apply BF16 boundaries only at source-authorized locations:

- RMSNorm results;
- FP8/BF16 projection results, including QKV and LM head;
- BF16 RoPE cos/sin and rotated Q/K;
- value scaling, attention score/scale/subtract, probabilities, and output;
- residual additions;
- SiLU, gate/up multiplication, and dense/expert projection results;
- the final F32 MoE weighted aggregate when returned to hidden dtype.

Keep router logits, sigmoid, correction bias, route weights, RMS variance, and
softmax's internal computation F32. Do not change weights, routes by policy,
topology, activation quantizer, cache policy, or acceptance thresholds.

## Success and kill criteria

Success requires exact fixture parity, all component gates, a safe complete raw
walk, and material movement toward the frozen hosted distribution. Promotion
to target-faithful still requires the PW-0052 chosen-token/top-20 limits.

Kill as the primary mismatch if hosted-token errors do not improve materially
or move oppositely without top-k agreement. Preserve source-authorized BF16
semantics as a correctness repair even if it does not independently establish
whole-model parity. Do not tune boundary placement against `Hello!`.

## Baseline

Clean `05c2bed437bdf6aa570848cbdd92fb77c071c4ed` with exact dynamic FP8
activation semantics generates `. -` for the frozen chat prefix. Hosted-token
logprob errors are 13.7936 and 7.4905 nats with 0/2 top-1 agreement.

## Result

The conversion fixture, two PyTorch BF16 causal-attention cases, all 27 Rust
tests, all 35 Python tests, strict Clippy, and the release build pass. The
pre-existing F32 causal-attention fixture remains a separate passing oracle;
the implementation did not weaken it to accept BF16 output.

Raw run 001 completed in 291.871 seconds and generated `[122046,13]` (`瀛.`).
Its report hashes to
`78ac1d3fcc94d28382df44c579dbba5b0c51d9216a6f623d2729f20f30c44b06`.

Frozen chat run 001 completed in 918.302 seconds and generated `[13,18]`
(`.3`) rather than hosted `[9707,0]` (`Hello!`). Relative to PW-0053, hosted
token 9707 error improved from 13.7936 to 12.8016 nats, while hosted token 0
error worsened from 7.4905 to 7.8176 nats. Top-1 agreement remained 0/2. The
chat report is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0054/chat-001/endpoint.json`,
SHA-256 `d6884788c0ae59125d7ddd3683040344b5106601b99357cc57da58bd69b62b96`.

The chat walk moved 82,872,061,696 logical source bytes and measured
84,124,729,344 process disk-read bytes. Minimum system free memory was 82%,
peak process residency was 4,440,047,616 bytes, swap did not grow, no new
throttled pages appeared, and every protected service remained alive.

## Decision

Keep the source-authorized BF16 boundaries as a correctness repair. Reject
their omission as the primary PW-0052 mismatch: movement was mixed and the
candidate remained orders of magnitude outside the hosted logprob limits.
Return to a line-by-line pinned-source/runtime audit before another full walk;
do not stack unproven rounding or tune against the hosted answer.
