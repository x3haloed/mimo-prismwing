# PW-0052 — Hosted chat and native multi-token prefill

- Status: complete
- Disposition: rejected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes hosted capture and runtime change
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; tokenizer config
  `fd34b805f75a890a5c123d79a2982bbe240b3b6efb156d22401bd619484d9bd2`;
  OpenRouter `xiaomi/mimo-v2.5`, provider `Parasail`
- Hardware, OS, compiler, storage, memory pressure: local M1 host and safety
  contract inherited from PW-0050; hosted request records its own metadata
- Related records: PW-0001, PW-0050, PW-0051

## Hypothesis and mechanism

The deterministic PW-0050 walking foundation can execute the checkpoint's
exact chat-template token prefix as one causal prefill, then retain every
layer's K/V state for incremental greedy decode. Comparing the local first two
positions with a frozen Parasail chat response will reveal whether the odd raw
continuation was merely an unsupported raw prompt or evidence of accumulated
whole-model semantic drift.

## Contract

Freeze a user-only `Hello` request, reasoning disabled, temperature zero, two
visible tokens, top-20 logprobs, one pinned provider, required parameters, and
no fallbacks. Preserve immutable request/response/manifest hashes before local
implementation changes.

Serialize locally from the pinned template with its default system message,
user turn, assistant generation marker, and reasoning-disabled empty think
block. Native tokenization must match an independently generated fixture. The
runtime must process every prompt token under causal attention in one prefill,
execute every model layer exactly once for that sequence, take the last
position's full LM-head logits as the first-token distribution, and then use
retained K/V for the second token. It must retain PW-0050's source identity,
arithmetic, trace, determinism, and shared-host safety gates.

Pass: hosted capture verifies; local prompt IDs match the frozen serialized
chat; two clean local processes agree; first two greedy token IDs match hosted,
and hosted-token logprob/top-20 comparison is within TARGET Section 5's
position-level limits.

Kill: unavailable hosted evidence leaves parity not proven. A local mismatch is
preserved and localized by first divergent position, top logits, route traces,
and component fixtures; it cannot be waived because output is readable.

## Baseline and candidate

Baseline is PW-0050 `5345aa6`, which supports only the raw one-token prompt and
produces deterministic `瀛.`. Candidate extends that single Rust authority to
multi-token prefill without changing weights, routing, attention, or numerical
mode.

## Isolated attribution

The hosted request completed in 0.619 seconds and cost USD 0.00000254. It used
27 prompt tokens and two completion tokens, with zero reasoning tokens. These
are external-reference diagnostics, not local performance measurements.

## End-to-end result

Parasail returned visible text `Hello!` and finish reason `length`. The first
token `Hello` had logprob -0.0611581; the second token `!` had logprob
-0.0168504. Both positions contain exactly 20 alternatives. The immutable
capture verifies offline; its manifest hashes to
`f9c5dd42a76e0eb87581fa427fe03c69ad32903c5711e5078a002ab7514732ea`.

Local run 001 used committed candidate `fe367df`. It processed the exact 27
prompt IDs in one causal prefill, retained cache length 27 and then 28, and
executed 8,070 FP8 matrix expansions and 2,656 routed experts. Complete wall
was 874.985 seconds; logical source traffic was 82,343,450,368 bytes and
process disk reads were 83,588,468,736 bytes. The raw local report hashes to
`c312b859b4d6dba5b1daecf4e553e8e06040c09bffe7d431aee5bca67db649c7`.

## Correctness result

Hosted acquisition passes. Request SHA-256 is
`50459a8b9d142947c34fbf819fb0e3fb4796cbbfc08a04cf253ad4cc08c70e48`;
response SHA-256 is
`e5a8956f3a7985e1ac3d5396c7bc9fe73bc77c6451eb2225c8df7c8973e3212d`.
Local identical-prefix parity fails decisively. Local greedy output was `.3`
with token IDs `[13,18]`, versus hosted `Hello!` `[9707,0]`. At position zero,
local logprob for hosted `Hello` was -13.5981 versus hosted -0.06116, absolute
error 13.5370 nats. At position one, local logprob for hosted `!` was -8.0170
versus hosted -0.01685, absolute error 8.0002 nats. Both exceed the target by
orders of magnitude and top-1 agreement is zero of two.

The safety gate passes: peak residency was 4,355,833,856 bytes, maximum phase
footprint 3,212,692,032 bytes, minimum system-free memory 84%, and swap growth
and new throttled pages were zero. Thus memory or batching failure does not
explain away the semantic result.

## Decision

Reject the PW-0050/PW-0052 direct-F32 FP8 execution mode as a whole-model
answer authority. The pinned checkpoint declares block-FP8 weights with
dynamic activation quantization, while the current endpoint dequantizes weights
and multiplies unquantized F32 activations. Open-source FP8 authorities specify
per-token-per-128-channel online activation quantization for this exact config.
Open a correctness repair for that omitted semantic before considering BF16
boundary rounding or performance work.
