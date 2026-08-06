# PW-0051 — OpenRouter raw-prefix falsification

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes capture
- Checkpoint/processor/reference hashes: local revision
  `63651580ca774f8504f676040460aed3e1244ac1`; OpenRouter model
  `xiaomi/mimo-v2.5`; provider `Parasail`
- Hardware, OS, compiler, storage, memory pressure: external reference request;
  local comparison is PW-0050 runs 005/006 on the recorded M1 host
- Related records: PW-0001, PW-0050

## Hypothesis and mechanism

OpenRouter's OpenAI-compatible `/completions` surface can expose the pinned
Parasail MiMo endpoint at the exact raw `Hello` prefix used by PW-0050. If it
returns two greedy tokens and top-20 logprobs, the first hosted distribution
cheaply distinguishes a plausible raw continuation from accumulated local
semantic drift without first implementing multi-token chat prefill.

## Contract

The request pins `xiaomi/mimo-v2.5` and exactly one provider, disables
fallbacks and reasoning, requires parameters, uses temperature zero and two
tokens, and requests OpenAI-compatible `logprobs=20`. Request, response, and
manifest are immutable and content-hashed. The API key is read from the
owner-only configured file and never enters evidence or console output.

Pass: Parasail returns aligned token-level top-20 logprobs for raw `Hello`.
Compare its token strings/bytes against the pinned tokenizer and its first
top-20 distribution against PW-0050's first local top-20 logits.

Kill: the raw surface transforms the prompt, does not support this model or
provider, lacks top-20 evidence, or Parasail is unavailable. That kills this
cheap comparison only; it does not weaken the hosted gate. The required next
path is native chat serialization and multi-token prefill.

## Baseline and candidate

Baseline is the deterministic PW-0050 local output `[122046,13]` (`瀛.`) with
normalized semantic trace hash
`c695deef67ff4036a717472debc681125eefcc8d6485df068dc817658cc1b2a6`.
Candidate is `evals/fixtures/requests/raw-hello-smoke.json` through the extended
fail-closed immutable capture tool.

On 2026-08-05, the live endpoint inventory still listed Parasail as the only
MiMo route advertising both `logprobs` and `top_logprobs`, with FP8
quantization, but reported status `-2`; availability must be tested rather
than inferred from this metadata.

## Isolated attribution

Unexecuted.

## End-to-end result

Unexecuted.

## Correctness result

Unexecuted.

## Decision

Unexecuted.
