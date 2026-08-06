# PW-0092 — Repaired two-token incremental decode

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0091 comparison
  `34f1d6e28622d66409d46e7407a9e54532e03821ea7dd36e65e94b50045216db`;
  frozen hosted manifest
  `f9c5dd42a76e0eb87581fa427fe03c69ad32903c5711e5078a002ab7514732ea`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust endpoint, retained per-layer K/V
- Related records: PW-0050, PW-0052, PW-0089 through PW-0091

## Shared construction contract

Capability: one real serialized chat prompt must cause the pinned tokenizer and
checkpoint to produce an accepted greedy token, retain authoritative K/V state
in all 48 layers, consume that state for a second incremental token, and expose
both tokens, text, logits, routes, timings, bytes, and safety measurements from
one Rust process.

The semantic authority is the now bit-exact source-checkpoint path proven by
PW-0091. The frozen OpenRouter response is a separate behavioral reference and
may disagree; no prompt, template, provider, threshold, or token is changed to
manufacture agreement. The evidence horizon covers this 27-token text-only
prompt, greedy two-token decode, retained caches, and the shared M1 host. It
does not establish sampling, long context, modalities, or accepted TPS.

Topology: tokenizer interpretation, model state, caches, routing, selection,
and emitted tokens remain in the existing single Rust endpoint authority.
Component traces and hosted JSON are evidence, not alternate accepted-token
authorities.

Embodiment depth: use the verified source mmap, bounded per-matrix expansion,
Accelerate, CPU arithmetic, and in-process K/V already authorized. Do not add
another runtime, cache representation, speculative scheduler, or modified
weights before this causal slice is measured.

## Hypothesis and gates

The repaired endpoint should complete both steps deterministically and safely.
The first step must reproduce PW-0091's exact source-checkpoint distribution
and greedy token 264. Every layer cache must retain 27 positions after prefill
and 28 after the incremental token. The second step must consume one new input
row rather than recompute the 28-token prefix, emit a finite complete logit
vector, and leave all caches at 28 positions.

Record the frozen hosted token/logprob comparison without treating a mismatch
as a local component defect. Run two clean processes only after the first run
passes the causal and safety gates. Compare token IDs, text, full logits, route
sets, cache lengths, hashes, and output bytes. Record cold/warm state, complete
wall, per-step wall, logical and actual bytes, batch 1, concurrency 1,
accepted tokens 2, `A=1`, and every layer's `U`.

Enforce normative Gate 8 at checkpoint open, every layer, LM head, and accepted
token boundary. Preserve stopped evidence. This cannot count as accepted TPS
or alter any correctness, hosted, capability, cost, power, or performance gate.

## Result

Unexecuted.

## Decision

Unexecuted.
