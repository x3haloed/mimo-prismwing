# PW-0092 — Repaired two-token incremental decode

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: `e417b099677b5bfc4b9a78e110ab847f9035de0e`,
  clean contract commit
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

Two clean processes completed the real prompt-to-token-to-cache-to-token path.
Both produced token IDs `[264, 13]`, text ` a.`, 48 layer traces per step, and
complete finite 152,576-value logit vectors. Step one consumed all 27 prompt
tokens and left every layer cache at length 27. Step two consumed only token
264 and left every cache at length 28. Its output token was 13. The two runs'
semantic projections are exactly equal, including identity, inputs, emitted
tokens and text, both full logit vectors, top logits, selected expert order,
route weights, cache lengths, `U`, execution contract, and logical-work
ledger. Both projections have SHA-256
`dd7ef51e2fe6104bc39132a5b41ff79126a0d581fd2ec3b2fcc35992df532898`.
Timing, process-disk counters, RSS, and safety snapshots were intentionally
excluded from this equality projection and compared separately.

Step one's packed F32 logits are byte-identical to PW-0091
`last_logits.f32`; both hash to
`c43be0909487235bddfe6e0de69aa42a98339faf43cd6b77d6ef4b5f1a853cab`.
Its layer identities, attention modes, selected experts, route weights, cache
lengths, and `U` are also exactly equal to PW-0091. This closes the causal gap
between the exact full-prefix trace and the accepted-token endpoint.

Run 001 completed in 951,253.685 ms: 792,212.067 ms for prefill/first token
and 158,521.015 ms for the retained-cache incremental token. Run 002 completed
in 955,390.419 ms: 796,251.550 ms and 158,614.709 ms. Thus this walking slice
is deterministic but extremely slow; its second-step diagnostic rate is only
about 0.0063 token/s and is not accepted TPS. The incremental step spends
151.56--151.65 seconds inside the 48 layers and about 6.96 seconds outside
them. Every routed layer executes eight experts for its single position. The
complete two-step ledger records 84,181,004,032 logical source bytes, 8,289
FP8 matrix expansions, 98 BF16 matrix expansions, and 2,729 routed-expert
executions. Replaying the ledger formula against the report's exact routes and
verified Safetensors headers assigns 66,973,098,880 logical bytes, 7,110 FP8
expansions, and 2,353 unique-expert executions to prefill; the incremental
step assigns 17,207,905,152 logical bytes (16.026 GiB), 1,179 FP8 expansions,
and 376 expert executions. The two partitions sum exactly to the runtime
ledger. Actual process disk reads were 85,457,002,496 bytes in run 001. The
current embodiment therefore rematerializes all eight selected experts in
every routed layer plus the shared spine for each decoded token.

The frozen hosted response remains behaviorally different. On the identical
first prefix it chooses token 9707 (`Hello`) at logprob -0.0611581, while the
exact local source checkpoint chooses token 264 (` a`); local logprob for
token 9707 is -12.599658, an absolute difference of 12.538500 nats. Hosted
step two is not distribution-aligned after that first-token divergence, so
the local projection of token 0 at the `[... ,264]` prefix is preserved only
as a diagnostic and is not presented as hosted parity evidence.

Raw report hashes:

- run 001:
  `18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25`
- run 002:
  `ee1151c7a780545df922593b04e4e1c304541824a7a4d761ce42cdab70fa8078`

Gate 8 passed in both processes. Minimum system memory-free percentages were
79% and 78%; peak RSS was 4,373,823,488 and 4,338,974,720 bytes; maximum
physical footprints were 3,178,568,960 and 3,180,224,320 bytes; final
footprints were 3,163,021,056 and 3,090,915,392 bytes. Neither run grew swap
or added throttled pages, and ChatGPT, WindowServer, both `nxnode` processes,
and both Syncthing processes remained resident through the final boundary.

## Decision

Promote the repaired two-token endpoint as the slow target-faithful walking
slice for text generation semantics. Preserve the hosted result as an
unexplained serving-behavior divergence; do not tune local semantics against
it or weaken the hosted gate. Do not promote any performance default or TPS
claim.

The next bounded work is an independent incremental-versus-whole-sequence
correctness check, followed by profiling the one-token path. The first
performance hypothesis should attack repeated selected-weight loading and
FP8-to-F32 expansion as an embodiment problem; layer arithmetic, KV retention,
and the exact source path are no longer speculative gaps.
