# PW-0112 — Wide teacher-forced route economics

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: pending
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  OpenRouter request `1fb4e9710958f352999b2301710c55eee8206e6f29d10f2707dbd8ee72285ad0`;
  response `9398c1f46f74d6e50be00c80746633ce74fb3cfc0f551659c8f011bb87326ae6`;
  capture manifest `9d0369870e5784324efaab5af710143b34a0b18e67e36d5b68f6299f2b8cee69`
- Hardware/runtime: Apple M1 shared 16 GiB host; internal SSD; source-FP8
  checkpoint; CPU target-faithful whole-prefix reference
- Exactness: L1 scheduling/cache analysis over unchanged source weights,
  token IDs, routing, top-k, route weights, and layer semantics
- Related records: PW-0044, PW-0091, PW-0102 through PW-0104, PW-0110,
  PW-0111

## Question and causal mechanism

PW-0110 proves that the measured internal-SSD source-FP8 acquisition floor
requires impossible-perfect `A/U >= 93.556` for the separately valuable 34.3
accepted-TPS horizon and `A/U >= 136.380` for 50 TPS. Those bounds assumed the
minimum expert union `U=1`. PW-0104's 27-position prompt trace is too short to
measure either required verification width or continuation reuse.

Freeze one 192-token greedy continuation from the pinned Parasail endpoint and
teacher-force it through the pinned local base in one causal whole-prefix pass.
Measure the target's actual expert union for wide verification and the best
possible value of a deliberately bounded shared expert cache. This tests
whether wide base-aligned speculation or 2--4 GiB residency can provide enough
source-expert-byte leverage to justify proposer training or cache-runtime work.
It does not assume that hosted and local source-FP8 arithmetic produce
identical logits; the hosted artifact supplies a frozen realistic suffix while
the local pinned base is the route authority.

## Frozen authority and runner

The request is `evals/fixtures/requests/pw0112-long-route-trace.json`: MiMo
V2.5, Parasail only, no fallback, required parameters, reasoning disabled,
temperature zero, 192 tokens, and top-20 logprobs. The successful capture has
87 prompt tokens and 192 continuation tokens. Authenticate its request,
response, and manifest hashes; exact provider/model identity; zero reasoning
tokens; finish reason `length`; position count; each selected-token byte
sequence; and the local tokenizer's exact prompt and continuation IDs.

Add one dedicated route-only fixture and command. It may reuse the established
CPU endpoint authority and `decode_step`, but must not weaken the three existing
fixture identities or write embeddings, 48 layer-final tensors, final norm, or
full logits. Run the concatenated 279 positions once with causal attention,
unchanged model semantics, no Metal candidate, no generated-token acceptance
claim, and no endpoint TPS claim. Preserve all 48 layer route traces, the
source-byte ledger, complete wall, exact fixture and checkpoint hashes, and
Gate 8 snapshots. Fail closed on unknown hashes, tokenization, token bytes,
shapes, route cardinality, expert IDs, non-finite route weights, or evidence
schema.

The route-only runner is a measurement topology, not a runtime architecture.
It introduces no service, cache, artifact bank, alternate router, shadow model,
or source-weight copy.

## Analysis protocol

Analyze only the 192 continuation positions for the primary claims. For every
sliding suffix window at `q = 8, 16, 32, 64, 94, 128, 137, 192`, compute each
routed layer's unique selected experts, normalized union
`U_layer = unique_experts / 8`, mean `U` across 47 layers, and the
impossible-perfect `A/U = q/U` with `A=q`. Report minimum, median, maximum, and
all window starts. Bind the resulting optimistic TPS ceiling to PW-0110's
2.727590151-second acquisition floor. This still makes draft execution,
attention, dense weights, compute, correction, rollback, and thermal effects
free.

Replay the continuation in causal token-major, layer-major, native-top-8 order.
Every `(layer, expert)` is a distinct equal-size 25,171,968-byte source-FP8
record. Report:

- reuse-distance and next-use distributions, compulsory misses, distinct
  layer-experts, expert frequency, and consecutive route-set persistence;
- exact LRU and offline-Belady curves at 2, 3, and 4 GiB;
- hit ratio, avoided logical bytes, logical miss bytes, and bytes per token;
- first-64-token calibration versus the held-out final 128 text positions; and
- deterministic access-list and analysis hashes.

Belady is an oracle upper bound, not a deployable policy. Logical hits are not
physical I/O savings or accepted TPS. This one text suffix does not satisfy the
project's eventual multimodal/held-out promotion corpus.

## Gates and dispositions

- **Authority:** all frozen hosted, tokenizer, checkpoint, request, response,
  route-shape, and evidence identities pass exactly.
- **Safety:** apply normative Gate 8 before checkpoint mapping, after each
  layer, after route evidence is serialized, after checkpoint release, and at
  final service-health readback. Stop if free memory falls below 20%, current
  or peak RSS/physical footprint exceeds 8 GiB, released footprint remains
  above 4 GiB, swap grows by more than 512 MiB, a throttled page appears, or a
  protected start-resident service disappears.
- **Wide speculation:** kill source-FP8 wide speculation for this held-out
  trace if the best impossible-perfect `A/U` at `q=94` is below 93.556 or at
  `q=137` is below 136.380. Continue to base-aligned proposer work only if a
  required width clears its PW-0110 physical bound before any omitted cost.
- **Cache:** kill a 2--4 GiB exact expert cache as a primary mechanism for this
  trace if 4 GiB offline Belady avoids less than 30% of selected expert bytes.
  A causal policy cannot exceed that oracle. Promotion remains forbidden
  without a repeated cold end-to-end gain and separate held-out modality
  traces even if this diagnostic gate passes.
- **Outcome scope:** a failure rejects these source-FP8 mechanisms on this
  frozen text trace. It does not reject speculation after executable-byte
  reduction, modified-mode route shaping, a different physical store, or a
  larger named hardware embodiment. It does not weaken the 34.3-TPS useful
  horizon or the formal 50-TPS target.

If the run stops safely or fails authority, preserve the partial manifest and
repair only the measurement path before interpreting route economics. If both
mechanisms fail their optimistic gates, return to executable-byte reduction
and the layer-transaction storage boundary rather than training a proposer or
building a cache runtime.
