# PW-0112 — Wide teacher-forced route economics

- Status: completed
- Disposition: scope-decision; wide source-FP8 speculation rejected on trace,
  bounded cache retained only as an unpromoted secondary experiment
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: `ffff868640457e6c927e39775c59911e796a9b41`
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

## Pre-result amendment after the first safe stop

The first execution at runtime commit
`64be9a2db15b2cd2479cf146a44050bcff98f959` stopped without emitting a
manifest after approximately 20 minutes because the source-directed router observed an
exact top-k boundary tie at global causal position 268. PyTorch documents that
indices of tied top-k elements are not stable across implementations/backends;
choosing the local C++ `nth_element` result would therefore weaken this
experiment's route authority.

No declared decision requires that ambiguous tail. Amend the executed prefix
to the first 137 of the 192 authenticated continuation tokens: 224 total
positions. This is the exact minimum width required by PW-0110's formal
50-TPS bound and contains complete windows at `q=94`, `q=128`, and `q=137`.
Measure the cache on the first 128 continuation positions, with the first 32 as
calibration and the following 96 as held-out text. The full 192-token hosted
capture, byte sequence, token IDs, and hashes remain the fixture authority;
the runner records both the full hosted suffix identity and the exact traced
prefix.

This is a scope reduction before any route-economics result exists, not an
acceptance-threshold change. The amended run must still fail closed on any tie
within the 137-token suffix, and the failed first attempt remains part of the
experiment record. The 4-GiB Belady kill threshold remains 30%, and the
`q=94`/`q=137` `A/U` thresholds remain 93.556/136.380.

## Result

The amended clean runtime at
`9647b740f8f19c075ec752ed044795fa20c1102a` completed the 224-position
causal walk in 1,312,791.400 ms. It authenticated the complete 192-token hosted
suffix, teacher-forced its first 137 tokens after the 87-token prompt, and
emitted every top-8 route and weight for all 47 routed layers. The process
expanded 3,173 distinct layer-local expert executions, accounted for
86,365,815,680 logical source bytes and 87,823,466,496 process-read bytes, and
made no endpoint-throughput claim.

The raw manifest at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0112/route-002/manifest.json`
hashes to
`584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e`.
Its route payload hashes independently to
`d6024840a97fd180aad17c39fef944da9a28db56bdc4de3301962b36c81923eb`.
The clean hash-pinned analyzer at
`35690c265c4cdd93979657b26a24e6f02dd38013` emitted
`/Users/chad/Models/mimo-prismwing/evidence/PW-0112/analysis-001/manifest.json`,
which hashes to
`e93d930549ee9fe761d7fc98bf59642088b3eb9f41c712968f8df26d5b2c8b98`.
The updated throughput model hashes to
`82005921bf0be529c093b4af055dca4991147d7e6b90e65c0f24de8e4aaa4e23`.

Wide source-FP8 verification misses both frozen physical bounds before any
draft rejection or omitted cost:

| Width | Observed mean `U` | Best impossible-perfect `A/U` | Optimistic TPS ceiling | Required `A/U` |
| ---: | ---: | ---: | ---: | ---: |
| `q=94` | median 2.125; range 2.019--2.301 | 46.567 | 17.072 | 93.556 for 34.3 TPS |
| `q=137` | 2.402 | 57.045 | 20.914 | 136.380 for 50 TPS |

At `q=137`, target routes touch 903 unique layer-expert records rather than
the impossible minimum 376. Even accepting every token with zero-cost draft,
attention, dense work, correction, and arithmetic cannot reach the separately
valuable 34.3-TPS horizon, much less 50 TPS, on PW-0110's unchanged cold
source-FP8 acquisition floor.

The cache result is more nuanced. Across the first 128 continuation positions,
there are 48,128 accesses to 895 distinct layer-experts. A 4-GiB/170-expert
offline Belady oracle reaches 44.716% hits and therefore clears this record's
deliberately low 30% continuation gate. Two and three GiB reach 22.286% and
33.369%. Causal global LRU reaches zero at every capacity because 170 slots
cannot span the 376 layer-expert accesses between adjacent tokens. A static
4-GiB frequency cache selected on the first 32 positions reaches 29.951% on
the following 96 held-out positions—promising as a secondary reduction, but
not promoted and nowhere near PW-0104's 93% primary-mechanism requirement.

Route sets are exactly identical across 57.838% of adjacent layer-position
pairs and have median intersection eight, explaining why an oracle/frequency
policy finds value despite global LRU's zero hits. The 4-GiB Belady oracle
still leaves 5.232 GB of logical source expert misses per token. Logical hits
do not establish physical page residency, cold wall reduction, or accepted
TPS.

Gate 8 passes all 53 boundaries: 78% minimum free memory, 805,994,496-byte
peak RSS, 712,235,008-byte maximum physical footprint, 49,645,504-byte final
footprint, zero swap growth, zero new throttled pages, and stable protected
services.

## Decision

Reject base-aligned proposer training and a wide verifier on the unchanged
source-FP8/internal-SSD representation for this held-out trace. PW-0044's
route-coherent selection premise cannot repair target-route union: the exact
greedy suffix itself supplies less than half the `A/U` needed for 34.3 TPS and
about 42% of that needed for 50 TPS under impossible-perfect acceptance.
Executable-byte reduction or a different physical store must change the
premise before reopening it.

Do not reverse PW-0104 or promote a cache runtime. Retain a 4-GiB
frequency/route-aware exact cache only as a secondary conditional experiment:
its oracle clears the 30% diagnostic gate and its held-out causal static policy
nearly matches it, but even the oracle leaves orders of magnitude too much
traffic for Prismwing 34.3/50 by itself. A later combined branch may build the
bounded artifact only if it freezes a physical cold end-to-end gain gate and
shows why the remaining bytes fit that branch's throughput budget.
