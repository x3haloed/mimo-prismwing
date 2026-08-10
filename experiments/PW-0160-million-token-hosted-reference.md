# PW-0160 — One-million-token hosted-reference viability

- Status: completed
- Disposition: conditional; inconclusive because the pinned provider was
  transiently unavailable
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; tokenizer and chat-template
  authority from `spec/model.lock.json`; OpenRouter model
  `xiaomi/mimo-v2.5`, provider `Parasail`
- Hardware/runtime: Apple M1 shared 16 GiB host for deterministic request
  generation and immutable capture; remote execution is reference acquisition
  only and cannot satisfy local inference
- Related records: PW-0001, PW-0051, PW-0052, PW-0112, PW-0158, PW-0159;
  TARGET Sections 3.2, 4, 7.2, and 8
- Implementation commit and dirty state:
  `6858fbb0d65ad10da774c23c5a11dfcb719285b0`, clean

## Question and causal mechanism

The project requires a one-million-token local smoke case and compares every
passing modified mode against a frozen OpenRouter whole-model reference.
PW-0001 proved top-20 logprobs on short prompts, while current public endpoint
metadata advertises a 1,048,576-token Parasail context. Neither proves that the
same pinned endpoint accepts a true one-million-token chat prefix, preserves an
early dependency, and returns token-aligned top-20 logprobs without truncation.

Prove that answer-key boundary before implementing changed attention. Generate
one deterministic chat request whose complete source-template serialization is
exactly 1,000,000 tokenizer tokens. Place a content-addressed random code near
the beginning, fill the middle deterministically, and ask for only that code at
the end. Pin Parasail, disable fallbacks and reasoning, require all parameters,
request greedy output plus top-20 logprobs, and preserve the raw response. This
is a transport, accounting, and reference-evidence preflight. It is not a final
long-context capability fixture and cannot promote any local architecture.

## Construction and evidence contract

1. Authenticate `TARGET.md`, `spec/model.lock.json`, the local `tokenizer.json`
   and `tokenizer_config.json`, and the exact MiMo revision before generating
   content. Fail closed on any hash, template, model slug, provider, or schema
   drift.
2. Freeze a generator seed and derive the needle code from SHA-256 rather than
   writing a hand-selected answer. Put the declaration within the first 256
   serialized tokens and the question within the final 256. The generator may
   use deliberately repetitive neutral padding because this is a viability
   preflight, but it must name that limitation and must not call the result a
   representative long-context benchmark.
3. Render with the pinned source chat template, `add_generation_prompt=true`,
   and thinking disabled. Locally tokenize the complete rendered prefix and
   require exactly `1,000,000` token IDs. Decode/re-encode must reproduce every
   ID. Record the first and last 256 IDs, complete token-ID SHA-256, rendered
   byte count/hash, needle offsets, and request hash without committing the
   large request.
4. Freeze the public OpenRouter models and endpoint responses before the paid
   call. Require the selected row to identify `xiaomi/mimo-v2.5`, Parasail,
   FP8, context at least 1,000,000, and both `logprobs` and `top_logprobs`.
   Endpoint metadata is moving evidence; preserve its raw bytes and capture
   time rather than silently refreshing it.
5. The request must use exactly one provider (`Parasail`), fallbacks disabled,
   `require_parameters=true`, reasoning disabled, `temperature=0`,
   `logprobs=true`, `top_logprobs=20`, `stream=false`, and at most 16 output
   tokens. Never serialize the API key. Keep request, response, metadata, and
   manifests outside Git.
6. Preserve every attempt, including HTTP errors, timeouts, empty completions,
   provider drift, and malformed logprobs. Permit at most three paid attempts;
   stop once one passes. The declared experiment spend ceiling is `$0.50`,
   below the owner's separately capped reference budget. Record provider usage,
   billed prompt/completion tokens, reported cost, elapsed wall, finish reason,
   cached-token accounting, and request IDs when exposed.
7. A passing response must report exactly `1,000,000` prompt tokens, identify
   Parasail, contain the derived needle code and no extra answer text after
   normalization, expose zero reasoning tokens, and provide at least 20 finite
   alternatives for every visible completion token. Offline verification must
   reproduce all request/response/metadata hashes and token alignment.
8. Apply Gate 8 to request generation, capture boundaries, and offline
   analysis. Record zero local accepted tokens and no local endpoint TPS. The
   hosted elapsed time and provider throughput are reference-path diagnostics,
   never Prismwing performance.

## Promotion and kill rule

Pass only if one preserved attempt satisfies every identity, exact-token-count,
needle, logprob, reasoning, and offline-integrity condition. Passing promotes
only the one-million-token hosted answer-key path and authorizes a separately
contracted changed-attention falsification experiment.

If all three attempts fail for the same reproducible provider capability reason
after endpoint metadata advertised support, mark the one-million-token hosted
distributional reference not proven. Do not waive or weaken TARGET; stop that
validation branch for an explicit semantic-authority decision. A transient
network or provider outage is inconclusive and does not justify three blind
paid retries.

Even a passing needle result is not the final one-million-token capability
slice. It uses artificial padding, one retrieval dependency, and hosted remote
execution. It proves neither native local inference, representative quality,
changed-attention fidelity, 30-minute TTFT, nor 1 accepted TPS.

## Result

The deterministic generator produced the same exact request at all three clean
capture commits. The rendered source-template prefix contains exactly
`1,000,000` pinned tokenizer IDs, round-trips every ID, and hashes to
`fd155578b24bbbe1c8ab7edea17ea615a2f907fae3709a59a88b7548a8810b92`
as little-endian `u32` IDs. Needle `PW-75FC1F69C84D` begins at token 32 and
the final question begins at token 999,973. The canonical 2,000,390-byte API
request hashes to
`a21c154c87bb2ce0f3c3305b52655cac04538fdfa4224b36f73e96503167048b`.

Frozen public metadata identified Parasail FP8 with context 1,048,576,
`logprobs`, and `top_logprobs`; the model and endpoint payloads hash to
`ce8154b4ee4ae42f5e14c071847a5acc35c9e89e69a42e84f4b8676d2cd3133e`
and `09cd75b1b2e4d053f99e1f13be64c680ec6de0db3ab2e36642af475d1e9e9033`.
No paid attempt reached model output:

1. The first returned HTTP 200 with JSON error code 502. Its preserved body
   hashes to
   `e242aee3ab74e8e3d41242e6faa7c65e59160e5e5a1bc5696b725368e6fdfd85`.
2. After classifying that wrapper correctly and regenerating at a clean commit,
   the second returned HTTP 429. Its body explicitly identifies Parasail,
   `upstream_provider_shared_pool`, and `Retry shortly`; it hashes to
   `9084455a85d6b17206d1f623195ddb3058561afc2c46a9deb3c771142123cac2`.
3. After a multi-minute cooldown and a final clean regeneration, the third
   returned the byte-identical 429 body. No fourth request is permitted.

All three provider errors omit usage and cost, so reported spend is `$0.00`;
that is not an independent account-ledger claim. Even charging all three full
one-million-token prompts and all 16 possible output tokens at the frozen
rates gives a `$0.42001344` worst case, within the `$0.50` contract.

The consolidated authoritative report hashes to
`635748d36a1fc6d690d0261c3526519f5b1bc558745cb4dc574432369f133048`.
Across preparation and attempt processes, Gate 8 has 47% minimum free memory,
1,230,258,176-byte maximum peak RSS, 330,927,936-byte maximum physical
footprint, zero swap growth, zero new throttled pages, explicit release
boundaries, and stable protected services. The analyzer separately passes at
52% minimum free memory and a 264,587,584-byte maximum physical footprint.

## Decision

PW-0160 neither passes nor reaches its capability kill condition. The observed
sequence `502, 429, 429` consists only of transient upstream/provider-pool
classes. No response exposed a prompt-token count, completion, logprobs,
truncation behavior, or needle answer, so it would be false to interpret these
errors as evidence that Parasail cannot execute the advertised context.

Keep the one-million-token hosted reference unproven. Do not authorize the
changed-attention validation branch from this experiment, do not weaken
TARGET, do not switch providers silently, and do not retry under this bounded
contract. A future attempt requires a new record and either demonstrated
Parasail availability or explicit authority for a new frozen reference epoch.
The result records zero local accepted tokens, no local endpoint TPS, and no
throughput-model constant change.
