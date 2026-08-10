# PW-0176 — MiMo 64K structured sparse-prefill layer-0 oracle

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0160 deterministic
  long-context probe construction; PW-0175 structured-sparsity authority
- Execution mode: target-faithful source layer-0 Q/K/V and dense sampled
  outputs with shadow L3 released vertical-slash candidates; no candidate state
  enters the model
- Related records: PW-0158, PW-0160 through PW-0162, PW-0175; E7

## Realization and compression contract

Capability: on one real, deterministic 65,536-token MiMo chat prefix, carry
the pinned token IDs through source embeddings, source RMS normalization,
source layer-0 QKV projection, RoPE, the released online vertical/slash
selection rule, and candidate attention to observable per-head output errors.

Envelope: batch one, one target-faithful global layer, all 64 query heads, all
released MInference vertical/slash parameter pairs, frozen sampled causal query
positions, bounded phase memory, the shared M1 host, and no concurrent full
model process. The artificial needle/padding prompt is a required long-context
viability slice, not representative language or modality coverage.

Accepted revisions: the candidate is L3 and may renormalize attention over
only its selected causal rows. Source Q/K/V, positions, RoPE, dense reference,
and projection numerics remain authoritative. No error threshold changes after
execution.

Exclusions: this does not execute layers 1–47, routes, logits, generation,
native modalities, a million-token prefix, a Metal/P100 sparse kernel, or an
endpoint. It cannot promote fidelity, performance, or hardware.

Evidence horizon: the local M1 source path, frozen checkpoint, deterministic
64K fixture, raw per-head observations, phase counters, and content-addressed
manifest. The risk frontier is whether the released online selector—not an
exact top-probability oracle—can preserve a real MiMo source attention output
inside the complete-system work allowance.

Central truth: a deterministic real prompt causes authoritative MiMo layer-0
Q/K/V; the authenticated last-64-query selector chooses causal key/value rows;
those rows cause a bounded candidate output whose error and physical work are
recorded without affecting source state.

## Contract

1. Authenticate TARGET, model lock, checkpoint receipt, tokenizer and template,
   PW-0160's deterministic probe generator authority, PW-0158, PW-0161,
   PW-0175, and the immutable MInference source capture by SHA-256. Fail closed
   on revision, layout, tensor, tokenizer, or source drift.
2. Before the real run, add a tiny deterministic fixture for:
   - last-64 causal score construction and F32 softmax;
   - vertical sums with the first 30 sink positions forced selected;
   - diagonal/slash sums with the most recent 30 diagonals forced selected;
   - descending selection with lower-index tie choice;
   - causal vertical/slash union, compact original-position order, and
     renormalized value reduction;
   - bit-exact dense/full-selection control.
3. Reuse PW-0160's pinned chat template, tokenizer, seed, framing, and filler
   construction to render exactly 65,536 token IDs. Require exact
   decode/re-encode roundtrip, needle in the first 256 tokens, question in the
   last 256, and a frozen little-endian-u32 token hash. Store the large token
   payload outside Git with a content-addressed manifest.
4. Execute only source layer 0. Stream embeddings, RMS normalization, QKV
   projection, and RoPE in bounded chunks. Retain all four K/V heads, only the
   final 64 queries needed by the online selector, and the frozen sampled query
   rows. Never materialize all 64K×64 query vectors or an attention matrix.
5. Sample all 64 heads at fixed early, interval, and final-question positions.
   Compute the authoritative dense source head output with source BF16
   boundaries. Require completeness, finite values, stable token/Q/K/V hashes,
   and a bit-exact dense/full-selection control.
6. Reproduce the released MInference online selector from the authenticated
   source: last 64 queries against all keys, F32 probability aggregation,
   vertical and slash top-k construction, then sparse attention over their
   causal union. Evaluate every unique released parameter pair from PW-0175's
   GLM-4-9B-1M configuration. Do not import its layer/head assignment or use
   exact source attention probabilities to select rows.
7. For each parameter pair, independently compute selected causal-pair work at
   `N=65,536`, add the last-64 index-QK cost, and require effective work no
   greater than PW-0175's reproduced `21.056139043683178%` ceiling before the
   pair can pass numerically.
8. A pair's phase-A numerical gate requires aggregate relative L2 at most 1%,
   per-position relative L2 at most 2%, and head-query relative-L2 p99 at most
   5%. Report maximum error, selected-row distributions, early/interval/final
   bands, and every candidate; do not conceal a failing band in the aggregate.
9. Kill this exact layer-0 continuation if no released pair passes both work
   and numerical gates. A failure at mandatory layer 0 rejects the audited
   MInference-style configuration for this 64K slice, but not trained or
   repaired selectors with changed mechanisms.
10. If any pair passes, promote only a deeper-global-layer and accumulated
    route/logit experiment on diverse text plus native-modality traces. Do not
    promote a kernel, runtime default, million-token capability, endpoint TPS,
    or purchase.
11. Enforce Gate 8 at fixture generation, QKV chunks, selector construction,
    reference/candidate evaluation, release, and final service health. Stop on
    less than 20% free memory, over 8 GiB process footprint/RSS, over 512 MiB
    swap growth, any new throttled page, missing protected service, or over
    4 GiB post-release footprint. Run no other full model process concurrently.

## Promotion and kill rule

Promote only the exact passing parameter pair(s) to a separately named deeper
MiMo fidelity experiment. The realized layer-0 slice proves selector causality
and a bounded local numerical result; it does not prove accumulated model
behavior or hardware speed.

If the source identity, causal path, sample completeness, full-selection
control, Gate 8, or serialization fails, reject the run as invalid rather than
adjudicating structured sparsity.

## Result

Unexecuted. No conclusion may be drawn from this record yet.
