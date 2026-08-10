# PW-0176 — MiMo 64K structured sparse-prefill layer-0 oracle

- Status: complete
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0160 deterministic
  long-context probe construction; PW-0175 structured-sparsity authority
- Execution mode: target-faithful source layer-0 Q/K/V and dense sampled
  outputs with shadow L3 released vertical-slash candidates; no candidate state
  enters the model
- Related records: PW-0158, PW-0160 through PW-0162, PW-0175; E7
- Implementation commit and dirty state:
  `71ff2992dd0bdd4332e105a8fe27e3fc8558a4d5`; clean at final fixture,
  runtime, and analysis

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
   Also report a deliberately noncausal best-pair-per-head-query oracle that
   chooses among the five released pairs using exact reference error.
9. Kill all combinations of the released pair family only if that favorable
   per-head-query oracle fails the same gates. Every fixed per-layer/head map is
   a restriction of this oracle, so a failure cannot be repaired merely by
   assigning different released pairs to different heads. It does not reject
   trained or repaired selectors with different widths or mechanisms.
10. If a uniform pair passes, promote it only to a deeper-global-layer and
    accumulated route/logit experiment. If only the favorable best-pair oracle
    passes, promote only a fixed headwise-assignment train/holdout experiment;
    do not treat exact-error choices as executable. In either case require
    diverse text plus native-modality traces and do not promote a kernel,
    runtime default, million-token capability, endpoint TPS, or purchase.
11. Enforce Gate 8 at fixture generation, QKV chunks, selector construction,
    reference/candidate evaluation, release, and final service health. Stop on
    less than 20% free memory, over 8 GiB process footprint/RSS, over 512 MiB
    swap growth, any new throttled page, missing protected service, or over
    4 GiB post-release footprint. Run no other full model process concurrently.

## Promotion and kill rule

Promote exact passing uniform parameter pair(s) to a separately named deeper
MiMo fidelity experiment. If only the favorable per-head-query oracle passes,
promote merely a fixed headwise train/holdout assignment test. The realized
layer-0 slice proves selector causality and a bounded local numerical result;
it does not prove accumulated model behavior or hardware speed.

If the source identity, causal path, sample completeness, full-selection
control, Gate 8, or serialization fails, reject the run as invalid rather than
adjudicating structured sparsity.

## Result

The first fixture at commit
`d32406fd07f94bc1963dfeb1dac5bc3ed8a0da5c` successfully renders and
round-trips exactly 65,536 token IDs. Its manifest hashes to
`9ab288863be5bf27b1339d803eae90cd7297be0b35bbd08e2981e5c50dbba4a5`;
the external token payload hashes to
`7a5c2d35b51d6a05b6d445d575bd08d68fed91a8997ec1e13cdc4c31e71cc507`.
Fixture Gate 8 passes at 70% minimum free memory, 300,010,368-byte maximum
physical footprint, 568,328,192-byte peak RSS, zero swap growth or
throttling, and resident protected services.

The corresponding source walk completed all 64 QKV chunks, 64 selector
heads, 1,536 sampled head-queries, and bit-exact full-selection controls. Its
raw manifest hashes to
`897a6ffe20863b6ecc64040d58fd5e8e930c99c759e1a29e4a0f8edd612adc9c`.
The frozen analyzer correctly refused to adjudicate it because it expected
one expanded BF16 matrix while the runtime ledger recorded zero: layer-0 RMS
uses a BF16 vector, and the only expanded matrix is the FP8 QKV weight. Repair
that ledger expectation with a direct regression test, regenerate the
commit-bound fixture, and rerun. Preserve `raw-001` as an invalid analyzer-
contract attempt and infer no structured-sparsity result from it.

The repaired execution is valid and deterministic. `fixture-002` hashes to
`d7c45847e2106a0ce5161a6e35fb87160888ea0eeebadf73b7040130ecd12526`;
`raw-002` hashes to
`1d6c4b4fd607fee439b170da0e26e4a9f1c380231a6baa47b009a7fd0061c9a9`;
analysis hashes to
`3176fed9199aba3d30ac1916d96ce1b8d5b55fbb005561b16b769873097da0da`.
The first and second source walks have identical token, QKV, sampled-query,
selector-query, key, value, pair, and complete observation payloads. The final
walk completes 64 bounded QKV chunks and all selector/sample work in
`273.078736` seconds. All 1,536 head-query identities are present and the
full-selection control is bit-exact for all 196,608 output values.

No uniform released pair passes. The widest `(1000,6096)` pair is strongest
overall and consumes `20.599935%` effective work, but its aggregate relative
L2 is `0.055171`, its maximum position error is `0.884388`, and head-query
p99 is `0.723112`, versus limits of `0.010000`, `0.020000`, and `0.050000`.
Its final-question band is promising at `0.008556` aggregate error, but the
early band is `0.258773` and the interval band is `0.030050`; the required
slice cannot be hidden by the favorable tail.

Even the noncausal best-released-pair-per-head-query oracle fails. It chooses
the lowest exact error independently for every evaluated head-query, yet
reaches only `0.047658` aggregate relative L2, `0.721474` maximum position
error, and `0.435570` p99. Because every fixed layer/head assignment of these
five released pairs is a restriction of that oracle, kill all such
combinations on the mandatory MiMo layer-0 64K slice. This does not reject a
trained or repaired selector with different widths or a different mechanism.

Gate 8 passes across 136 snapshots at 70% minimum free memory,
790,664,192-byte maximum physical footprint, 815,284,224-byte peak RSS, zero
swap growth or throttling, 23,102,976 bytes after release, and stable
protected services. The result records zero accepted tokens and no endpoint
TPS. No throughput-model constant changes: PW-0175's work fractions and the
`21.056139%` two-P100 ceiling reproduce; PW-0176 kills the released numerical
mechanism inside that allowance.
