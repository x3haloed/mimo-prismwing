# PW-0327 — Target-bonus q8 causal route pilot

- Status: complete
- Disposition: conditional
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0204, PW-0208, PW-0325, PW-0326

## Question

Does the promoted `target_bonus_full_match_v1` transaction behave correctly on
one real source-target q8 verification transaction in each of the ordinary,
code, multilingual, and rare-route text categories, and does that evidence
justify regenerating the complete 32-window route/economics corpus?

## Contract

Use the existing receipt-authenticated MiMo-V2.5 checkpoint, Apple M1, source
weights/routes, corrected QKV layout, and target self-proposer. This is a
target-faithful correctness/route pilot, not a fast proposer or endpoint run.

Run `arbitrary-text-route-trace` in a fresh process for each frozen PW-0208
prompt with a two-token diagnostic output bound. The prefill supplies the first
observable target anchor; the runtime must still execute exactly one complete
q8 proposal/verifier transaction, retain its full verifier-authorized vector
in the report, and clip only the observable output/cache prefix to the one
remaining requested token.

Prompt SHA-256 authorities:

- ordinary: `d15e7fad81828b710303ce5e9dc5fd9c2104450108eb627167e6bc2080b9ee5d`;
- code: `ad2940784d5028baa1dfab4585cb3a5a7fbffa22ca224f455fabc851549daefa`;
- multilingual: `6ece2dd3189d6b482f3356d344db6228e428db60a7530283eedc39be77d1beca`;
- rare-route: `5a71638364fff89af264dd3acea1ce31ef92128c3922cc8fb64826e793643373`.

For each report, authenticate the clean commit, model/checkpoint/kernel and
prompt identities, `target_bonus_full_match_v1` semantic, Apple M1, batch one,
concurrency one, cold process state, one transaction, exact eight-row verifier
trace at all 48 layers, full physical/logical byte ledgers, and Gate 8.

## Correctness and continuation gates

- Recompute the commit from proposal/posterior tokens. On convergence require
  the exact seven-token proposal suffix plus final target bonus, eight verifier
  rows, and the bonus as next-anchor authority. On mismatch require the
  unchanged first-correction rule.
- Require the single observable clipped token to be the prefix of the full
  verifier-authorized vector; require one retained output row while preserving
  the larger verifier-retention count in evidence.
- Recompute `U`, unique `(layer, expert)` identities, source bytes, `A`, and
  `A/U` from exact route rows. Here `A` means the full verifier-authorized
  transaction, not the intentionally clipped observable count.
- Require target-authorized token identity and all safety gates; make no TPS
  claim from a one-transaction diagnostic.

Authorize the complete four-category 32-window causal recapture if all reports
pass semantic/authority gates and at least three categories exercise the
full-match bonus branch. Otherwise preserve the exact mismatch and decide from
the observed `A/U`; never edit or project old reports.

## Claims excluded

- sustained or complete endpoint TPS;
- native-MTP proposal quality or latency;
- stale-route `A+1`, K4 fidelity, bank/cache construction, or a runtime
  performance default;
- multimodal/full-capability promotion, Prismwing-2, or Prismwing-50.

## Result

Four clean, cold-process Apple M1 captures ran from commit
`75d09e5e0e8ca9679faa5c29ed71445c9b741ad1`, one for each frozen prompt. The
command shape was:

```text
target/release/prismwing arbitrary-text-route-trace <checkpoint> \
  spec/model.lock.json <checkpoint-receipt> kernels/block_fp8_gemv.metal \
  <frozen-prompt> 2 <category>/report.json \
  75d09e5e0e8ca9679faa5c29ed71445c9b741ad1
```

Every report authenticates q8 target-self-proposal, seven single-token
proposal traces, one eight-row verifier trace, exact prompt/model/kernel
authority, batch one, concurrency one, accepted-token count two (prefill
anchor plus one output-clipped transaction token), and the repaired semantic.
The full verifier-authorized transaction results are:

| Slice | Full `A` | `U` | `A/U` | Logical transaction bytes | Physical transaction bytes | Complete wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 3 | 4.601063829787234 | 0.6520231213872831 | 171,746,143,232 | 171,934,068,736 | 442,421.709 |
| code | 8 | 4.672872340425532 | 1.7120091064314171 | 172,425,786,368 | 172,612,730,880 | 528,367.674 |
| multilingual | 8 | 4.226063829787234 | 1.8930144745122717 | 168,196,895,744 | 168,375,603,200 | 474,723.439 |
| rare-route | 8 | 4.901595744680851 | 1.6321215409658167 | 174,590,575,616 | 174,782,697,472 | 767,548.970 |

Code, multilingual, and rare-route fully converge and append target bonuses
2268, 33108, and 17588. Ordinary mismatches at proposal index two and follows
the unchanged first-correction rule. Across the four diagnostics, full `A` is
27, summed `U` is `18.40159574468085`, and `sum(A)/sum(U)` is
`1.4672640554993497`. Transaction-only logical/physical bytes total
686,959,400,960/687,705,100,288. Including prefill and proposal work, reports
record 2,658,630,945,792 logical and 2,661,595,774,976 physical bytes over
2,213,061.791 ms. Those totals are diagnostic and are not sustained TPS.

All runtime Gate 8 reports pass with at least 71% free memory, zero swap
growth, zero throttling, preserved protected services, maximum physical
footprint 206,791,424 bytes, and maximum peak RSS 421,920,768 bytes. No warm
run was measured.

Raw report/progress SHA-256 pairs:

- ordinary: `06a0c0322f47025d84d0f9e453c4ab4737816e0760037bc8433c64bd93ca1719` /
  `b35094c5ec53c978d5388b6477cf8f1b2f11f55844f7679df6caebef7bcbc526`;
- code: `83f9a37ae0da6e12b3289d70d3295539b0e4c67f8aaaa084cbcf0e1ef236910e` /
  `df941ef2989ffe3acfc88318ba55171622be5e0ed0c4b68b5152480ab24237cc`;
- multilingual: `1c726babd7f18ed93838daa5c0ed8f520eae56da8c731617026a5f3a554a9a71` /
  `bdb047a190641a175b122db8ab06e7e2c07aa8331fad31efdd3491c25670d066`;
- rare-route: `ffd414e2ef0dc71700758eb0522d479b712a01ce6c86aef0d06af24037367f47` /
  `7e786f4409c5de104a9d00bd2ba97194b731319389a8ffc2344a0f3a8085b5a6`.

The fail-closed analyzer at clean commit
`caa5406891c61362af09434ed0240062d4a18cfc` recomputes token commitment,
route unions, `A`, `U`, identity bytes, byte-ledger ordering, progress hashes,
and every memory bound. Canonical report:

`/Users/chad/Models/mimo-prismwing/evidence/PW-0327/analysis-001/analysis.json`

SHA-256:
`a54eeab1d136b938ddebe01a4206d6084bbeb2a2ca6a1395d88edfac337eaeed`.

## Decision

The predeclared continuation gate passes: all four reports authenticate and
three exercise the full-match target-bonus branch. Authorize a complete
four-category corrected-semantic causal recapture before recomputing the K4
envelope. Preserve PW-0208/PW-0325 as bonus-free historical evidence and do
not project these four diagnostic windows into sustained acceptance or TPS.
No runtime default changes.
