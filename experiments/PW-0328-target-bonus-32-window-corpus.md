# PW-0328 — Target-bonus 32-window causal corpus

- Status: complete
- Disposition: conditional
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0208, PW-0325, PW-0326, PW-0327

## Question

What are the real chronological q8 acceptance, route-union, and byte values
after the repaired full-match target bonus across eight windows in each frozen
ordinary, code, multilingual, and rare-route category, and can those windows
form a complete native-MTP history authority for the next Prismwing-1
falsifier?

## Prediction being resolved

PW-0208 and PW-0325 are exact for their historical bonus-free transaction
boundary, but the target bonus changes the next generated anchor and therefore
can change every later proposal, route, `A`, and `U`. PW-0327 proves the repair
works in one transaction per category and passes its predeclared continuation
gate. It does not authorize projecting `A+1` or reusing later legacy routes.

The fixed 64-token capture may now contain only eight transactions when every
q8 proposal converges. The old corpus builder skipped transaction zero because
it bound the proposal anchor only to a preceding verifier window. That premise
is superseded for a complete-history corpus: transaction zero has the exact
preceding target-hidden authority in the already authenticated prefill
artifact. Its target hidden history is every serialized-prompt row, its target
input IDs are the serialized prompt IDs, and its MTP layer-zero input IDs are
the prompt IDs shifted left with the first target anchor appended.

## Frozen inputs and execution

Use the receipt-authenticated internal MiMo-V2.5 checkpoint, `spec/model.lock.json`,
`kernels/block_fp8_gemv.metal`, Apple M1, batch one, concurrency one, q8 target
self-proposal, and the four PW-0208 prompt files and hashes frozen in PW-0327.
Companion hardware is inadmissible.

Run one clean cold process at a time from the committed contract/runtime:

```text
target/release/prismwing native-mtp-window-capture <checkpoint> \
  spec/model.lock.json <checkpoint-receipt> kernels/block_fp8_gemv.metal \
  <frozen-prompt> <category> verification-layer47-hidden.f32 report.json \
  <clean-capture-commit>
```

Preserve raw reports, progress logs, and hidden payloads outside Git. Before
each generation capture, run a fresh prefill hidden capture from the same clean
commit, model receipt, kernel, and prompt:

```text
target/release/prismwing native-mtp-prefill-capture <checkpoint> \
  spec/model.lock.json <checkpoint-receipt> kernels/block_fp8_gemv.metal \
  <frozen-prompt> <category> target-layer47-hidden.f32 report.json \
  <clean-capture-commit>
```

Authenticate its report/payload hashes, prompt tokens, serialized prompt,
first anchor, model/checkpoint/kernel identities, shape, finite payload, safety
evidence, and exact agreement with the generation report's prompt/chunk/anchor
authority. The historical PW-0208 prefill reports remain immutable evidence,
but their older kernel cannot authorize PW-0328 transaction-zero hidden state.

## Correctness and evidence gates

For every new source report, require the repaired semantic, clean capture
commit, exact model/checkpoint/tokenizer/kernel/prompt identities, Apple M1,
cold state, requested and accepted token count 64, requested-maximum stop,
batch one, concurrency one, at least eight chronological q8 transactions,
complete route and proposal traces, progress-hash closure, hidden-artifact
hash/shape closure, positive ordered logical/physical byte ledgers, and every
normative Gate 8 memory/service condition.

Require the sum of all transaction logical and physical ledgers not to exceed
the corresponding complete report ledger. Per-transaction ordering alone is
insufficient.

Recompute each transaction commit from proposal and posterior tokens. A full
match authorizes the seven-token proposal suffix plus target bonus and eight
verifier-retained rows. A mismatch follows the unchanged first-correction rule.
Distinguish full verifier-authorized `A` from terminal output clipping; the
last selected window may expose only a prefix to satisfy the 64-token bound.

Select transactions zero through seven in each category. Recompute `U`, exact
layer-qualified identity lists, source expert bytes, proposal/verification
walls, full `A`, observable emitted tokens, and all byte ledgers. For native-MTP
history:

- transaction zero binds to the final row and full history of the authenticated
  prefill target-hidden artifact;
- later transactions bind to the final retained row of the preceding verifier
  transaction;
- every history segment, target input ID, shifted MTP input ID, and anchor must
  close exactly.

Add deterministic fixtures for transaction-zero prefill binding, later-window
binding, full-match target bonus, mismatch correction, terminal clipping, and
eight-transaction fixed-64 selection. Preserve the legacy PW-0208 builder and
manifest unchanged under their original names.

## Rejected provenance attempt and repair

The first ordinary launch from clean detached commit
`20457474d00911354a5b4415abd9a8f21c2a02a5` was manually stopped before
prefill completed. Source review found that `native-mtp-window-capture`
recorded the supplied commit and dirty flag but, unlike resident and external
native-MTP generation, did not require the supplied commit to equal clean Git
`HEAD`. The launch was in fact clean, but the raw format was self-asserted and
therefore inadmissible for this experiment.

The empty progress artifact is preserved at SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
its rejection note hashes to
`8d8c4551d8714a0d27756f66168f45f5f9c99aa5b3d12df35ae4b37c933342a8`.
It produced no report, hidden payload, accepted token, or measurement.

The runtime now includes native-MTP window capture in the exact-clean-HEAD
gate and has a dedicated fixture covering ordinary, resident, external-MTP,
and capture evidence modes. All canonical captures must start after that
repair from one clean commit; none may reuse the rejected artifact.

Independent builder review then found that the proposed reuse of PW-0208
prefill hidden could not prove current hidden equality: its report used an
older kernel, while the new generation report exposes only prompt/chunk/anchor
agreement, not prefill hidden bytes. PW-0328 therefore requires fresh
same-commit prefill captures and rejects the historical payloads as current
transaction-zero authority. This changes evidence acquisition only, not model
semantics or thresholds.

## Continuation and kill gates

Publish a new, distinctly named 32-window manifest only if all four sources and
all 32 primary windows pass. Then authorize a fresh Prismwing-1 K4/cache
envelope computation from these exact `A`, `U`, identities, and routes.

Kill the regenerated-envelope branch if any category cannot supply eight valid
chronological windows, if transaction-zero history does not close against the
prefill authority, or if any semantic, byte, hidden, or safety gate fails. Do
not substitute a stale legacy window or projected bonus.

## Claims excluded

- sustained or complete endpoint TPS;
- native-MTP proposal latency, accelerated proposal execution, or general
  runtime promotion;
- K4 fidelity, construction, cache residency, common-weight cost, or achieved
  Prismwing-1 throughput;
- multimodal/full-capability promotion or any companion-hardware branch.

## Result

The canonical manifest is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/manifest.json`,
SHA-256
`36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403`.
It was built from clean detached capture commit
`26d2ea31852c0d63bd022df6d571fd722137c39f` and authenticates 24 bound
artifacts: generation report, progress log, verifier hidden, prompt, fresh
prefill report, and fresh prefill hidden for each of the four categories.

All 32 primary windows pass exact raw-route reconstruction, chronological
transaction ordering, verifier-authorized commit replay, causal hidden-history
binding, target/MTP token-history closure, byte-ledger ordering, and Gate 8.
The corrected corpus totals full verifier `A=232`, observable `A=231`,
`sum(U)=142.71808510638297`, and `sum(A)/sum(U)=1.62558234877567`:

| category | windows | full `A` | observable `A` | `sum(U)` | `sum(A)/sum(U)` | unique identities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 8 | 50 | 50 | 35.58510638297872 | 1.4050822122571003 | 4,393 |
| code | 8 | 58 | 58 | 38.015957446808514 | 1.5256751084371063 | 5,076 |
| multilingual | 8 | 60 | 60 | 33.11702127659574 | 1.8117571474461935 | 4,229 |
| rare-route | 8 | 64 | 63 | 36.0 | 1.7777777777777777 | 5,167 |

The eight capture processes were cold, batch one, concurrency one: four fresh
prefill authorities followed by four fixed-64 generation captures. Generation
reports record `8,107,486,292,992` logical source bytes and
`8,116,404,166,656` process-read bytes; the separate fresh-prefill authorities
record `1,971,671,544,832` and `1,973,921,333,248` bytes respectively. Capture
free memory stayed at or above 52%, peak RSS was 618,397,696 bytes, and every
run recorded zero swap growth, zero new throttling, live protected services,
and a release boundary. The manifest itself accepts zero tokens and contains
`performance_claim: null`; these capture totals are evidence acquisition, not
endpoint TPS.

## Decision

Promote this manifest only as the corrected four-category causal authority for
PW-0329 and PW-0332. It supersedes the assumption that later bonus-free
PW-0208 routes or an `A+1` projection can represent the repaired transaction
boundary. Preserve both historical sources under their original names. No
runtime default, achieved TPS, K4 fidelity, cache, or general full-capability
claim follows.
