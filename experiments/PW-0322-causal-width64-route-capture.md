# PW-0322 — Causal width-64 corrected-route capture

- Status: complete
- Disposition: rejected
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0321

## Question

Does one real target-generated width-64 proposal, verified in one causal target
transaction, retain a two-TPS hybrid-storage upper bound after replacing
PW-0321's stitched route unions?

## Contract

Use the authenticated ordinary PW-0208 prompt and pinned checkpoint. Generate
63 proposal positions autoregressively with the same local target arithmetic
from one anchor, then verify all 64 positions in one target call with complete
per-layer route traces. Retain the prefill-produced anchor plus only one
transaction-authorized observable token so execution stops after this
transaction. This target-generated proposal measures a favorable
route/acceptance ceiling; it is not a deployable draft.

Require verifier width 64, one transaction, 48 layer traces, 47 routed layers,
64 positions, eight unique experts and normalized weights per position, exact
cache rollback, complete source-byte accounting, and Gate 8 through release.
Analyze the real union with PW-0320's 2,048-identity bank, perfect free 4 GiB
cache, record sizes, and measured cold bandwidth. Continue only if actual
verifier-authorized `A` and union imply at least 2 optimistic accepted TPS.

No K4 construction, streaming runtime, endpoint TPS, hosted parity, modality
claim, or proposer promotion is authorized by this capture.

## Preserved failed attempts

Two initial attempts stopped during prefill because PW-0323's superseded Rust
Gate 8 rule treated healthy supervised `nxnode` PID replacement as service
loss. A third process completed prefill but executed zero transactions because
the capture requested one output token and the prefill path had already
produced that anchor. Its report SHA-256 is
`d9d4f5c6ae5ce229e5cc4c3c322274fe7d4d9d9c66bc75d2c19a375439f8b0d6`;
it records `transactions=0` and zero proposal/verification wall and is rejected.
The corrected runner requests exactly two observable tokens, forcing one loop
entry while retaining only one transaction-authorized token.

That corrected process completed all 63 target-generated proposal steps, then
failed closed at the first q64 layer-0 QKV because the accelerated FP8/BF16
linear interface admits at most eight rows. Its progress artifact is preserved
as `ordinary-q64-report-002.progress.jsonl`; no completed report exists. The
capture-only repair chunks q64 QKV and BF16 linears into ordered width-eight
calls and charges repeated logical bytes. Its timing is therefore diagnostic
and cannot be used as a q64 performance claim. Routed-layer route authority
remains one causal 64-position transaction.

The next process again completed proposal generation and entered q64
verification, then failed closed at routed expert 18 because that expert had
more than the runtime's 32-placement panel capacity. Its progress artifact is
preserved as `ordinary-q64-report-003.progress.jsonl`. The capture-only repair
partitions each expert's ordered placements into complete panels of at most 32,
reuses the same expert identity and exact weights, and scatters every panel into
the same 64-row layer output. No placement may be truncated or reordered.

## Result

Clean implementation commit `be33e94b1dd586fd57243de222ce65b88792444c`
completes exactly one causal q64 transaction. It records 64 proposal and 64
posterior positions, 48 layer traces, and 64 complete eight-expert route rows
for each of 47 routed layers. The target authorizes only three tokens: one
retained proposal row plus correction. Proposal wall is 1,230.694 seconds and
chunked verification wall is 124.087 seconds; both are diagnostic-only.

The real union contains 4,482 layer/expert identities. PW-0319's fixed
2,048-identity bank covers 1,353 and leaves 3,129 source identities. A perfect
free 4 GiB cache still leaves 91,589,858,640 bytes. At structural `A=64`, the
storage-only ceiling is 2.425 TPS; at actual `A=3`, it is only
`0.113673556` TPS and needs 61.060 GB/s to reach two TPS.

Raw report:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0322/ordinary-q64-report-004.json`,
SHA-256
`ef893b83105009576771b4dcbd98b4f82320b838e58920f5c32011e9a52acb60`.
Canonical analysis:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0322/analysis-001/analysis.json`,
SHA-256
`8c824040776c5ca2b9d9f0854d9066c00f7b5495296c86e113729e6e07a6b98d`.
Capture Gate 8 retains at least 69% free memory, at most 494,551,040-byte peak
RSS, zero swap growth/new throttling, and healthy named services despite
recorded `nxnode` replacement. Analysis Gate 8 also passes.

## Decision

Reject target-generated q64 as the acceptance mechanism for the current M1
hybrid-storage architecture. PW-0321's stitched acceptance sums cannot stand in
for a single causal transaction: an early mismatch truncates the accepted path.
Do not construct the K4 bank or build the q64 streaming runtime. Reopening wide
speculation requires an independently qualified proposer with dramatically
higher single-transaction conditional agreement, not a wider block alone.
Zero endpoint tokens were accepted for performance and no runtime default or
throughput constant changes.
