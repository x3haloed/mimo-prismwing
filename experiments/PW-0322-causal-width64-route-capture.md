# PW-0322 — Causal width-64 corrected-route capture

- Status: planned
- Disposition: pending
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
