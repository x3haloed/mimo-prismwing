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
per-layer route traces. Emit only the first observable token so execution stops
after this transaction. This target-generated proposal measures a favorable
route/acceptance ceiling; it is not a deployable draft.

Require verifier width 64, one transaction, 48 layer traces, 47 routed layers,
64 positions, eight unique experts and normalized weights per position, exact
cache rollback, complete source-byte accounting, and Gate 8 through release.
Analyze the real union with PW-0320's 2,048-identity bank, perfect free 4 GiB
cache, record sizes, and measured cold bandwidth. Continue only if actual
verifier-authorized `A` and union imply at least 2 optimistic accepted TPS.

No K4 construction, streaming runtime, endpoint TPS, hosted parity, modality
claim, or proposer promotion is authorized by this capture.
