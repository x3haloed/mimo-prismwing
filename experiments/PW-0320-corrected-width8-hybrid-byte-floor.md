# PW-0320 — Corrected width-eight hybrid-bank byte floor

- Status: planned
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0319

## Question

Can any PW-0319 bank frontier support two accepted tokens/s for the corrected
width-eight verifier windows on the measured M1 internal-SSD transport, even
under an optimistic perfect 4 GiB expert cache and with compute omitted?

## Authorities and mechanism

Use exactly PW-0208's 32 primary corrected windows, their observed accepted
token count `A`, and PW-0319's deterministic selection order at budgets 512,
1,024, and 2,048. Bind executable record sizes to PW-0318's authenticated
schema-2 bundle: 12,654,604 bytes per K4 identity and 25,171,968 bytes per
source-FP8 identity. Bind storage bandwidth to PW-0136's selected two-worker
cold median: 201,719,808 bytes in 58.125 ms, or 3,470,448,309.677 bytes/s.

For each window and routed layer, count every distinct selected expert across
all eight verifier positions once. Charge K4 or source bytes according to the
fixed bank. Evaluate no cache, 2 GiB, and 4 GiB. The cache is an intentionally
optimistic per-window oracle that removes the largest required records up to
capacity; it has no fill cost, perfect foresight, and no competing common
weights. Omit all GPU, attention, router, proposer, synchronization, and common
weight time. Thus the resulting TPS is an upper bound, not performance.

## Gates

- Authenticate all upstream files and reproduce PW-0319's corrected route hash.
- Preserve 32 windows, 47 routed layers, eight positions, unique expert union,
  and each frozen `A` in `[1,8]`.
- Report unique K4/source identities, bytes, bytes per accepted token, implied
  storage wall, optimistic accepted TPS, and bandwidth required for 2 TPS for
  every window and budget/cache combination.
- Reject a budget/cache pair if any category has zero windows whose optimistic
  TPS reaches 2, or if fewer than half of all windows reach 2 TPS.
- The entire width-eight hybrid branch is storage-falsified on current M1
  transport if even budget 2,048 plus perfect 4 GiB cache fails that gate.
- A pass authorizes a cold integrated streaming runner; it does not authorize a
  K4 construction tranche or endpoint claim.

## Claims excluded

No new weights are qualified, no endpoint runs, no accepted tokens are emitted,
and no storage-only bound is reported as measured TPS. Wider speculation,
faster purchased storage, modalities, Prismwing-2, and Prismwing 50 remain
outside this record.
