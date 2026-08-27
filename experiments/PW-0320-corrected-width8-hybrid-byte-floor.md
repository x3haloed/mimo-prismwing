# PW-0320 — Corrected width-eight hybrid-bank byte floor

- Status: complete
- Disposition: rejected
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

## Result

Every tested bank/cache pair fails. With no cache, median optimistic accepted
TPS rises only from `0.6309` at 512 identities to `0.7353` at 2,048. Granting
the 2,048-identity bank a perfect per-window 4 GiB cache raises the median to
only `0.8452`; the range is `0.2312` to `1.0710`, and zero of 32 windows reaches
2 TPS. Every category therefore has zero passing windows.

The strongest configuration still moves 22.691–33.282 GB per width-eight
transaction after the free oracle cache. At the observed `A`, individual
windows require 6.483–30.017 GB/s for 2 TPS. Even replacing every observed
acceptance with the structural maximum `A=8` leaves the best window below
1.23 TPS on the measured transport, before charging any compute or common
weights.

Canonical evidence:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0320/analysis-001/analysis.json`,
SHA-256
`de6424aa68d0c65f8f9206a53f61475286bde501873cd4f6ee06299c9b37d7a9`.
Gate 8 retained 70% free memory, at most 134,266,880-byte peak RSS, zero swap
growth, zero new throttling, and stable protected services through release.

## Decision

Reject an integrated width-eight source/K4 streaming runner on the current M1
storage path. PW-0136's cold acquisition miss was not a scheduling accident:
corrected width-eight expert union is too large even with a much larger K4 bank,
perfect caching, and all non-storage work removed.

Do not construct the 1,024-identity bank for this architecture. Reopening
requires a premise that changes the bound: a materially smaller executable
record, a wider verifier with measured sublinear expert-union growth and much
higher accepted tokens per transaction, or a resident companion embodiment.
Zero tokens were accepted and no throughput-model constant or runtime default
changes.
