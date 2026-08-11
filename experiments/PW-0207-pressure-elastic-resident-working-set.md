# PW-0207 — Pressure-elastic resident working set

- Status: running
- Disposition: offline wall gate passed; pressure-safe implementation authorized
- Date: 2026-08-10
- Execution mode: L1 target-faithful representation/lifetime change; modified arithmetic controls separate
- Related records: PW-0105 through PW-0111, PW-0181, PW-0205, PW-0206

## Hypothesis and mechanism

PW-0205 emits 47 coherent tokens after about 1.636 TB of process reads while
peaking below 4 GiB. The old 8 GiB ceiling left much of the 16 GiB host
deliberately unused. Gate 8 now permits a declared 12 GiB persistent working
set under a 13 GiB process ceiling.

Hypothesis: choosing resident objects by *measured marginal critical-path stall
avoided per byte*—not tensor frequency—can cut physical reads at least 4x and
complete-path wall time at least 2x on the PW-0205 prompt. Candidate objects
include shared projections/norms, LM-head partitions, and recurrent expert
tiles. The source checkpoint remains authoritative backing storage.

PW-0104 already rejected a 6–8 GiB equal-size expert-only cache on the old
trace: even offline Belady reached only about 60% hit rate at 8 GiB. This record
does not erase that result. Its changed premises are a corrected-QKV route
trace, up to 12 GiB total declared residency, a heterogeneous shared-plus-
expert object set, and selection by attributed stall avoided rather than hits.
If those premises do not change the offline bound, the old rejection stands.

## Contract

Before allocation, emit a residency manifest with object identities, hashes,
exact bytes, reuse distance, expected avoided reads, lifetime, and eviction
order. Total declared persistent residency is at most 12 GiB; process peak is
at most 13 GiB; at least 3 GiB remains outside the process; swap growth and new
throttled pages are zero. Above 8 GiB, sample Darwin pressure state before each
growth step and checkpoint. Warning pressure evicts in declared order (a
runtime without an eviction callback stops); critical pressure stops.

Cold, warm, and pressure-eviction runs are separate. Cache hits count as bytes
avoided only when the control would perform an attributed physical read.

## Cheap falsifier and gates

First replay the corrected PW-0206/PW-0205 route trace through a byte-accurate
offline cache optimizer with measured acquisition latency. Kill implementation
if no legal 12 GiB set predicts either 4x fewer physical bytes or 2x less
attributed acquisition wall.

Then run one real repeated decoder transaction, interleaving control and
candidate. Require exact token/route parity, at least 2x complete-transaction
speedup, and pressure-safe release. Only then run the 32–64-token endpoint.
Endpoint promotion requires at least 2x complete-path TPS, not a cache-hit or
kernel-only number.

## Decision

The first exact route authority is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0207/route-001/report.json`,
SHA-256
`e5c0b93d039ec8d8c6b1f7a0087ec3991ba55df2a1cee7d388f08d6e668d830b`.
Clean commit `871db1ff3c3f654d9184de22c958c877e1402006` reproduces PW-0205
run 009 transaction zero exactly: proposal, posterior, seven committed tokens,
convergence, and `U=4.582446808510638` all match. It captures all 7×48
one-token proposal traces and the 48-layer width-eight verifier trace. Complete
wall is 383.781 seconds, peak RSS is 3,951,427,584 bytes, and no high-residency
allocation occurs.

The clean offline falsifier at commit
`2498dc3b7cdcb1cc7d498ff75428cf4b0cdaf0e0` hashes to
`ee9f71b83ca427bd1a98d166ada77c778c53d81157dfa6ccb071afada54e73eb`.
Its 7,418 object accesses and 171,569,939,456 logical bytes close exactly to
the endpoint ledger. Positive proposal/verification equations attribute
2.180946 ns/byte to the shared spine and 0.450719 ns/byte to routed experts.

The ratio-ranked static 12 GiB set contains 388 shared-spine objects
(7,743,294,336 bytes) and 204 expert bundles (5,135,081,472 bytes), totaling
12,878,375,808 bytes with 6,526,080 bytes unallocated. It predicts
95,872,213,233 physical bytes, only a 1.791485× reduction, so the 4× byte gate
fails. It predicts 56,562.852 ms of attributed acquisition wall versus the
184,510.448 ms control, a 3.262043× speedup, so the alternative 2× wall gate
passes.

Authorize the Darwin pressure observer, declared residency authority, warning
eviction callback, and one repeated-transaction implementation. This is an
offline static prediction on one corrected transaction, not endpoint TPS or a
13 GiB default. The experiment remains running until the real interleaved
transaction and pressure-eviction gates pass or reject it.
