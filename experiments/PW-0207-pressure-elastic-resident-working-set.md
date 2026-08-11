# PW-0207 — Pressure-elastic resident working set

- Status: proposed
- Disposition: unexecuted
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

Unexecuted. This is the highest-priority direct use of the relaxed memory
target because it attacks the measured dominant embodiment without changing
weights or accepted outputs.
