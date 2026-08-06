# PW-0104 — Real-route Belady cache bound

- Status: complete
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: `01cc65e1281bb93cf0292e9bb66f0798d2b86dbe`,
  clean executable
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0091 manifest
  `87466b59480a5a5b4256c490f1dfe670fe09f28d21d169085ab13bb1b4b7ab59`
- Hardware/runtime: policy replay on the Apple M1; no model tensors loaded
- Related records: PW-0002, PW-0017, PW-0091, PW-0092, PW-0096, PW-0102,
  PW-0103; prospective E2

## Hypothesis and causal mechanism

An exact resident cache helps only when the same layer-local experts recur
before capacity pressure evicts them. Replay PW-0091's 27 causally valid token
positions in decode order: token major, then routed layers 1--47, then each
position's eight selected experts. Because all source experts have equal
25,171,968-byte executable payloads, exact byte capacity maps to an integer
item capacity without an approximation.

Offline Belady knows the entire future and is an upper bound on every causal
LRU, LFU, TinyLFU, or prefetch policy using the same bytes. If Belady is far
below E2's measured-bandwidth requirement, policy engineering cannot make this
cache footprint the primary Prismwing-50 mechanism.

## Contract and gates

Authenticate the complete PW-0091 manifest and validate every routed row:
27 positions, 47 layers, eight unique in-range experts per layer-position.
Build exactly 10,152 accesses keyed by `(layer, expert)`; experts with the same
ID in different layers are distinct payloads. Report 1--10 GiB capacities,
integer expert slots, compulsory misses, LRU/LFU/Belady hits and misses, hit
ratios, miss bytes, and miss bytes per token. Preserve the complete access-list
hash and per-policy deterministic replay checks.

At 6 and 8 GiB, Belady must approach the predeclared 93--98% required hit range
to keep exact caching as a primary M1 track. Far lower oracle performance kills
cache replacement and prefetch as throughput mechanisms for this trace;
prefetch may still hide latency, and a larger/hardware-resident cache remains a
separate embodiment. This 27-position text trace does not substitute for E2's
eventual multimodal million-position corpus, but an oracle failure is already
decisive for this trace and cannot be repaired by a causal policy.

No endpoint TPS, cache-warm timing, or avoided physical bytes are claimed. The
analysis reports logical source miss bytes only.

## Result

The authenticated PW-0091 trace contains exactly 10,152 accesses to 2,353
distinct layer-local experts. The canonical access-list SHA-256 is
`1127968ca78291ed36530f0e3921c8b3e6751ef9731e11421fb89c7952a3b56f`.
At 6 GiB (255 experts), offline Belady reaches 5,130 hits out of 10,152,
50.531915%. At 8 GiB (341 experts), it reaches 6,095 hits, 60.037431%.
The latter leaves 4,057 misses, 102,122,674,176 logical source bytes, or
3,782,321,265.78 bytes per causal token. It misses the predeclared 93% floor
by 32.962569 percentage points.

The causal controls are strictly worse. At 8 GiB lifetime LFU reaches
36.830181%; LRU reaches zero because the 341-slot cache cannot span the 376
layer-expert accesses between adjacent token boundaries. The analysis checks
that Belady upper-bounds both controls at every integer-GiB capacity from one
through ten, that hits are monotonic with capacity, and that hit/miss and byte
ledgers close exactly. The immutable evidence manifest is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0104/cache-001/manifest.json`,
SHA-256
`7e88f6613f5a3f84970763f90ce357cbdff77e499f2f3673c4482829b918ab17`.

This is deterministic policy replay, not a model process or performance run;
Gate 8 is inapplicable because no model tensors are loaded. It reports no
endpoint TPS or avoided physical I/O.

## Decision

Reject a 6--8 GiB exact resident cache, cache-replacement engineering, or
prefetch as the primary Prismwing-50 throughput mechanism for this trace. Even
future knowledge cannot supply the required reuse. Prefetch can still hide
some latency, and larger companion hardware could change the capacity premise;
both are separate mechanisms. Preserve the single short text trace limitation:
this result does not replace E2's multimodal million-position corpus and does
not assert a universal workload hit rate.
