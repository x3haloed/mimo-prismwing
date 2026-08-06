# PW-0104 — Real-route Belady cache bound

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: to be recorded by the analysis manifest
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

Unexecuted.

## Decision

Run the deterministic policy replay before implementing any resident-cache or
prefetch service.
