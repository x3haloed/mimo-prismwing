# PW-0078 — Real layer-14 substage localization

- Status: complete
- Disposition: promoted diagnostic; repair not yet selected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0077 comparison
  `60b14d05e68f06c0fe4246cb856aec960f6382090f62affb0fdf1bdb7db518be`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0074 through PW-0077

## Hypothesis and contract

PW-0077 proves layer 13 is the last bit-exact accumulated state and layer 14
is both the first actual and formal divergence. Extend the generalized
routed-layer diagnostic selector to layer 14 without adding execution or
comparison authority.

Rust must causally recompute layers 0–13, then capture production layer 14.
PyTorch must load the hash-verified PW-0060 layer-13 final and independently
derive the same 21 attention, routing, selected-expert, scatter, and residual
substages. Bind all source, revision, checkpoint, prompt, numerical-policy,
shape, schedule, commit, and evidence hashes. Preserve every existing BF16,
final-state, exact-expert, and `5e-7` route-weight gate. Identify the first
actual differing substage before changing arithmetic.

Retain normative Gate 8 at every phase, including free-memory, current/peak
RSS, post-release footprint, swap-growth, throttling, buffer-release, allocator,
and protected-service stops. Record batch 1, concurrency 1, accepted tokens 0,
and wall time. This diagnostic cannot alter any acceptance threshold or count
as TPS.

## Result

The independent oracle completed in 15.445 seconds and the production Rust
trace in 230.205 seconds. Incoming state and every attention capture through
the post-attention residual are bit-exact. The first actual difference is
post-attention RMSNorm: 41 of 110,592 `moe_input` BF16 values differ, all in
position 1, with `0.0078125` maximum error and `1.2464162877813474e-4`
relative L2.

The input row is exact, so the discrepancy is reduction order rather than
weights or attention. Rust's prior high-precision square sum yields F32
variance raw `0x41913477`; PyTorch yields adjacent `0x41913476`. Replaying
the pinned `SumKernel.cpp` contiguous-inner cascade—four interleaved vector
rows, 16-vector hierarchical chunks across four levels, then vector-lane
reduction—reproduces PyTorch's variance bit exactly. Forward and simple
four-lane sums do not.

The one-ULP variance difference changes the inverse RMS by two F32 ULPs and
tips 41 weighted outputs across BF16 boundaries. Router logits are the first
formal substage failure, route weights differ by at most
`0.00027570375320434826`, and expert sets/order remain exact. The final state
reproduces PW-0077 exactly.

Both captures passed Gate 8. Rust peaked at 746,684,416 bytes RSS and
657,842,368 bytes physical footprint, ended at 376,264,896 bytes, retained at
least 82% free memory, grew no swap, observed no throttling, and kept every
protected service healthy. Evidence hashes:

- Oracle manifest:
  `53e6b5db1d63128fddc2d3d6a8445424021f89f6f20131c98a42ab857f819e1f`
- Rust manifest:
  `b53fff7912c7b2f6bd8aadf7786c2ff1783b6a91130fad323aa4157b01e4ebda`
- Comparison:
  `8f704093d177b83ed4963f2107ff230d0267c411f70149537f16616cc5083cc7`

## Decision

Promote the localization, not the repair. Build a deterministic real-row
fixture for the pinned PyTorch F32 cascade sum and RMS inverse, then replace
the high-precision RMS variance reduction only after the full suite gates it.
Replay layer 14 before any full walk or downstream routing change.
