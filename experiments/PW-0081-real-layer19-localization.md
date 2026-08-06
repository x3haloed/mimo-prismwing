# PW-0081 — Real layer-19 substage localization

- Status: complete
- Disposition: promoted diagnostic; repair not yet selected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0080 comparison
  `eb2f578b983a6be8befc29dc2724607d33fa81ec6cc4a77311dda1ad8a7d02c2`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0079, PW-0080

## Hypothesis and contract

PW-0080 proves layer 18 is the last bit-exact accumulated state and layer 19
is both the first actual and formal divergence. Extend the generalized
routed-layer selector to layer 19 without adding execution or comparison
authority.

Rust must causally recompute layers 0–18, then capture production layer 19.
PyTorch must load the hash-verified PW-0060 layer-18 final and independently
derive the same 21 substages. Bind source, revision, checkpoint, prompt,
numerical policy, shapes, schedule, commit, and evidence hashes. Preserve all
BF16, final-state, exact-expert, and `5e-7` route-weight gates. Identify the
first actual differing substage before changing arithmetic.

Retain normative Gate 8 at every phase, record batch 1, concurrency 1,
accepted tokens 0, and wall time, and preserve stopped evidence. This
diagnostic cannot count as TPS or alter any threshold.

## Result

The independent oracle completed in 19.458 seconds and production Rust in
303.484 seconds. Incoming state, input norm, QKV, query, key, value, and sinks
are bit-exact. The first actual difference is exactly one of 25,920 centered
attention scores: position 12, head 25, source token 2 is `-0.30078125` in
PyTorch and `-0.3046875` in Rust.

For that real width-192 pair, PyTorch's BF16 dot is `11.4375`. Forward and the
PW-0070 four-lane topology both produce raw F32 `0x41368000` and round to
`11.375`; the already pinned specialized eight-vector topology produces raw
`0x41368001` and rounds to the PyTorch value. The PW-0070 layer-7 pair does not
discriminate these two topologies at BF16—both round to its oracle value—so the
earlier belief that sink-bearing SWA scores require the four-lane fallback is
superseded by this discriminating pair.

The one score changes two probabilities and propagates through 33 attention
values. Post-attention is the first formal substage failure, with `0.125`
maximum error; route weights differ by at most `0.00008033359680170715`,
expert sets/order remain exact, and final state reproduces PW-0080.

Both captures passed Gate 8. Rust peaked at 709,148,672 bytes RSS and
598,204,672 bytes physical footprint, returned to 138,948,352 bytes, retained
at least 82% free memory, grew no swap, observed no throttling, and kept every
protected service healthy. Evidence hashes:

- Oracle manifest:
  `5bf6ed69aa01293e8020e3d4b2dc3a34dd087672901f59e321258f8ab1c0313b`
- Rust manifest:
  `e436e33ac408c2a816a60e75812ba90c0968d87e6b5b86aa422dc86ed6ad0436`
- Comparison:
  `aac02bb6a371534fc7be72ca6c6e2dda123ecde9d710c65cac7406668009a898`

## Decision

Promote the localization, not the repair. Add the real layer-19 pair as a
deterministic specialized-vector fixture, prove it also preserves every prior
dot fixture and tiny semantic, then use the specialized topology for all BF16
attention score dots. Replay layer 19 before another full walk.
