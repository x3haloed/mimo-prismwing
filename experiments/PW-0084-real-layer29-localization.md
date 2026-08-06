# PW-0084 — Real layer-29 substage localization

- Status: complete
- Disposition: promoted diagnostic; repair not yet selected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0083 comparison
  `c8c6b94313aa780fe1fb1d728529d8fa903e06c4182404c2e096247b2a40c75f`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0082, PW-0083

## Hypothesis and contract

PW-0083 proves layer 28 is the last bit-exact accumulated state and layer 29
is both the first actual and formal divergence. Extend the generalized
routed-layer selector to layer 29 without adding execution or comparison
authority.

Rust must causally recompute layers 0–28, then capture production layer 29.
PyTorch must load the hash-verified PW-0060 layer-28 final and independently
derive the same 21 substages. Bind source, revision, checkpoint, prompt,
numerical policy, shapes, schedule, commit, and evidence hashes. Preserve all
BF16, final-state, exact-expert, and `5e-7` route-weight gates. Identify the
first actual differing substage before changing arithmetic.

Retain normative Gate 8 at every phase, record batch 1, concurrency 1,
accepted tokens 0, buffer release, allocator relief, and complete wall time,
and preserve stopped evidence. This diagnostic cannot count as TPS or alter
any threshold.

## Result

The independent oracle completed in 20.368 seconds and production Rust in
466.678 seconds. Incoming state, input norm, QKV, query, key, value, sinks,
and all 24,192 centered attention scores are bit-exact. The first actual
difference is exactly one attention probability: position 22, head 15, source
token 20 is `0.0245361328125` in PyTorch and `0.0244140625` in Rust.

That one probability propagates through nine of 221,184 attention values and
six post-attention residual values. Router logits are the first formal
substage failure, with `0.00006103515625` maximum error. Route weights differ
by at most `6.262503280618503e-6`, expert sets/order remain exact, and the
20-value final-state difference exactly reproduces PW-0083.

Both captures passed Gate 8. Rust peaked at 750,059,520 bytes RSS and
659,628,352 bytes physical footprint, returned to 148,944,128 bytes, retained
82% free memory, grew no swap, observed no throttling, and kept every
protected service healthy. Evidence hashes:

- Oracle manifest:
  `94c7411a5879f4ade7a700a4309d3a2b48354cc67409701e39003391cadde736`
- Rust manifest:
  `7b00b37526c872b90e25b8c4e151051b0ad436de116b77c3f94718ec0973cadd`
- Comparison:
  `e1309ffe9bec70866181ebd6333212d9964ef82aedbd6fe604473019ec3e1a8f`

## Decision

Promote the localization, not a repair. Freeze the exact 23-value centered
score row and PyTorch F32/BF16 probability payloads, then distinguish
exponential evaluation from denominator reduction and normalization order.
Do not change downstream attention, routing, experts, or thresholds before
that focused fixture proves the causal arithmetic boundary.
