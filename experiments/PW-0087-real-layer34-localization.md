# PW-0087 — Real layer-34 substage localization

- Status: complete
- Disposition: promoted diagnostic; repair not yet selected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0086 comparison
  `d23e411ab91712636d45553463ef162652403a60d3ee76f9bb835c007dce001f`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0085, PW-0086

## Hypothesis and contract

PW-0086 proves layer 33 is the last bit-exact accumulated state and layer 34
is the first actual divergence, even though the six-value delta does not
formally fail the layer-final gate until layer 36. Extend the generalized
routed-layer selector to layer 34 without adding execution or comparison
authority.

Rust must causally recompute layers 0–33, then capture production layer 34.
PyTorch must load the hash-verified PW-0060 layer-33 final and independently
derive the same 21 substages. Bind source, revision, checkpoint, prompt,
numerical policy, shapes, schedule, commit, and evidence hashes. Preserve all
BF16, final-state, exact-expert, and `5e-7` route-weight gates. Identify the
first actual differing substage before changing arithmetic.

Retain normative Gate 8 at every phase, record batch 1, concurrency 1,
accepted tokens 0, buffer release, allocator relief, and complete wall time,
and preserve stopped evidence. This diagnostic cannot count as TPS or alter
any threshold.

## Result

The independent oracle completed in 19.204 seconds and production Rust in
543.499 seconds. Incoming state, input norm, QKV, query, key, value, sinks,
all centered scores, and all probabilities are bit-exact. The first actual
difference is exactly one of 221,184 attention values: position 24, head 49,
value dimension 2 is `0.09521484375` in PyTorch and `0.095703125` in Rust.

PyTorch's vector-by-25×128 BF16 matrix operation uses the generic GEMM
four-part reduction. The exact pair produces raw F32 `0x3dc37fff` and rounds
to the oracle BF16 `0x3dc3`; Rust's specialized contiguous-dot topology
produces tie `0x3dc38000` and rounds to `0x3dc4`. The PW-0076 fixture does not
discriminate these topologies—both round to its oracle BF16 value—so its
narrower specialized-operation inference is superseded by this pair.

The one value spreads to 44 projection values and three post-attention values.
Router logits are the first formal substage failure; route weights differ by
at most `6.902825546273306e-6`, expert sets/order remain exact, and final state
reproduces PW-0086.

Both captures passed Gate 8. Rust peaked at 719,912,960 bytes RSS and
603,709,696 bytes physical footprint, returned to 153,817,280 bytes, retained
83% free memory, grew no swap, observed no throttling, and kept every
protected service healthy. Evidence hashes:

- Oracle manifest:
  `dcf92f0c37e825766984f524b2338701adf28dd528ffafd374d59e6f20673fc1`
- Rust manifest:
  `23e5f820d3df4eb9982056a913ece1fae12c9c3cc615784d041cd9b3616cb895`
- Comparison:
  `7f455767cbe8b065a87185b6e75a09a51a80cd702ccfb786b3086f5451a5f5de`

## Decision

Promote the localization, not the repair. Freeze the exact layer-34 pair as a
fixture that distinguishes generic four-part GEMM from the specialized dot.
Preserve PW-0076 as a non-discriminating historical fixture, then use the
generic topology for attention value-by-matrix reduction only if both fixtures
and the complete suite pass. Replay layer 34 before another full walk.
