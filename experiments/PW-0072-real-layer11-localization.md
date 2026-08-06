# PW-0072 — Real layer-11 substage localization

- Status: complete
- Disposition: promoted diagnostic; repair not yet selected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0071 comparison
  `744fad6a7ba4b9ea883c5f53eda2f4fafa67569e82718a65ec9cbdaac526a9c4`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0065, PW-0069 through PW-0071

## Hypothesis and contract

PW-0071 proves layer 10 is the last bit-exact accumulated state and layer 11 is
the first actual divergence. Extend the already-regressed routed-layer
diagnostic to target layer 11 without duplicating execution or comparison
authority.

Rust must causally recompute layers 0–10, then capture production layer 11.
PyTorch must load the hash-verified PW-0060 layer-10 final and independently
derive layer-11 attention, routes, selected experts, expert tensors, scatter,
and residual. Capture the same 21 substages and bind the same source, revision,
checkpoint, prompt, numerical policy, shapes, schedules, and hashes as PW-0069.

Keep all BF16, final-state, exact-expert, and `5e-7` route-weight gates. Name
the first differing substage before changing arithmetic. Retain every normative
Gate 8 phase-level RSS/free-memory/swap/throttling/release/protected-service
stop. Generated tensors stay external; accepted tokens are zero and no
throughput or hosted threshold can change.

## Result

Oracle run 001 failed closed before a manifest because the earlier diagnostic
assumed every routed target used the SWA QKV layout. Layer 11 is a global
attention layer: 4 KV heads, 13,568 QKV rows, no sink, 24,192 causal scores,
and RoPE theta 10,000,000. The corrected generalized authority derives those
properties from the frozen hybrid pattern. Oracle run 002 completed in 15.132
seconds with the correct global topology.

The production trace completed in 180.959 seconds. Incoming state, input norm,
QKV, query, key, and value are bit-exact. The first actual arithmetic difference
is one of 24,192 centered scores: position 22, head 3, source token 16 is
`-1.421875` in PyTorch and `-1.4296875` in Rust. It changes one probability and
eight attention-output values, then propagates to five of 110,592 final BF16
values, exactly reproducing PW-0071's layer-11 result. Expert sets/order remain
exact and route-weight error is `2.8038110733152877e-7`, inside the `5e-7` gate.

Pinned PyTorch source explains why PW-0070's four-lane sum is insufficient for
this case. The specialized reduced-precision GEMV dot uses eight four-element
F32 vector accumulators over 32-element blocks, a pairwise accumulator tree,
and an ARM horizontal reduction. Replaying that topology gives the PyTorch
centered score exactly; the four-lane sum gives the Rust value. Gate that
specialized reduction on the real pair before changing production arithmetic.

The Rust trace peaked at 752,926,720 bytes RSS and 662,659,264 bytes physical
footprint, returned to 130,096,832 bytes after captures, retained at least 81%
system-free memory, grew no swap, observed no throttling, and retained every
protected service. Evidence hashes:

- Oracle run 002 manifest:
  `639730fb729855f94eecb5716abdbf68d6d98849c0cfbfbf1d87d86dc9d462dd`
- Rust manifest:
  `ec530539a5db6af1633ec0215aa096a46f4bc33ef85be87a707eabd185746d48`
- Comparison:
  `eb764c15082cbe78c61ab59d80af6c4607ce502702e8bc99e1b427c79c52bc9d`

## Decision

Promote the localization and global-attention diagnostic support, not an
arithmetic repair. Build a deterministic fixture for the specialized PyTorch
BF16 vector dot and gate it before replacing the narrower four-lane helper.
No full walk, throughput claim, or threshold change is justified yet.
