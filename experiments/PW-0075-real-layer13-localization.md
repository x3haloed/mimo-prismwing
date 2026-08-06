# PW-0075 — Real layer-13 substage localization

- Status: complete
- Disposition: promoted diagnostic; repair not yet selected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0074 comparison
  `0dd64f521715c86fea52557168a5101cdaef76421269b6ed6c1b46b964c9ced6`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust trace
- Related records: PW-0060, PW-0071 through PW-0074

## Hypothesis and contract

PW-0074 proves layer 12 is the last bit-exact accumulated state and layer 13
is the first actual divergence. Extend the generalized routed-layer diagnostic
selector to layer 13 without introducing a second execution or comparison
authority.

Rust must causally recompute layers 0–12, then capture production layer 13.
PyTorch must load the hash-verified PW-0060 layer-12 final and independently
derive the same 21 attention, routing, selected-expert, scatter, and residual
substages. Bind source input, revision, checkpoint, prompt, numerical policy,
shapes, schedules, commit, and hashes. Preserve the BF16, final-state,
exact-expert, and `5e-7` route-weight gates. Name the first actual differing
substage before changing arithmetic.

Retain normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after release, above 512 MiB swap
growth, on new throttled pages, or on protected-service loss. Record buffer
release, allocator relief, hardware, batch 1, concurrency 1, accepted tokens
0, and wall time. Generated tensors stay external. This diagnostic cannot
alter hosted, fidelity, capability, cost, power, safety, or TPS thresholds.

## Result

The independent oracle completed in 17.991 seconds and the production Rust
trace in 213.801 seconds. Incoming layer-12 state, input norm, QKV, query, key,
value, sinks, all 25,920 centered attention scores, and all 25,920 BF16
probabilities are bit-exact. The first actual difference is exactly one of
221,184 attention-output values: position 24, head 4, value dimension 52 is
`-0.0208740234375` in PyTorch and `-0.020751953125` in Rust.

The current forward F32 reduction lands exactly on raw `0xbcaa8000`, a BF16
tie that rounds to the Rust value. Replaying the pinned PyTorch specialized
BF16 vector-tail topology lands two F32 ULPs lower at `0xbcaa8002` and rounds
to the oracle value. The already implemented specialized helper therefore
explains this boundary, but it is not promoted for attention-value reduction
until a hash-bound real fixture and the full correctness suite gate that use.

That single BF16 quantum changes nine attention-projection values and two
post-attention/normalization values. Router logits are then the first formal
substage failure (`9.250640869140625e-5` maximum error), route weights differ
by at most `4.8274204254017405e-6`, and expert sets/order remain exact. The
final state exactly reproduces PW-0074: 21 BF16 differences, `0.015625`
maximum error, and `1.6284499569784697e-6` relative L2.

Both captures passed Gate 8. The Rust path peaked at 749,797,376 bytes RSS and
660,988,160 bytes physical footprint, returned to 131,729,280 bytes after
captures, retained at least 79% free memory, grew no swap, observed no
throttling, and retained every protected service. Evidence hashes:

- Oracle manifest:
  `294e25355d4cb6ca3dcdcb060e131e7599b6603987eaaf0664a39f95ff0ddf74`
- Rust manifest:
  `496c045e3bf8689a7db899636739fa0eca0363e09ec4100a5f0bc8c1f901d8f2`
- Comparison:
  `3d4b1b9deea655713e3438ece5f41a574bdeb27cf6d890c3a4e0716d80bd8b4f`

## Decision

Promote the localization, not the repair. Gate the real 25-element
probability/value pair and its PyTorch specialized vector-tail result, then
use that topology for BF16 attention-value reduction and replay layer 13. Do
not change routing, experts, thresholds, or run another full walk first.
