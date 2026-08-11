# PW-0206 — Corrected-QKV proposal authority regeneration

- Status: running
- Disposition: conditional; native-MTP and DFlash branches reopened, width-eight authority pending
- Date: 2026-08-10
- Execution mode: target-faithful audit plus separately named modified controls
- Related records: PW-0102, PW-0103, PW-0150, PW-0186, PW-0187, PW-0203, PW-0205

## Hypothesis and mechanism

PW-0205 proves that earlier whole-model paths and their Python oracle assigned
the checkpoint's four tensor-parallel `[Q,K,V]` shards to the wrong semantic
rows. Because attention outputs feed later routers and logits, proposal tokens,
posterior tokens, route unions, `A`, and `U` derived through that mapping may
not transfer to the corrected runtime.

The hypothesis is deliberately diagnostic: regenerating the smallest decisive
traces will either preserve the old speculative conclusions or identify which
ones must be rerun. No speedup is claimed.

## Contract

Use the pinned checkpoint, tokenizer, prompts, seeds, and greedy verification.
Deinterleave global and sliding-window QKV exactly as the pinned SGLang loader
does. Keep source-faithful and SGLang-directed block-scaled arithmetic in
separate artifacts. Validate deinterleaving with a deterministic row-identity
fixture before a model walk.

Regenerate, in order:

1. the native-MTP first proposal and target posterior used by PW-0103;
2. the corrected DFlash-mask control used by PW-0150;
3. the Jacobi `q=8` convergence trace underlying PW-0186/PW-0187; and
4. only if materially changed, the smallest route-union slice needed to
   recompute `A/U` and the physical acquisition bound.

## Cheap falsifier and gates

Kill further regeneration if token IDs, accepted prefix, every per-layer route
set, and `A/U` reproduce the old trace exactly. Promote only the statement that
the prior conclusion survived corrected semantics.

If any of those authorities changes, mark the affected old conclusion
non-portable and regenerate its cheapest downstream bound. Do not rerun a full
endpoint until that bound can alter a branch decision. Record old/new token
IDs, `A`, `U`, routes, logical and physical bytes, cold/warm state, and safety.

## Partial result — corrected prefix, decode, native MTP, and DFlash

Clean commit `23d37093c613c979f567755bc3a1e33d09b0eb69` regenerates the
27-token Hello prefix with deinterleaved QKV. Its manifest hashes to
`0002c617c5459d7531de99e779ecad7335afc1e6f86cbbb6071afa23da107807`.
The greedy token changes from old-layout token 264 to token 9707 (`Hello`),
and all 1,269 routed positions across all 47 MoE layers change expert sets.
The corrected layer-47 capture hashes to
`e485df1c61820505c431b390825849ae05af0b190a568dae023ec7a215644fbe`.

The subsequent two-token source-faithful retained-cache decode hashes to
`f405225ea063bf3bfaf38a450fe752dc32c5afe54f69f5803c3ae61308caab2d`
and produces `[9707,0]`, exactly the frozen hosted `Hello!` fixture. Complete
wall is 1,343.751 seconds; peak RSS is 4,446,715,904 bytes, minimum free memory
is 69%, and swap growth and new throttled pages are zero.

Commit `134020acbecb4882912b96d6397b1d0b173a07c2` then reruns native MTP
layer zero against the corrected layer-47 states and shifted target pair
`9707 -> 0`. The result reverses PW-0103: MTP proposes token 0 at rank one,
with logit 25.5. The manifest and full-logit capture hash to
`07233ee71f194c887d96aac2cb341239df1728a7e05fc36f692a1188c65b3379`
and `72adac56eb786f95a84bd18f655bdfe5a65202798cc96644aa9b52af6466c3dd`.
It completes in 6.198 seconds, peaks at 3,867,754,496-byte RSS, and passes the
same zero-swap, zero-throttling safety gate.

Commit `b749a745f20882dd97326f3486ebcf86ead4028b` then reruns the
authenticated exported-mask DFlash control. Its manifest hashes to
`e5084e606349fb9fe0b01f8e5505f43fa58969cae5398330b343f40dab7228c9`
and proposes `[9707,0,0,0,0,0,0,0]`. The first speculative suffix token is
therefore exactly the corrected target's second token, reversing PW-0150's
first-suffix mismatch. Complete draft wall is 77.118 seconds, peak RSS is
3,356,966,912 bytes, and the run records no swap growth or throttling. This is
still draft-side diagnostic evidence; it records `accepted_tokens=0` and no
performance claim until source-target verification measures the complete
width-eight posterior and route union.

These are causal prerequisites, not accepted TPS. They reopen PW-0208 and the
DFlash branch, and make PW-0103/PW-0150's rejections explicitly non-portable to
corrected QKV semantics. The corrected width-eight target/Jacobi and `A/U`
authorities remain pending, so PW-0206 is not complete and PW-0203's economics
are not yet carried forward.

## Decision

Continue. The cheap falsifier decisively rejects trace identity: token IDs and
every route set changed. Promote the corrected prefix/decode authorities and
reopen native MTP and DFlash for exact target verification, while completing
the remaining `A/U` regeneration before endpoint performance work. No measured
throughput constant changes at this partial checkpoint.
