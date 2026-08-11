# PW-0206 — Corrected-QKV proposal authority regeneration

- Status: complete
- Disposition: corrected Jacobi leverage promoted; native-MTP open; endpoint promotion withheld
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

## Partial result — corrected DFlash target verification

Clean commit `ab24f8070d207cd7b0d48d9d2d4c11ce4ac26deb` executes the
source-faithful width-eight target walk. Its manifest hashes to
`edf677be8406bd663e0d99b67c8cfb12fdad3914a100dbfd31f9d92b4787693e`.
The exact posterior is `[0,2585,2585,2585,2585,2585,2585,2585]`. One draft
suffix token matches, so formal accepted length improves from the stale
PW-0102 trace's `A=1` to `A=2`; the committed block is `[9707,0]` and the
correction token is 2585.

Across the 47 routed layers, the eight-position pass selects 1,010 unique
layer-local experts, 14--31 per layer. Mean normalized union is
`U=2.6861702127659575`, hence `A/U=0.7445544554455445`. This retains a real
acceptance improvement, but fails the minimum expert-byte leverage gate
`A/U>1` and reaches only 9.863% of PW-0011's otherwise-free INT4 requirement
`7.548793`. Verification-source expert traffic is 25,423,687,680 bytes and
complete verification-source traffic is 33,166,990,208 bytes.

Complete process wall is 1,457.385 seconds; post-prefill wall is 306.953
seconds and its single-trace diagnostic is 0.006516 accepted token/s, not
endpoint TPS. Physical reads are 133,008,273,408 bytes. The run retains at
least 60% free memory, peaks at 4,150,575,104-byte RSS and 212,273,984-byte
physical footprint, and records zero swap growth and throttling.

Reject this raw corrected DFlash block as routed-expert leverage, while
preserving its doubled acceptance and using its exact posterior as the
authorized corrected Jacobi successor falsifier. PW-0206 remains open until
that convergence/`A/U` authority is regenerated.

Corrected Jacobi iteration two runs cleanly at commit
`a54c46b6e04937507539a167ae17f4f42f44e60a`. Its manifest hashes to
`dd53f80c02418d4d0321b400a47c1a88bcc70cf72626570fb5302266e6cf39cf`.
The successor block `[9707,0,2585,2585,2585,2585,2585,2585]` produces
posterior `[0,2585,646,646,2585,2585,2585,2585]`, increasing formal accepted
length to `A=3`. Route union also rises to `U=3.1675531914893615`, leaving
`A/U=0.947103274559194`: 27.204% better than the raw corrected block, but still
below the leverage gate of one.

Post-prefill wall is 357.799 seconds, verification-source traffic is
37,723,116,416 bytes, and physical reads are 137,583,656,960 bytes. Minimum
free memory is 62%, peak RSS is 3,947,216,896 bytes, and swap growth and new
throttled pages remain zero. Because `A/U` materially improves, authorize the
third posterior-fed iteration under the predeclared convergence rule; do not
promote iteration two as a runtime path.

Corrected Jacobi iteration three completes cleanly at commit
`ea5dc4e051fceb00f2a53f12a8517c56f122d9af`. Its manifest hashes to
`cf9403b441b9453557d9c6fb2481d0dd361e319efbd6dad2c4e21d5c424ed3d1`.
The successor `[9707,0,2585,646,646,2585,2585,2585]` produces posterior
`[0,2585,646,358,358,646,2585,2585]` and commits four tokens. Measured
`U=3.702127659574468` yields `A/U=1.0804597701149425`, crossing the minimum
expert-byte leverage gate by 8.046%. This is 14.080% better than iteration two
and 45.115% better than the raw corrected DFlash block.

The gain is real but small: it reaches only 14.313% of PW-0011's otherwise-free
INT4 requirement `7.548793`. Post-prefill wall is 410.954 seconds, target-source
traffic is 42,782,681,984 bytes, physical reads are 142,661,935,104 bytes, and
the single-trace diagnostic is 0.009733 accepted token/s. Minimum free memory
is 61%, peak RSS is 3,904,618,496 bytes, and swap growth and new throttled pages
are zero.

PW-0206's contracted corrected authorities are now complete. Promote the
existence of modest corrected Jacobi expert-byte leverage and carry it forward
as a lower-milestone mechanism, but do not promote an endpoint or infer that it
can reach 50 TPS. Native MTP remains open for PW-0208's cost-aware chaining.

## Decision

Continue. The cheap falsifier decisively rejects trace identity: token IDs and
every route set changed. Promote the corrected prefix/decode authorities and
retain native MTP for PW-0208, reject the raw DFlash block's expert-byte
economics, and promote only corrected Jacobi iteration three's measured
`A/U=1.080460` as a lower-milestone input. The regenerated authority closes
PW-0206; no endpoint throughput constant or runtime default changes.
