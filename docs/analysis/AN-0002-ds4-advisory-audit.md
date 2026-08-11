# AN-0002 — DS4 advisory audit and research routing

- Kind: analysis record; no Prismwing model run or performance claim
- Date: 2026-08-11
- Owner-supplied input SHA-256:
  `ca34e22f2171b83c73d214d23a7d88f5a38a59273746c3871aee18677d1333c8`
- Primary source: DS4 commit
  `84cc882352757baf628a1776badf7cc54d584e28`, pinned by
  `spec/ds4.lock.json`
- Related records: AN-0001, PW-0207, PW-0208, PW-0210, PW-0211, PW-0212
- Accepted tokens: zero

## Verified transfers

DS4 independently implements the topology PW-0210 intends to test. Its routed
Metal path pairs gate and up reductions, applies SwiGLU and route weight in the
same kernel, and avoids a separate activation dispatch on the normal release
path. Its grouped path also stores the routed intermediate as F16 when the
down-projection consumer loads half for SIMD-group MMA; quality mode retains
F32. This is strong design-prior evidence for producer-consumer fusion and
consumer-matched early narrowing. It is not a Prismwing speedup or correctness
result, and it does not change PW-0210's current `1.026x` impossible-best cold
layer bound.

DS4 also fuses horizontal co-consumers of the same normalized FFN input: router
projection and shared-expert gate/up. MiMo-V2.5 has no analogous shared expert,
so the specific kernel does not transfer. The general search rule does: audit
large freshly produced inputs for independent immediate consumers before
accepting component boundaries as materialization boundaries.

DS4's current mainline issues bounded `F_RDADVISE` requests for streaming
expert ranges. The stronger `F_NOCACHE` plus disabled `F_RDAHEAD`, page-aligned
owned double-buffer, and trailing `MADV_DONTNEED` mechanism is reported in the
measured prototype attached to DS4 issue 437, not in the pinned mainline. The
issue attributes its prefill benefit to the no-cache double-buffer/trailing-
drop path and reports that unaligned reads and automatic read-ahead defeated
the intended file-backed-footprint bound. This is credible experimental prior
for a Prismwing falsifier, not production authority.

The DS4 confidence-gated speculative path and its acceptance regression tests
confirm that verification horizon is a control variable in a real engine.
Prismwing must price `q` against verifier work and corrected per-position route
union, rather than copying DS4's threshold or acceptance values.

## Prismwing state-retention audit

The suspected accepted-token replay topology is already absent. In
`src/text_endpoint.rs`, the verifier writes directly into the authoritative
K/V caches, `commit_jacobi_transaction` selects the retained prefix, and every
cache is truncated to that position. No accepted token is decoded again.
PW-0211 additionally appends the verifier-produced final-hidden rows and paired
input IDs directly to native-MTP history. Preserve this invariant; no repair or
new experiment is justified without contrary runtime evidence.

## Claims not adopted

The owner-supplied review's 2.4--2.6 TPS base-M5 report has no primary artifact
or immutable identity in the supplied file. It is excluded from Prismwing's
throughput constants. DS4's model shape, quantization, active-expert bytes,
hardware, and acceptance distribution also differ from MiMo-V2.5, so none of
its rates transfer numerically.

## Research decisions

1. Finish PW-0212 first. It is the cheapest corrected-route falsifier and can
   distinguish whether early knowledge exists before transport work.
2. Reserve PW-0213 for an exact page-aligned uncached stream-transport test.
   Separate page-cache footprint, read amplification, acquisition wall, Metal
   wait, and complete verifier wall. Do not bundle prediction into its first
   control.
3. Reserve PW-0214 for a corrected-route, cost-adaptive verification-horizon
   oracle over `q=2..8`, followed only by calibration-frozen policies.
4. Amend PW-0210's future fixture with a precision-crossing ledger and a
   horizontal co-consumer audit. Keep execution conditional on a changed cold
   critical-cut premise.

Every smaller repeatable full-path gain remains promotable under a conditional
disposition even when it misses 10, 25, or 50 TPS.
