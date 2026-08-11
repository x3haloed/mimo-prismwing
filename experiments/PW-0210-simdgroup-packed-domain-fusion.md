# PW-0210 — SIMD-group packed-domain fusion

- Status: in progress
- Disposition: width-128 gate/up-to-SwiGLU falsifier authorized by PW-0209
- Date: 2026-08-11
- Execution mode: L1 when bit-exact/function-preserving; approximate codecs separately named L3
- Related records: PW-0043, PW-0111, PW-0196 through PW-0202, PW-0205

## Hypothesis and mechanism

The rejected PW-0043 matrix tile does not establish that all SIMD-group uses
are unhelpful. The stronger attack is not “portable SIMD replaces GEMM.” It is
that unpacking, block-scale application, masks, RoPE, reductions, routing,
scatter, and intermediate materialization form an irregular envelope that can
cost more bytes and barriers than their arithmetic warrants.

Hypothesis: keep source FP8/low-bit values in their packed code domain until an
M1 Metal SIMD group consumes them, fuse scale/reduction and the next local
transform, and avoid at least one full intermediate round trip. A complete
projection envelope should improve at least 1.5x, with identical declared
arithmetic and lower measured bytes/barriers.

## Contract

Query `threadExecutionWidth`; never assume a warp size. Add a deterministic
correctness fixture covering nonuniform block scales, tails, masks, NaNs,
signed zero, and reduction order before a real tensor. Preserve the existing
GEMM/projection control. VectorWare's NVIDIA mapping is a design prior only;
the implementation is native Metal and uses only M1-supported features.

Track packed/source bytes, decoded bytes materialized, threadgroup and device
traffic, command buffers, barriers, GPU time, complete wall, and numerical
error. Storage-only or unpack-only speed is diagnostic.

Before kernel work, add a precision-crossing ledger for each large candidate
intermediate: producer arithmetic precision, stored precision, consumer load
conversion, and actual multiply precision. A move such as F32-memory-F16 to
F16-memory-F16 is L1 only when a discriminating fixture proves the earlier
narrowing is bit-identical at the consumer boundary; otherwise it is a
separately named L3 approximation. Also audit independent immediate consumers
of the same large input for horizontal fusion, while preserving each original
reduction tree. DS4 commit `84cc882352757baf628a1776badf7cc54d584e28` is a
pinned design prior for both shapes, not Prismwing performance evidence.

## Cheap falsifier and gates

Before kernel work, bind the proposal to PW-0111's cold decomposition. Even
granting a 1.5x improvement to its entire 8.383 ms GPU interval predicts only
`107.007 ms` versus `109.801 ms`, or `1.026x`, for the cold routed layer. This
misses the 20% complete-layer gate, so execution is conditional on a named
premise that makes the projection envelope material: wider amortized compute,
changed storage, corrected-layout executable-byte reduction, or substantially
greater legal residency. Recompute the enclosing bound when that premise
exists.

First test the smallest complete block-scaled projection plus its immediately
adjacent transform. Kill if fusion saves less than one full intermediate,
misses 1.5x complete-envelope speedup, or changes the reference result outside
the predeclared exactness class. Freeze the byte-exact packed-FP8 control, use
compile-time width variants, sweep tile shape on the deterministic fixture,
and measure under the promoted command-buffer topology. Stop after the first
negative complete-envelope result; do not tune another matrix tile.

Only after two representative shapes pass may it enter a real layer. Runtime
promotion requires at least 20% interleaved complete-layer gain and unchanged
endpoint tokens; a new endpoint ID is required for any combined claim.

## Precision-crossing and co-consumer ledger

PW-0209 supplies the named wider-amortized-compute premise and a real
context-128 authority. For its routed gate/up envelope, both source weights and
activations remain packed FP8 codes with independent F32 block scales. Each
projection accumulates in F32. The current SwiGLU consumer rounds gate and up
to BF16, evaluates SiLU, rounds that result to BF16, multiplies by rounded up,
and rounds the hidden value to BF16. The down projection then dynamically
encodes that BF16-widened F32 hidden row to group-128 FP8.

The exact candidate therefore fuses the two projection reductions through
the unchanged BF16-staged SwiGLU and materializes only hidden. It removes two
F32 projection writes and two F32 consumer reads: 33,554,432 device bytes for
PW-0209's 1,024 routed placements. It does not change weight, scale,
activation-code, hidden, or down-projection traffic. Storing hidden as F16 is
not L1: the existing consumer first observes BF16-widened F32, not half.

The router is the only immediate horizontal co-consumer of normalized MoE
input. It must complete before the expert schedule and source bindings exist;
combining it with routed gate/up would change the authoritative dependency and
is excluded from this falsifier. The candidate may remove dispatches only as a
consequence of vertical producer-consumer fusion, not as an independent claim.

## Decision

Unexecuted and conditional. This preserves the useful SIMD idea while the
current cold critical-cut accounting proves it cannot pass its layer gate.
Execute only after a named premise moves projection execution onto the active
critical cut; do not promote a warm-only result as cold endpoint TPS.
