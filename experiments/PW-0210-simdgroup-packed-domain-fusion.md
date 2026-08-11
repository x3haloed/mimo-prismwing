# PW-0210 — SIMD-group packed-domain fusion

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-10
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

## Cheap falsifier and gates

First test the smallest complete block-scaled projection plus its immediately
adjacent transform. Kill if fusion saves less than one full intermediate,
misses 1.5x complete-envelope speedup, or changes the reference result outside
the predeclared exactness class. Do not tune another matrix tile after failure.

Only after two representative shapes pass may it enter a real layer. Runtime
promotion requires at least 20% interleaved complete-layer gain and unchanged
endpoint tokens; a new endpoint ID is required for any combined claim.

## Decision

Unexecuted. This preserves the useful SIMD idea from the referenced discussion
while aiming it at the byte and synchronization boundary the repository has
actually measured.
