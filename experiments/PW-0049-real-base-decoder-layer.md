# PW-0049 — Real base-model decoder layer

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract pending; no execution
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; the completed checkpoint and
  every used artifact must pass the committed model lock before execution
- Hardware, OS, compiler, storage, memory pressure: Apple M1 Mac mini;
  remaining environment fields unmeasured
- Related records: PW-0029, PW-0030, PW-0039, PW-0044

## Hypothesis and mechanism

The promoted native attention and dynamic source-FP8 MoE components can be
joined across the real learned layer-43 boundary. A deterministic
production-width base-model input should flow through input RMSNorm, learned
fused QKV, source-faithful SWA attention, BF16 output projection, the first
residual, post-attention RMSNorm, native noaux-tc routing, newly selected exact
source-FP8 experts, weighted scatter, and the final residual under one
fail-closed native authority.

This crosses the missing base decoder-layer causal boundary. It does not claim
that a deterministic layer-local input is a real decode trace, that one layer
establishes accumulated fidelity, or that component timing is endpoint TPS.

## Contract

Use base layer 43, whose routed MoE and existing real fixtures make the
attention-to-routing seam directly comparable with PW-0039. Freeze a seeded,
finite, production-width sequence of at least 128 positions; compute the full
SWA K/V history and the final eight query positions so the routed execution
unit remains batch eight. Record the seed, generator, input hash, exact
positions, RoPE policy, and every tensor authority.

Pass only if:

1. after the incoming rsync exits successfully, the checkpoint census and
   complete model lock pass before any shard supplies source truth. Validate
   every used layer-43 tensor by name, dtype, shape, bounds, and locked shard
   hash; unknown layouts or split tensor authorities fail closed;
2. an independent readable Python source oracle implements the published
   layer equations and emits hash-bound captures at normalized input, Q/K/V,
   attention output, projected/residual state, normalized MoE input, route IDs
   and weights, routed residual, and final layer state. It may execute only the
   experts selected by its own routes, but may not consume native route output;
3. the native path consumes the same frozen input and derives all downstream
   state causally. No PW-0039 route IDs, route weights, selected union, expert
   schedule, attention output, or post-attention input may supply execution;
4. source-faithful attention preserves layer-43 SWA semantics: RMSNorm epsilon
   `1e-5`, partial RoPE over the first 64 of 192 Q/K dimensions with theta
   `10000`, 64 Q heads, eight KV heads, 192-wide Q/K, 128-wide V, value scale
   `0.707`, window 128, learned sink logits, causal masking, BF16 output
   projection, and both residual edges;
5. sampled raw FP8/BF16 scalar checks agree with independently decoded source
   arithmetic at maximum absolute error `2e-4`. Native route sets equal the
   independent oracle, route-weight maximum absolute error is at most `5e-7`,
   and complete final-state relative L2 is at most `4e-5` with maximum absolute
   error at most `3e-6`; repeated native outputs are byte-identical;
6. a correctness fixture is committed with the new layer semantic. It binds
   captures and selected expert artifacts by hash, remains small enough for
   Git or records an external content-addressed manifest, and includes negative
   tests for wrong revision, tensor shape, source shard, input hash, position,
   and route/expert authority;
7. after correctness passes, report cold and warm complete-layer timing with
   batch eight, concurrency one, accepted tokens `0`, `A=0`, observed per-layer
   `U`, logical and actual bytes, resident buffers, storage state, hardware,
   commit, and full process wall. Any affine8 or other approximate cache is a
   separately named L3 candidate and cannot supply the target-faithful result.

Passing promotes one real base-model decoder-layer vertical slice and makes
its intermediate traces available to PW-0044 through PW-0046. It does not
promote a complete text endpoint, accumulated whole-model parity, speculative
acceptance, or accepted TPS.

Kill or split the implementation if the source-faithful layer cannot fit the
16 GiB host, if independent and native semantics disagree below the accelerated
boundary, or if selected-expert materialization requires an unbounded copy of
the expert bank. A split record must preserve the exact failed boundary.

## Baseline and candidate

Baseline is the independent readable source-FP8/BF16 layer-43 oracle with
uncompressed source K/V. Candidate is one Rust-owned native command consuming
validated source checkpoint views and runtime-derived routes. PW-0039 is a
component control only; its frozen post-attention input and expert union are
not a valid base-layer baseline.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0049`.

## Isolated attribution

Unexecuted. The first attribution is semantic composition across learned
attention, residual/norm, and dynamic expert selection. Performance attribution
begins only after that full causal path passes.

## End-to-end result

Unexecuted. No endpoint or TPS claim exists.

## Correctness result

Unexecuted.

## Decision

Unexecuted. This is the immediate checkpoint-unblocked risk frontier and the
prerequisite for representative route traces and the slow complete text path.
