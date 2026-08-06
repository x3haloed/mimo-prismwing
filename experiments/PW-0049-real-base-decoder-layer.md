# PW-0049 — Real base-model decoder layer

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract `08c6fb7`; result implementation is the
  commit containing this record
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; the completed checkpoint and
  every used artifact must pass the committed model lock before execution
- Hardware, OS, compiler, storage, memory pressure: Apple M1 Mac mini,
  Macmini9,1, 16 GB; macOS 26.6 (25G72); Rust 1.96.0 release; verified
  checkpoint and selected evidence on internal APFS SSD; memory pressure not
  independently sampled
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

Raw evidence is under
`/Users/chad/Models/mimo-prismwing/evidence/PW-0049`. The final 74-file
`SHA256SUMS` manifest hashes to
`afabfbc14b33eba0e9af92e2c8cefd206d3e8aa7cc354ed272ee08fed5bca5e6`;
the complete checkpoint-verification manifest hashes to
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`.

## Isolated attribution

The first native Metal-only composition failed honestly at `7.390976e-6`
maximum final error. Stable RMS reduction lowered MoE-input maximum error from
`5.245209e-6` to `1.430512e-6`. Increasing FP8 Metal reduction width and a
compensated FP8 kernel did not cross the final gate and were rejected.

A bounded Rust-owned Accelerate path did. It decodes only the independently
selected 56-expert source-FP8 union to F32, executes the 64 real route
placements with native SGEMM, and scatters in Rust. The same Accelerate SGEMM
for source-FP8 QKV reduced QKV relative L2 to `1.4169e-7`. No Python output,
frozen route, or oracle activation supplies execution.

## End-to-end result

Two final-code process walls are `60.055` and `59.554` seconds. They include a
complete SHA-256 pass over the 34.37 GB source shard, source validation,
selected-expert decoding, compilation, warmups, and measurements, so they are
evidence-process walls rather than resident layer latency.

Resident component diagnostics are approximately 272--274 ms QKV, 11--12 ms
CPU SWA, 7--8 ms output projection, and 227--231 ms median routed MoE. The MoE
uses one warmup and five measurements, has `U=7.0`, and reports 7,058,908,160
resident bytes. This is a faithful correctness baseline and decisively not a
performance default or endpoint TPS result.

## Correctness result

The complete final state passes at relative L2 `3.642936e-7` and maximum
absolute error `2.384186e-6`. QKV, post-attention, MoE-input, and routed-MoE
relative L2 values are respectively `1.416872e-7`, `9.070980e-8`,
`1.035320e-7`, and `5.464336e-7`. Route sets are exact, maximum route-weight
error is `6.705523e-8`, and the minimum top-k boundary margin is
`7.410049e-4`.

Independent final-code processes produce byte-identical MoE inputs, MoE
outputs, and final states. Final output SHA-256 is
`efa95f0df1c08f4fea0e049c0f66f54d71ae06f8f33024c0406295cad411f2fb`.
Rust has 18 passing tests, Python has 33, clippy is clean with warnings denied,
and the release build succeeds.

## Decision

Promote this command as the first complete target-faithful base decoder-layer
correctness baseline. It closes the learned attention-to-routing-to-final-
residual causal boundary and unblocks representative activation work.

Do not promote its expanded-F32 Accelerate expert embodiment or infer endpoint
TPS. The seeded layer-local input's `U=7.0` supersedes the assumption that
PW-0039's `U=1.125` frozen fixture is representative. The next work must use
this trace to falsify or promote the predeclared topology/embodiment jumps, and
must still build the slow complete text endpoint before a delivery claim.
