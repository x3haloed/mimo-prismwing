# PW-0039 — Native dynamically routed source-FP8 MoE block

- Status: complete
- Disposition: production
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `d8b5ade`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; exact layer-43 router SHA-256
  `12c1579d28b78dd69ec9342eb9d1f378efc5aa3c2f2a28b5ec73578e6a8bbcdd`;
  PW-0037 manifest SHA-256
  `a2b30be7ab767c754fd4680887420246dcd28d314bff242b144f664a8ff12470`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal;
  selected exact tensor artifacts on external platter
- Related records: PW-0016, PW-0036, PW-0037, PW-0038

## Hypothesis and mechanism

PW-0038 makes exact native route decisions in about 0.34 ms, while PW-0037
executes the resulting nine-expert source-FP8 schedule in about 16.15 ms. A
single Rust-owned request can dispatch the router, synchronously derive the
heterogeneous gather/weight/scatter schedule from its output, and execute the
experts without using frozen route IDs or weights as runtime authority.

## Contract

Add `metal-dynamic-fp8-moe-block`. Pass only if:

1. validate the exact router, PW-0037 manifest, input, reference output, and
   every selected expert tensor artifact by pinned hash, dtype, shape, and
   semantic identity. The frozen routes remain parity oracle only;
2. each complete timed request dispatches the exact F32 router projection and
   waits for it, computes fail-closed native noaux-tc selection, then derives
   expert union, gathers, route weights, positions, and scatter shapes from
   those computed decisions before dispatching source-FP8 experts. Unknown
   selected experts, non-finite states, duplicate experts, tied top-eight
   boundaries, or invalid normalization fail closed;
3. native selected sets and route weights pass PW-0038's independent Torch
   gates. The derived union/counts must be exactly the observed nine-expert
   `{8×7,5,3}` fixture, but those values may not be copied into execution from
   the oracle manifest;
4. complete `8×4,096` output relative L2 versus independent Torch source FP8
   must be at most `4e-5`, maximum absolute error at most `3e-8`; repeated
   outputs must be byte-identical and create-new behavior must reject overwrite;
5. after five warmups, 30 serialized complete-request measurements report
   p10/median/p90 and must include router dispatch/wait, CPU decision and
   schedule materialization, and expert/scatter dispatch/wait. Median must be
   at most 20 ms. Report PW-0037 control context and the measured routed-only
   `A=8` diagnostic over 47 layers, but no endpoint TPS claim;
6. report cold and warm state, full process wall, batch eight, concurrency one,
   accepted tokens, `A`, observed `U`, logical/resident bytes, hardware,
   commit, and the fixed-input/selected-union limitation.

Passing promotes the first complete dynamically routed native MoE block on the
real layer-43 fixture. It does not establish representative route reuse, a
transformer layer, storage-cold behavior, whole-model fidelity, or endpoint
TPS.

## Baseline and candidate

Baseline is PW-0037's fixture-scheduled target-faithful 16.1513 ms mean median.
Candidate composes PW-0038's promoted router with the PW-0037 expert runtime in
one causally connected measurement. The independent PW-0037 Torch output and
route manifest remain correctness oracles.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0039`.

## Isolated attribution

Each measured request starts before router dispatch, waits for the `256×4,096`
F32 projection, computes sigmoid/correction-biased top eight and normalized
weights in Rust, derives the nine expert schedules, rewrites gather, weight,
position, and scatter-shape buffers, then dispatches all source-FP8 expert and
scatter work. No frozen route value supplies an execution buffer.

Two final-code 30-measurement medians after five warmups are `17.075666` and
`17.154208` ms, with p10/p90 pairs `16.777292/17.298958` and
`16.798542/17.440125` ms. Their mean is `17.114937` ms, `0.963646` ms (5.97%)
above PW-0037's frozen-schedule 16.151292 ms control. The final cold request is
27.786792 ms and complete process wall is 1.49 seconds. The complete resident
path reports `231,005,184` logical source/I/O bytes and `232,520,704` resident
buffer bytes.

At this exact fixture's `A=8`, `U=1.125`, repeating the mean component cost
over 47 routed layers gives 9.9453 routed-only accepted TPS. This excludes all
non-MoE work, storage misses, draft-model cost, acceptance failures, and
endpoint overhead, so it is not endpoint TPS.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Every iteration recomputes the exact nine-expert union and `{8×7,5,3}` count
distribution. Selected sets equal the independent Torch oracle, route-weight
maximum absolute error is `1.4901161e-8`, and the minimum top-eight boundary
margin is `1.5354156e-4`.

Complete output has relative L2 `1.709222e-6` and maximum absolute error
`7.366907e-11` versus independent Torch source FP8. Separate final-code
processes are byte-identical with SHA-256
`ca5b3b38fb0c3fe27b0cd5b8b150a428f5b827ae04e6bc04eb6c02c264ef167e`.
Create-new rejection exits 1. The refactored frozen-schedule control still
produces PW-0037's exact `fae802...` output at a 16.132875 ms median. Rust has
15 passing tests, Python has 21, and clippy is clean with warnings denied.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0039`.
Its `SHA256SUMS` manifest hashes to
`e6209140991c7e6362f628c62e52e3d60a6545362442e59bee1e5cc92f9a37f1`.

## Decision

Promote the dynamically routed target-faithful layer-43 MoE block. For this
real fixture, native router decisions now causally determine expert gather,
execution, weighting, scatter, and observable output in one measured request.
The frozen route manifest is retained only as an independent correctness
oracle and tensor-authority inventory.

Do not generalize this fixed-input union to real decode distributions or claim
a complete transformer layer. The next causal boundary is a base decoder-layer
input and its learned attention/norm tensors; exact EP0 source material remains
the preferred authority when its durable download completes.
