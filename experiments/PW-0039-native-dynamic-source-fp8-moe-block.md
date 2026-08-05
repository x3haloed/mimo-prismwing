# PW-0039 — Native dynamically routed source-FP8 MoE block

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `bda836b`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
