# PW-0042 — Fused gate/up source-FP8 MoE dispatch

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `25621db`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0039 exact authorities
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal
- Related records: PW-0036, PW-0039, PW-0040, PW-0041

## Hypothesis and mechanism

PW-0039 dispatches separate 2,048-row gate and up projections for every expert.
They use the same gathered input and independent weights but each small dispatch
realizes much less effective bandwidth than the larger PW-0033 projection.
One 4,096-threadgroup dispatch that selects gate or up by row can enlarge the
work unit and remove nine dispatch boundaries while retaining per-expert
gate/up/SwiGLU/down temporal locality that PW-0040 showed is beneficial.

## Contract

Add a fused gate/up mode to the promoted dynamic native MoE runtime. Pass only
if:

1. use the same separately validated source FP8 gate/up weights and F32 scale
   grids. The kernel may select between them explicitly but may not concatenate,
   reinterpret, quantize, or alter source bytes;
2. add an independent deterministic scalar fixture covering both projection
   branches, expert batch-eight input indexing, output partition, block scales,
   and every FP8 encoding needed by the fixture;
3. preserve PW-0039 dynamic router authority, derived heterogeneous scheduling,
   SwiGLU, down, weighting, scatter, and all complete-output correctness gates.
   Candidate and control outputs must be byte-identical;
4. use paired candidate/control then control/candidate process orders, each with
   five warmups and 30 complete-request measurements. Promote only if candidate
   mean median is at least 10% faster than PW-0039 control and at most 15.5 ms;
5. report p10/median/p90, cold and complete process wall, logical/resident bytes,
   batch eight, concurrency one, accepted tokens, `A=8`, `U=1.125`, hardware,
   commit, cache state, and routed-only diagnostic. No endpoint TPS claim;
6. retain PW-0039 as default on any correctness or performance-gate failure.

Passing promotes fused gate/up dispatch inside the exact dynamic MoE component.
It does not establish a complete layer, representative routes, or endpoint TPS.

## Baseline and candidate

Control is PW-0039's separate gate/up source-FP8 dispatches. Candidate changes
only those two dispatches into one dual-branch dispatch per selected expert.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0042`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
