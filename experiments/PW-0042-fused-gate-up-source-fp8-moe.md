# PW-0042 — Fused gate/up source-FP8 MoE dispatch

- Status: complete
- Disposition: rejected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `df3df6f`; implementation dirty
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

The candidate uses one 4,096-threadgroup gate/up dispatch per expert while the
control uses two 2,048-threadgroup dispatches. All other route, expert, and
scatter work is shared. In candidate/control then control/candidate order,
candidate medians are `17.339375` and `17.483875` ms versus controls
`17.031917` and `17.077792` ms. Candidate mean is `17.411625` ms versus
`17.054855` ms: a 0.9795× speedup, or 2.09% slowdown. Candidate p10/p90 pairs
are `17.148958/17.776500` and `17.302042/17.715625` ms.

Cold candidate requests are 33.67 and 31.23 ms; complete process wall is 1.55
seconds. Logical and resident bytes are unchanged from PW-0039. The candidate
fixed-fixture routed-only diagnostic is 9.7758 TPS at `A=8`, `U=1.125`; it is
not endpoint TPS.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

The independent dual-projection scalar fixture covers distinct gate/up FP8
patterns, scales, output partitions, and all batch-eight positions; maximum
absolute error is `1.9790605e-8`. Candidate, repeat, and controls are
byte-identical with SHA-256
`ca5b3b38fb0c3fe27b0cd5b8b150a428f5b827ae04e6bc04eb6c02c264ef167e`.
Complete parity remains `1.709222e-6` relative L2 and `7.366907e-11` maximum
absolute error. Create-new rejection exits 1. Rust has 15 passing tests, Python
has 23, and clippy is clean with warnings denied.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0042`.
Its `SHA256SUMS` manifest hashes to
`b31752ce9c2fda1a56ab00164b2d47dce23984c68c4e545ad931f6b0b4f1fa75`.

## Decision

Reject fused gate/up dispatch. Eliminating nine command-encoder dispatches and
doubling the projection grid does not improve the real dynamic MoE path. Retain
the correctness-backed kernel as a diagnostic, but keep PW-0039's separate
gate/up schedule as the default.

Together with PW-0040 and PW-0041, this result rules out dispatch enlargement,
union phase ordering, and exact-F32 matrix expansion as near-term fixes. The
next performance branch must change the inner direct-FP8 arithmetic/data path
or reduce model work under an explicitly validated fidelity contract.
