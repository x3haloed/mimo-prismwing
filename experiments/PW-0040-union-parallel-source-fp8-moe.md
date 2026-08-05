# PW-0040 — Union-parallel source-FP8 MoE schedule

- Status: complete
- Disposition: rejected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `b481117`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0039 exact router, input,
  expert artifacts, and Torch source-FP8 reference
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal;
  selected exact tensor artifacts on external platter
- Related records: PW-0036, PW-0037, PW-0038, PW-0039

## Hypothesis and mechanism

PW-0039 executes each selected expert's gate, up, SwiGLU, down, and scatter
before beginning the next expert. The nine-expert union moves about 227 MB of
source weights but realizes only about 13 GB/s at a 17.1 ms mean median, far
below the real single-projection Metal primitive's measured bandwidth. Packing
the selected exact tensors into explicit expert-major buffers and dispatching
each projection phase over the whole union may expose enough independent work
to improve occupancy and reduce inter-expert pipeline hazards.

## Contract

Add a union-parallel mode to the dynamic native MoE path. Pass only if:

1. preserve PW-0039's exact artifact, router, route-selection, and fail-closed
   authorities. Packing may concatenate only already validated source bytes;
   it may not reinterpret, quantize, or alter weights or scales;
2. add expert-major source-FP8 GEMM8 and SwiGLU kernel semantics with explicit
   expert grid dimensions and non-overlapping expert outputs. Add deterministic
   scalar fixtures before relying on either new indexing rule;
3. every complete timed request still recomputes native routes and derives
   gather/weight/position/scatter state from them. Dispatch all selected experts
   gate-wide, then up-wide, then SwiGLU-wide, then down-wide, then scatter;
4. complete output must meet PW-0039's `4e-5` relative-L2 and `3e-8` maximum-
   absolute-error gates versus independent Torch and be byte-identical across
   processes. The PW-0039 serial dynamic path is the interleaved performance
   control and must retain its correctness;
5. use two paired process orders, each with five warmups and 30 complete-request
   measurements. Promote only if the union-parallel mean median is at least 20%
   faster than control and at most 14 ms. Report p10/median/p90 and the measured
   routed-only `A=8`, `U=1.125` diagnostic; no endpoint TPS claim;
6. report packed/logical/resident bytes, packing and compile wall separately,
   cold/warm state, full process wall, batch, concurrency, accepted tokens,
   `A`, `U`, hardware, commit, and the fixed-input/preloaded-union limitation.

Passing promotes the union-parallel execution schedule for the exact dynamic
MoE component. Failure rejects this scheduling mechanism, not source FP8,
dynamic routing, or future storage/quantization alternatives.

## Baseline and candidate

Control is PW-0039's promoted serial-expert dynamic runtime. Candidate packs
the same validated nine-expert source representation and parallelizes across
the expert-union dimension while preserving all equations and route authority.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0040`.

## Isolated attribution

The first attempt failed before measurement because Metal requires the
multidimensional dispatch built-ins to use matching vector types. Changing the
group, thread-index, and thread-count declarations to `uint3` corrected the ABI;
the failure is preserved in raw evidence.

The candidate concatenates the exact validated expert-major weight and scale
bytes, dispatches all nine gates, all ups, one flat union SwiGLU, all downs,
then the nine scatters. In candidate/control then control/candidate process
orders, candidate medians are `17.809458` and `17.822000` ms while controls are
`17.101917` and `17.107833` ms. Candidate mean is `17.815729` ms versus
`17.104875` ms control: a 0.9601× speedup, or 4.16% slowdown. Candidate p10/p90
pairs are `17.616125/18.153417` and `17.447958/18.144500` ms.

Packing preserves the same `231,005,184` logical source/I/O bytes but this
shared diagnostic implementation retains the serial buffers too, so reported
Metal buffers rise from `232,520,704` to `463,197,184` bytes. Complete candidate
process wall is 1.71 seconds and peak process footprint is 903,843,712 bytes.
Removing duplicate control buffers could reduce embodiment cost, but cannot
satisfy the missing 20% speedup and is not a reason to rerun this schedule.

The candidate's fixed-fixture routed-only `A=8`, `U=1.125` diagnostic is
9.5541 TPS, below the promoted schedule. It is not endpoint TPS.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

The independent two-expert, batch-eight, `128×128` scalar fixture covers every
expert-major weight, scale, input, and output offset; maximum absolute error is
`1.6763806e-8`. Candidate, repeat, and both controls are byte-identical with
SHA-256
`ca5b3b38fb0c3fe27b0cd5b8b150a428f5b827ae04e6bc04eb6c02c264ef167e`.
Complete output remains `1.709222e-6` relative L2 and `7.366907e-11` maximum
absolute error versus independent Torch source FP8. Create-new rejection exits
1. Rust has 15 passing tests, Python has 21, and clippy is clean with warnings
denied.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0040`.
Its `SHA256SUMS` manifest hashes to
`ff0bba21625b77cd98c046831e0b2f78c9d579b51277d1724720db16eb086ca2`.

## Decision

Reject union-parallel phase scheduling. The proposed occupancy mechanism is
not present on this workload: broader phase dispatch is consistently slower
than keeping each expert's intermediates temporally local. Retain the exact
expert-major kernel and command only as a correctness-backed diagnostic, not a
default.

PW-0039 remains the promoted dynamic target-faithful MoE schedule. A further
performance experiment must change a different mechanism—such as avoiding
padding work, reducing exact source bytes, or using a better matrix primitive—
rather than merely reordering the same padded expert work.
