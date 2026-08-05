# PW-0040 — Union-parallel source-FP8 MoE schedule

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `672e84d`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
