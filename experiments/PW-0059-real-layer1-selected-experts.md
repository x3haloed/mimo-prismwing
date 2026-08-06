# PW-0059 — Real layer-1 selected-expert execution

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: diagnostic contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0058 comparison
  `1cea804681d175b0fe4c359aafe120a6659f3dc45d5c125a3f5aad5ca36880d2`;
  checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, PW-0050 safety contract
- Related records: PW-0039, PW-0049, PW-0056, PW-0058

## Hypothesis and mechanism

PW-0058 clears the causal real prefix through routed layer-1 expert selection.
Its 27 positions select 216 placements across 28 unique source-FP8 experts.
The first untested semantic is therefore dynamic per-expert gather, gate/up
projection, BF16 SwiGLU staging, down projection, weighted scatter, and the
second residual—not attention or routing.

## Contract

Extend the Rust trace through the existing production `routed_mlp` authority.
It must derive layer 0, layer-1 attention, router decisions, expert schedule,
expert outputs, scatter, and residual causally in one process. It may not
consume oracle routes or intermediate arrays. Record the sorted expert-major
placement schedule and capture gate, up, SwiGLU, and down results in that exact
schedule, followed by routed output and final layer state.

Build an independent PyTorch oracle from PW-0058 oracle run 001. Validate its
manifest, source-input hash, captures, checkpoint revision and complete
verification before using its independently selected routes and normalized
MoE input. Resolve every selected expert tensor through the checkpoint index;
fail closed on tensor name, shard, dtype, shape, scale grid, file identity, or
route-placement disagreement. Execute only the 28 independently selected
experts and never consume a Rust route or expert intermediate.

Every categorical authority and placement must match exactly. BF16 captures
pass at relative L2 at most `5e-4`, maximum absolute error at most `2e-2`, and
payload equality at least 99%. The final layer state must additionally satisfy
relative L2 at most `4e-5` and maximum absolute error at most `3e-6`, preserving
PW-0049's stronger complete-layer gate. Any first failure is localized before
another whole-model walk.

Retain the PW-0050 shared-host stops after causal layer 0, layer-1 attention,
routing, every completed expert, scatter/residual, and capture writes. Release
file pages and allocator transients before completed-phase measurement. Stop
below 20% system-free memory, above 8 GiB current or peak process footprint,
above 4 GiB post-phase footprint, above 512 MiB swap growth, on any new
throttled page, or when a start-resident protected service disappears.

Generated arrays remain external and content-addressed. Record logical and
actual bytes, expert executions, route union, wall time, cold/warm state,
batch 1, 27 prompt positions, concurrency 1, accepted tokens 0, hardware,
commit, and safety telemetry. These are correctness diagnostics and cannot
change hosted thresholds or become accepted TPS.

## Result

Unexecuted.

## Decision

Unexecuted.
