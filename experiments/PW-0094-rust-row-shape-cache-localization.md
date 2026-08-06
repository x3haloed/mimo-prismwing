# PW-0094 — Rust row-shape versus cache localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0093 oracle manifest
  `f143a6c9ee526eaddd40e809ffa18e20a3eb1cbc9e0b5d0af2a86ba80757b596`;
  PW-0092 run 001
  `18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust whole-sequence trace and retained-cache endpoint
- Related records: PW-0060, PW-0091 through PW-0093

## Hypothesis and mechanism

PW-0093's direct comparison confounds retained K/V with row-count-dependent
matrix arithmetic. Add one fail-closed trace-only fixture that appends token
264 to the frozen 27-token prompt. The existing Rust full-prefix trace will
evaluate all 28 tokens in one call and capture every layer. The production
incremental endpoint must reject this trace-only fixture.

Compare PyTorch-28 with Rust-28 first. Exact parity means row-count-dependent
source arithmetic is represented correctly and permits the second comparison:
Rust-28's appended-position state, routes, and logits versus Rust 27+1. If the
first comparison fails, localize source row-shape arithmetic before drawing a
cache conclusion. If the first passes but the second fails, localize retained
K/V from the first divergent layer.

## Gates

Add a deterministic fixture test proving the schema-3 token sequence is exactly
the frozen prompt plus `[264]`, that the trace accepts it, and that the slow
endpoint cannot. Preserve schema-1/raw and schema-2/chat behavior. Unknown
append tokens, multiple tokens, revisions, hashes, layouts, or safety policies
fail closed.

For PyTorch-28 versus Rust-28, apply PW-0091's unchanged capture and route
gates, including complete final-logit byte identity. For Rust-28 versus Rust
27+1, require exact appended-position BF16 layer states if available, exact
expert sets/order and route weights, and exact final logits. Do not average a
failure away. Record complete wall, bytes, matrix expansions, route union,
cold/warm state, batch 1, concurrency 1, `A=0`, and every layer's `U`.

Enforce normative Gate 8 at open, every layer, LM head, and capture boundary:
minimum free memory 20%, peak/current process memory at most 8 GiB,
post-release footprint at most 4 GiB, swap growth at most 512 MiB, no new
throttled pages, release relief, and health of ChatGPT, WindowServer, `nxnode`,
and Syncthing. Preserve stopped evidence. One Rust-28 run is authorized; repeat
only if comparison is nondeterministic or ambiguous.

No result from this correctness localization is accepted TPS or a promoted
performance default.

## Result

Unexecuted.

## Decision

Unexecuted.
