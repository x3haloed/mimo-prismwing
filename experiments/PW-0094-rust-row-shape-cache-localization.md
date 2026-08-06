# PW-0094 — Rust row-shape versus cache localization

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: clean implementation and execution at
  `7737d2812d500fe76c2629979fa04c33d0ad0cd2`
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

The trace-only schema and admission tests pass: the Rust full-prefix trace
accepts exactly the frozen 27-token chat prefix plus `[264]`, while the slow
endpoint rejects the trace-only fixture. Schema-1/raw and schema-2/chat tests
remain unchanged. The fixture SHA-256 is
`138a2f99449d5c11b7411e32afc6904ee37fd5154e25b4f51c276fb31ac47100`.

One preflight invocation failed before model inference because the scoped
evidence parent did not exist; it created no trace. After creating only that
parent and supplying the exact commit, the one authorized Rust-28 process
completed in 785,197.518 ms. Its manifest is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0094/rust28-001/manifest.json`
with SHA-256
`e7f316dd14abd0f45ec134ce7b045caabddde49f2301141fc86260f036a1db1a`.

The first discriminator clears completely. Existing full-prefix comparison
reports exact PyTorch-28 versus Rust-28 embedding, all 48 layer finals, final
RMSNorm, every route set and route weight, and all 152,576 final logits. There
is no first failure; final-logit equality is 100%, relative L2 and maximum
error are zero, and both logit files hash to
`05ce9d5cdbcf55aa70f56ad20a9885263e4a0ddcbd1e1d3985b43cebfdcc4050`.
The comparison report SHA-256 is
`1c98bb1ce5086d519ce2a3d63079f7dda11c97f5abc3cb9f91fc3fc7c9960b00`.

The second discriminator therefore confirms a real difference between the
exact Rust-28 whole-sequence result and PW-0092 Rust 27+1. Its final metrics
are exactly PW-0093's: 8.3309% logit equality, `0.0246957` relative L2, `0.5`
maximum error, with greedy token 13 preserved. Route-weight drift at the
appended position starts at layer 1, expert order first changes at layer 3,
and expert sets first change at layer 11. This narrows the cause to the
one-row incremental execution, but does not yet distinguish retained K/V from
the matrix backend's one-row reduction topology.

The Rust-28 ledger records 67,098,966,912 logical source bytes,
68,166,766,592 actual process disk bytes, 7,125 FP8 matrix expansions, 49 BF16
expansions, and 2,358 unique-expert executions. Batch and concurrency were one,
accepted tokens were zero, and every layer's `U` is preserved in the raw
manifest. This is diagnostic, not accepted TPS.

Gate 8 passed with at least 74% system memory-free pressure, peak RSS
4,154,261,504 bytes, maximum/final physical footprint 2,890,079,808 bytes,
zero swap growth, zero new throttled pages, and ChatGPT, WindowServer, both
`nxnode` processes, and both Syncthing processes resident at the final
boundary.

## Decision

Promote the schema-3 trace-only fixture as correctness infrastructure and the
PyTorch-28/Rust-28 exact result as the row-shape authority. Preserve the
Rust-28/Rust-27+1 mismatch; do not describe retained K/V as correct or broken
yet, and do not relax any gate.

The next cheapest decisive experiment is an independent PyTorch cached walk:
run the exact 27-token prefill while retaining source K/V, then consume only
token 264 through those caches. If its per-layer appended states, routes, and
logits match Rust 27+1, cache semantics clear and whole-sequence divergence is
row-shape arithmetic. If it matches PyTorch-28 instead, localize Rust cache
state from the first divergent layer. No throughput-model constant changes.
