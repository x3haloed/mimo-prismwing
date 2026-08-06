# PW-0105 — Cold weight-install tomography

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: to be recorded by the evidence manifest
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0095 cached oracle
  `75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8`
- Hardware/runtime: Apple M1 shared 16 GiB host; verified SSD checkpoint;
  existing PW-0100 bounded source-FP8 Metal path
- Related records: PW-0040, PW-0042, PW-0092, PW-0095, PW-0096,
  PW-0097 through PW-0101

## Shared construction contract

Capability and preservation: replay PW-0100's one-row incremental token through
the identical tokenizer, retained K/V, attention, router, source tensor views,
Metal kernel, sparse repair, residual, norm, LM-head, and greedy-output path.
Tomography may observe the failed L3 candidate but must not change its
arithmetic, ordering, selected experts, output token, correctness thresholds,
or release behavior. The target-faithful CPU and PW-0095 oracle remain the
semantic authorities.

Causal path and risk frontier: a routed expert currently maps three verified
source-FP8 projection views, copies each into fresh shared `MTLBuffer` objects,
encodes and submits one projection, synchronously waits, copies the result into
a CPU vector, performs BF16 staging/repair and SwiGLU, and repeats. The
least-proven boundary is source-backed pages through buffer installation and
CPU/GPU synchronization to the returned routed residual. Measure that real
path before replacing it.

Topology: keep one Rust endpoint authority and one existing Metal projection
executor. Instrument the existing function rather than creating a parallel
benchmark implementation. Attribute projections to their real layer, expert,
and gate/up/down identities. A diagnostic record flows upward from projection
to expert to layer to token; it does not become a second execution scheduler.

Embodiment depth: this experiment may add clocks and Darwin process counters
around existing mmap views, Metal allocation/copy, command encoding, submit,
wait, readback, CPU staging/repair, scatter, and explicit release. It may query
completed-command GPU timestamps when supported. It may not introduce
`bytesNoCopy`, Metal I/O, persistent arenas, argument buffers, indirect command
buffers, fused kernels, or async scheduling; those are candidate mechanisms
selected only after attribution.

Project constraints: run one full model process at a time under normative Gate
8. Keep the instrumentation bounded and evidence external to Git. This is a
diagnostic run, not accepted TPS, and a changed arithmetic result invalidates
the attribution.

## Hypothesis and measurement contract

PW-0100's warm routed-row component extrapolation missed its 75,725.919 ms
complete-token result by roughly 73 seconds. The hypothesis is that synchronous
source-page acquisition, `new_buffer_with_data` copies, per-projection resource
allocation/destruction, and 1,128 submit/wait/readback boundaries dominate the
missing wall time, rather than GPU arithmetic alone or a radically different
expert distribution.

For every one of the 376 routed experts and each of its three projections,
record monotonic wall intervals and cumulative process-counter deltas for:

1. checkpoint tensor lookup and FP8 view validation;
2. source weight and scale buffer installation separately from small activation,
   shape, LUT, and output allocation;
3. command-buffer and encoder creation, resource binding, and dispatch encoding;
4. commit call, synchronous wait, and completed-command GPU interval when the
   device exposes it;
5. shared-buffer readback into the CPU vector;
6. explicit Metal resource destruction/release;
7. dynamic-FP8 activation staging, BF16 rounding, sparse repair, SwiGLU, routed
   weighting/scatter, and layer-final BF16 rounding; and
8. process physical disk-read bytes, page-ins, minor faults, major faults, user
   CPU time, and system CPU time at the smallest practical boundaries.

The ledger must close within measured wrapper overhead: projection component
intervals sum to projection wall, expert intervals sum to expert wall, and
layer routed intervals reconcile with existing layer/token wall. Report p10,
median, p90, maximum, total, and the slowest records for all 1,128 projections
and 376 experts, split by stage, projection kind, layer, source shard, and
physical-read/page-fault state. Preserve raw per-projection and per-expert rows
in the external artifact rather than only aggregates.

Before the full walk, prove on deterministic tiny accounting tests that deltas
cannot underflow, totals close, unsupported GPU timestamps remain explicitly
`null`, and serialization retains every identity. Then run one layer-local real
fixture and require unchanged output bytes before authorizing exactly one full
incremental diagnostic.

## Gates

- **Validity:** same PW-0100 route sequence, greedy token 13, first failed layer
  and numerical metrics within deterministic replay tolerance, 376 experts,
  1,128 projections, and complete timing/counter identities. Any semantic drift
  rejects the run.
- **Attribution:** at least 95% of the incremental token wall must be assigned
  to measured layer/expert/projection/CPU-stage intervals or an explicitly named
  non-MoE component. No overlapping interval may be double-counted.
- **Decision:** promote the layer-transaction/no-copy/async branch when copy,
  allocation/release, wait/readback, or page-acquisition time explains at least
  half of routed-MoE wall, or when GPU active intervals occupy less than half of
  routed-MoE wall. If GPU execution itself dominates, prioritize kernel or
  representation work instead. Mixed evidence selects the largest measured
  removable category and preserves the rest.
- **Safety:** enforce every Gate 8 stop already required by PW-0100. Instrument
  snapshots must not replace phase-level host safety.

No optimization is promoted by this experiment. The next candidate must use
the exact category measured here and independently pass correctness plus a
minimum 2x cold routed-layer improvement against copied-buffer control.

## Result

Unexecuted.

## Decision

Instrument the current production-shaped failure before implementing a
Metal-ready artifact, no-copy mapping, Metal I/O, fused routed layer, or
asynchronous buffer arenas.
