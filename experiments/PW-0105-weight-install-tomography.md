# PW-0105 — Cold weight-install tomography

- Status: complete
- Disposition: scope-decision
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: profiled runtime
  `e67fe4b3927bf027b5fa91f176435989576715e8`; analysis
  `18285374cfdc3131a8ae18fe1be5b58f87f85f82`; clean trees
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

The real layer-4 seam first proved that observation does not alter the causal
path. The profiled run exactly reproduces PW-0101's route IDs, route weights,
all expert-stage parity metrics, routed residual, and final residual. All 24
command buffers expose valid Apple M1 GPU timestamps, and Gate 8 passes. Its
report is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0105/layer4-002/report.json`,
SHA-256
`2de060cbd7afd22e2d21615883127649a8f564b513a9aa8d1abfca52ca4d18bf`.

The sole authorized full walk then captures 376 real routed experts, 1,128
gate/up/down projections, and all 47 routed layers. It reproduces generated
tokens `[264, 13]`, the known first failure at layer 4, and the rejected final
norm/logit behavior rather than manufacturing a pass. Incremental wall is
76,077.168 ms, only 0.4638% above PW-0100's 75,725.919 ms despite the added
counters. The raw 7.5 MiB report hashes to
`49c1f85b24e8864d43a3a901de9c7c40e8745a4427599248bd937abba4ce3e11`.

The token partitions exactly into 40,560.763 ms of routed MoE, 28,673.258 ms
of non-MoE work inside the 48 layer intervals, and 6,843.147 ms outside those
intervals. Within routed MoE:

- tensor lookup, FP8-scale validation, and page acquisition consume
  16,790.296 ms;
- `release_matrix_transients`, which invalidates and `MADV_DONTNEED`s every
  mapped checkpoint shard after every expert, consumes 21,012.196 ms;
- all 1,128 copied source-buffer installations consume 772.576 ms;
- all synchronous waits consume 815.605 ms, including only 403.657 ms of
  measured GPU-active interval;
- the GPU intervals are 0.9952% of routed-MoE wall; and
- the four named transaction categories cover 97.1152% of routed-MoE wall.

The expert intervals record 9,526,915,072 physical read bytes and 578,450
page-ins against 9,464,659,968 installed source bytes. Complete-process reads,
including the long CPU prefill and other weights, are 85,342,269,440 bytes.
Projection copies themselves show zero contemporaneous disk-read deltas because
Darwin accounts the asynchronous page I/O later in each enclosing expert
interval; the closed expert ledger is the physical-read authority.

Gate 8 passes throughout: minimum free memory is 77%, maximum peak RSS is
4,344,627,200 bytes, post-release physical footprint is 3,090,965,312 bytes,
swap growth and new throttled pages are zero, and every protected service
survives. The canonical analysis is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0105/token-001/analysis-002.json`,
SHA-256
`26d649f8babbf00a21bace7c522fab178992d092972ffc55ffb076ac033b1150`.

## Decision

Promote a cold routed-layer transaction experiment, not any current runtime
default. The primary first removal is the expert-scoped global checkpoint page
invalidation plus repeated tensor/scale validation and reacquisition. A
`bytesNoCopy` substitution alone addresses only 1.9% of measured routed wall
and cannot satisfy the 2x continuation gate. The next candidate must combine a
prevalidated page-aligned L1 runtime artifact or equivalent stable tensor
authority with bounded layer-scoped page residency/release; copied-buffer and
no-copy views then become controlled subvariants. After that causal boundary
passes, batch gate/up/down work and async arenas can target the remaining waits
and CPU staging.

Do not infer endpoint viability from the routed opportunity. Even deleting all
routed-MoE time from this diagnostic leaves 35.516 seconds in the current
unoptimized non-MoE path. That work requires its own complete-layer
transaction/computation experiments, and no timing here changes the rejected
L3 correctness status.
