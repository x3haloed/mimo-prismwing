# PW-0107 — Two-barrier routed-layer transaction

- Status: completed
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: `9c375dac723977a869905a3fb3c7865e804b1a9b`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0101 layer-4 oracle
  `9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d`;
  PW-0106 artifact
  `fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21`
- Hardware/runtime: Apple M1 shared 16 GiB host; internal SSD; PW-0106
  page-stable no-copy source-FP8 representation
- Related records: PW-0040, PW-0042, PW-0097 through PW-0101, PW-0105,
  PW-0106

## Shared construction contract

Capability and preservation: execute the identical authenticated layer-4
routed row, source tensors, route order and weights, dynamic activation
quantization, Metal projection kernel, BF16 boundaries, value-derived sparse
repair, SwiGLU, weighted reduction, and final residual as PW-0106 C2. Every
expert gate/up/SwiGLU/down diagnostic, repair count, routed byte, and final
residual byte must remain identical to PW-0106. The known rejected L3 arithmetic
result remains visible; this experiment changes scheduling only.

Causal path and risk frontier: PW-0106 C2 spends a cold median 123.053 ms for
one routed layer. Creating 24 no-copy source bindings costs only 0.425 ms and
GPU execution occupies 8.291 ms, but 24 synchronous waits total 95.784 ms while
201,719,808 artifact bytes are physically read. The least-proven boundary is
whether a layer-scoped command topology can make those page demands and GPU
dispatches progress coherently rather than serializing them behind 24 CPU
round trips.

Topology: retain one Rust execution authority, one existing Metal projection
kernel, one PW-0106 artifact mapping, and one correctness ladder. Add one
transaction executor rather than a parallel layer scheduler. It owns two
explicit phases: all eight experts' gate/up projections in one command buffer,
then all eight down projections in one command buffer. The existing C2 executor
remains the interleaved control. Shared input, LUT, and shape buffers replace
per-projection duplicates where their bytes are identical.

Embodiment depth: the candidate may collapse 24 command buffers, encoders,
commits, and waits into two; reuse the already validated no-copy weight/scale
regions; compute the identical dynamic-FP8 input once for all eight experts;
and retain all gate/up outputs until the first phase completes. CPU BF16
rounding, sparse repair, SwiGLU, dynamic hidden staging, final repair, weighted
scatter, and the one layer-boundary artifact release remain unchanged. It may
not introduce Metal I/O, argument buffers, indirect commands, persistent weight
caches, full-bank artifacts, new arithmetic kernels, or asynchronous arenas.

Project constraints: one real layer and one Metal process at a time. Keep Gate
8's phase-level memory-pressure, RSS, swap, throttling, release, and protected-
service checks. Large evidence remains external to Git. The control and
candidate use the same artifact mapping and cold/warm preparation; no endpoint
TPS or full-model inference claim is authorized.

## Candidate comparison

Three deeper shapes remain distinct:

1. **Selected C3:** two barriers preserve the existing CPU repair authority and
   cheaply test whether command topology is a material part of cold wait time.
2. **Deferred one-barrier GPU-native layer:** keeping dynamic FP8, SwiGLU,
   reduction, and scatter entirely on GPU is physically preferable, but cannot
   reproduce the current Accelerate sparse-repair topology without new
   numerical semantics. It requires a named L3 contract and distributional
   gates.
3. **Deferred Metal-I/O/arena pipeline:** direct I/O plus independent compute
   queues may overlap acquisition and execution, but should be implemented only
   after C3 reveals how much ordinary no-copy command aggregation buys.

The first candidate crosses the current risk frontier with the fewest new
authorities. Failure does not disprove either deeper mechanism; it identifies
the surviving physical floor they must remove.

## Verification and measurement contract

Before real timing, add deterministic tiny transaction fixtures proving:

- batched projection result slots cannot alias or reorder;
- one failed command fails the complete phase closed;
- shared input, LUT, and shape lifetimes extend past command completion;
- no-copy alignment and region lengths are checked for every binding;
- phase GPU timestamps remain subsets of their synchronous waits; and
- the eight-expert CPU stage and scatter order exactly match the serial C2
  authority.

On the real fixture, run at least three cold and three genuine warm trials for
PW-0106 C2 and C3 in alternating order, reversing the first variant each
repetition. Cold preparation invalidates only the artifact before each trial;
warm preparation explicitly prefaults once and does not invalidate between
trials. Both variants drop the mapping at the layer boundary; only cold trials
request `MS_INVALIDATE`/`MADV_DONTNEED`.

Record mapping/open, trusted tensor binding, source-buffer binding, shared and
output allocation, encoding, commit, waits, GPU intervals, CPU dynamic input,
gate/up repair, SwiGLU, dynamic hidden, down repair, scatter, final release,
physical reads, page-ins, signed fault observations, CPU time, RSS, memory
pressure, swap, throttling, and service health. Record phase and complete-layer
wall for every trial; only complete-layer wall decides performance.

## Gates

- **Correctness:** exact identity with PW-0106 for all routes, route weights,
  expert diagnostics, repair counts, routed bytes, and final residual bytes in
  every control and candidate trial.
- **Accounting:** C3 performs exactly 16 gate/up and eight down projection
  dispatches, two command buffers, two commits, and two waits. GPU intervals are
  available and do not exceed their containing waits. Installed logical bytes
  and physical-read ledgers close against the artifact.
- **Performance:** promote command aggregation only if C3 achieves at least
  2.0x median cold complete-layer speedup over interleaved C2, no cold candidate
  regresses, and warm complete-layer wall does not regress. A smaller repeatable
  gain may be retained as a diagnostic mechanism but cannot authorize broad
  artifact construction or a full walk.
- **Safety:** every Gate 8 boundary passes; no swap growth, new throttled pages,
  protected-service loss, failed mapping release, or unexplained resident
  retention.

If C3 fails 2x because cold physical reads remain serialized inside its two
waits, promote the bounded Metal-I/O/compute-overlap experiment instead of
adding command-topology variants. If C3 passes, the next experiment may combine
it with bounded arenas. In neither case build the approximately 303 GB full
expert bank or rerun a complete token until the selected layer mechanism clears
its own gate.

## Result

The release runtime at
`09d6efd64418c9eebdea1b6dbcb053adba03feab` executed the frozen protocol on
Apple M1. C3 issued exactly 16 gate/up and eight down dispatches through two
command buffers, two commits, and two waits in every candidate trial. The
candidate retained its source mappings, Metal resources, shared input, LUT,
shape, and output slots through command completion; every binding passed the
page-alignment and region-length checks. The 52-test Rust suite, including the
new fail-closed transaction primitives, and strict Clippy both pass.

All 12 interleaved trials reproduce one identical set of eight expert
diagnostics, repair counts `[6, 4, 3]`, routed bytes, and final-residual bytes.
The routed hash is
`6577967c5c847228ca900a03e39279c63359fbaf3102dc1472612b5301c84ace`;
the final-residual hash is
`112757cb90f05804fd887e7fc4c10563321ba49ed2e9eda792d32f4abfbdd8c3`.
This is exact identity with PW-0106's C2 output and does not promote its known
L3 arithmetic divergence.

Complete routed-layer measurements are:

| State and variant | Trial walls (ms) | Median (ms) | Candidate speedup |
| --- | --- | ---: | ---: |
| Cold C2, 24 barriers | 142.315, 106.669, 134.570 | 134.570 | control |
| Cold C3, 2 barriers | 115.447, 117.193, 114.812 | 115.447 | 1.166x |
| Warm C2, 24 barriers | 41.484, 40.358, 40.003 | 40.358 | control |
| Warm C3, 2 barriers | 23.711, 23.821, 24.060 | 23.821 | 1.694x |

The warm result proves that command aggregation removes material CPU/queue
overhead when source pages are resident. It does not clear the contracted cold
gate. One paired cold trial regresses (repetition 1), and the median cold gain
is only 1.166x rather than 2x. C3's cold median two-phase wait remains 96.001
ms while its GPU intervals total only 8.320 ms. Gate/up accounts for roughly
66.4--67.0 ms of wait and 133.5--134.1 MB of reads; down accounts for
29.3--33.0 ms and 58.4--67.0 MB. Collapsing 24 barriers to two therefore moves
but does not hide the physical page-acquisition floor.

All six warm rows record zero physical reads and page-ins. Cold invalidation
produces 193,314,816--201,719,808 candidate read bytes; the control's
134,479,872-byte low trial demonstrates that Darwin invalidation is not a
perfectly uniform physical-I/O generator, so the paired regression and
distribution are preserved rather than discarded. Gate 8 passes with 77%
minimum free memory, 568,229,888-byte peak RSS, 122,327,104-byte final physical
footprint, zero swap growth, zero new throttled pages, and stable protected
services.

The immutable raw report is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0107/trials-001/report.json`,
hash
`39d2a678212a7d98aee33396119928c0e9c2baa7aa4e9f5a19c63ce0fd005bd2`.
The clean analyzer at
`c3310a84f1e4f06a29cf708597c5e75e39554f92` emitted
`analysis-001.json`, hash
`bc2299248006b349eb2a6a9cee4c5b1a715968fbc9bf118a3d6c9aec702165e2`.
The updated throughput model hashes to
`4d04093ee3e0fa0f61d69da2d2ad787fddee98ca5ccb95e177c3bc33f888b29c`.

## Decision

Reject two-barrier command aggregation as the promoted cold architecture. It
is a useful warm diagnostic and remains available as a control, but it cannot
authorize a full-bank artifact or another token walk. The next causal boundary
is a bounded Metal-I/O/compute-overlap experiment: acquire the next expert or
tile into reusable arenas on an independent path while the GPU consumes the
current one, with explicit queue-overlap and arena-residency evidence. This
does not yet authorize a one-barrier Metal-native L3 arithmetic branch, a
route-cache default, or the approximately 303 GB full expert bank.
