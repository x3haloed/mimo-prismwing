# PW-0111 — One-barrier Metal-native routed layer

- Status: completed
- Disposition: rejected cold architecture; retained warm mechanism
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: `69f9cd51b21d069a6bc827b92c3f2324811d1906`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0101 layer-4 oracle
  `9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d`;
  PW-0106 artifact
  `fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21`
- Hardware/runtime: Apple M1 shared 16 GiB host; internal SSD; PW-0106
  page-stable no-copy source-FP8 representation
- Exactness: L1 storage and weights; named L3 Metal-native arithmetic candidate
- Related records: PW-0040, PW-0042, PW-0097 through PW-0101, PW-0105
  through PW-0110

## Governing premise

A routed layer, not a projection or individual expert, is the smallest
CPU-visible transaction. The current complete-token Metal path installs one
projection, dispatches, waits, reads back, performs CPU staging, and repeats
1,128 times per token. That topology was an effective correctness vehicle but
does not test Apple silicon's ability to let CPU and GPU share one allocation
and retain intermediate state across a useful body of work.

PW-0105 already supplies the requested weight-install tomography over all 376
experts and 1,128 projections. It attributes 16,790.296 ms to tensor
validation/page acquisition, 21,012.196 ms to repeated global invalidation,
772.576 ms to copied source installation, 815.605 ms to synchronous waits, and
only 403.657 ms to GPU-active intervals within 40,560.763 ms of routed-MoE
wall. PW-0106 then proves the L1 prerequisite: its page-stable artifact reaches
2.601x cold with copied Metal buffers and 6.381x with page-aligned no-copy
bindings. PW-0107 reduces 24 waits to two but retains CPU semantic staging and
reaches only 1.166x cold. PW-0108 separately rejects internal-SSD Metal-I/O
loading as fast enough to clear that experiment's frozen 2x gate. These
results do not test a one-wait GPU-resident layer.

Apple documents that its CPU and GPU access the same physical memory through
shared resources, recommends avoiding duplicated or shadowed resources, and
recommends batching enough encoders into each command buffer to avoid small-
submission bubbles. The same guidance requires bounding allocations by the
device's reported working-set recommendation. Sources:
[Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/),
[`makeBuffer(bytesNoCopy:length:options:deallocator:)`](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29),
and [Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading).

## Shared construction contract

Capability and preservation: execute the authenticated PW-0101 layer-4 routed
row using the identical input, route IDs and weights, source-FP8 weights and
scales, projection shapes, dynamic E4M3FN group-128 activation quantization,
BF16 boundaries, SwiGLU equation, expert order, weighted reduction, scatter,
and final residual shape. The PW-0106 artifact remains an L1 transform whose
records must verify against the checkpoint before execution. The source-
derived layer oracle and current sparse-repaired executor remain unchanged
controls.

Causal path and risk frontier: CPU routing produces the ordered eight-expert
schedule once. One layer-scoped Metal command buffer then consumes page-aligned
no-copy weight and scale views, performs input dynamic quantization, all gate
and up projections, BF16 staging, SwiGLU, hidden dynamic quantization, all down
projections, route weighting, deterministic reduction/scatter, and the layer-
final BF16 boundary. The CPU commits once, waits once, and reads back only the
final routed residual and bounded diagnostics. Intermediate activations never
become CPU scheduling inputs.

Topology: extend the existing Rust/Metal authority with one explicit
`MetalNativeRoutedLayer` executor and production kernels. Do not create a
generic graph runtime, second router, background service, compatibility layer,
or alternate artifact schema. Bind resources directly for the first causal
test. Argument buffers or indirect command buffers require a later experiment
showing that measured CPU encoding, rather than storage or GPU execution,
limits this transaction.

Embodiment depth: retain only the existing approximately 202 MB authenticated
layer artifact, one input, per-expert gate/up/hidden/down scratch regions, one
final residual, and bounded diagnostic slots. Reuse scratch storage where
command order proves lifetimes do not overlap. Total resident candidate
capacity must be declared before commit and remain below both 1 GiB and the
device's `recommendedMaxWorkingSetSize`. Do not build the approximately 303 GB
full expert bank, add a route cache, use Metal I/O, train a proposer, change
weights/routes/top-k, or materialize decoded matrices.

## Named numerical branch

The existing source-directed L3 control retains value-derived Accelerate
sparse repair and its 24-wait CPU staging topology. The candidate deliberately omits that
CPU-only repair and uses deterministic Metal-native reduction and BF16 staging.
It is therefore **L3 — bounded approximation**, even though source weights,
routes, layout, and equations are otherwise preserved. It may not be called
source-exact, L1, or target-faithful merely because a greedy token agrees.

No acceptance threshold changes. Record source-derived gate/up/SwiGLU/down,
routed-residual, final-residual, route, and sparse-repair-control comparisons.
The candidate must also report the applicable unchanged external thresholds
from `TARGET.md`: projected top-20 JSD, chosen-token logprob error, top-1 and
stable-top-1 agreement, reference-token regret, and greedy agreement. A
layer-local pass authorizes only the next correctness-ladder step; it cannot
prove hosted or endpoint parity.

## Correctness ladder

Before the real layer, add deterministic tiny fixtures and an independent CPU
reference for:

1. dynamic E4M3FN group-128 quantization including zero, signed zero,
   half-way, saturation, subnormal, and non-finite rejection cases;
2. BF16 conversion at both sides of every half-way boundary used by the
   production kernels;
3. SwiGLU with independently rounded gate, up, sigmoid/product, and hidden
   boundaries;
4. eight-expert route weighting and deterministic reduction/scatter, including
   zero and cancelling weights;
5. command/resource lifetime, scratch non-aliasing, ordered dispatch, one
   failed-kernel fail-closed propagation, and one-command/one-wait accounting;
   and
6. a tiny end-to-end routed-layer fixture comparing all bounded diagnostics and
   the final residual to the slow reference.

Then run the real PW-0101 layer fixture. Preserve every mismatch rather than
repairing or thresholding it after observation. Only a separately frozen
whole-prefix experiment may test accumulated divergence and local logits; only
the frozen hosted protocol may test external distributional and capability
gates.

## Measurement protocol

Compare interleaved controls and candidates in alternating order, reversing the
first variant each repetition:

- **C2 control:** PW-0106 page-stable no-copy representation with three waits
  per expert and existing CPU sparse repair/staging;
- **C4 candidate:** identical artifact and route with one Metal command buffer,
  one commit, one wait, GPU-resident intermediate stages, and one final
  readback.

Run at least three genuine cold and three genuine warm trials per variant.
Cold preparation invalidates only the layer artifact before each trial; warm
preparation prefaults once outside timing and performs no invalidation between
trials. Record mapping/open, binding, allocation, encoding, commit, wait, GPU
interval, readback, release, complete layer wall, bytes mapped and physically
read, page-ins/fault observations, CPU time, peak RSS, physical footprint,
memory pressure, swap, throttling, and protected-service health. For C4 also
record every kernel identity, dispatch count, scratch high-water mark, and a
closed GPU-stage timing ledger when counters are available. Complete layer
wall is the performance authority.

## Gates

- **Causal/accounting:** C4 performs exactly one command buffer, one commit,
  one synchronous wait, and one final routed-residual readback. No intermediate
  activation is read by the CPU or used to choose a later dispatch. Every
  authenticated source record is bound exactly once or through a manifest-
  proven shared view, and all allocations release at the layer boundary.
- **Fixture correctness:** every tiny primitive and end-to-end fixture passes
  its frozen exact/tolerance gate. Unknown shapes, layouts, revisions,
  alignments, resource limits, command failures, and non-finite values fail
  closed.
- **Real-layer numerical evidence:** routes and weights remain identical; all
  source-derived metrics and the existing unchanged parity gates are emitted.
  Continue to a whole-prefix L3 experiment only if C4 does not regress routed-
  and final-residual relative L2 or maximum absolute error versus C2 and does
  not introduce a new route change at the
  next authenticated boundary. This is a continuation gate, not a claim of
  source identity.
- **Performance:** C4 must achieve at least 2.0x median cold complete-layer
  speedup over C2, no cold candidate trial may regress, and warm wall may not
  regress. A smaller gain remains diagnostic and does not authorize the full
  artifact bank or another complete-token walk. Preserve PW-0108's 58.034 ms
  cold acquisition result as an independent physical comparison, not as time
  silently subtracted from C4.
- **Safety:** enforce normative Gate 8 before artifact mapping, after mapping
  and scratch allocation, immediately before commit, immediately after command
  completion, after final readback, after buffer/mapping release, and at final
  service-health readback. Stop and preserve failed evidence if free memory is
  below 20%, process physical footprint or peak RSS exceeds 8 GiB, released
  footprint remains above 4 GiB, swap grows by more than 512 MiB, any new
  throttled page appears, or a protected start-resident service disappears.
  No full walk is authorized by a layer-local safety pass.

If fixture or safety gates fail, reject the implementation. If numerical
continuation fails, preserve C4 as a performance diagnostic but do not advance
it up the correctness ladder. If the 2x cold gate fails, do not build the full
bank or hide the result behind a warm claim; use the measured ledger to decide
whether exact executable-byte reduction, faster named storage under the $500
cap, or a wider multi-position transaction is the next cheapest falsification.
If all gates pass, freeze the whole-prefix L3 experiment before executing it.

## Result

The runtime at `22c5e1a161a3eae80e6320ed63294855454caeec` added three
production Metal kernels and a single-command layer executor. A subsequent
measurement repair at `1334876bce437816dd4502b0486c39852f207226`
separated the Gate 8 observation harness from the performance interval while
retaining every required safety snapshot. The 55-test Rust/device suite,
including the real M1 one-command primitive fixture, and strict Clippy pass.
The standalone `xcrun metal` compiler was unavailable during this run, but
Metal's runtime compiler accepted and executed every kernel on the Apple M1.

C4 issues exactly one command buffer, six ordered encoders, one commit, one
wait, 24 source-FP8 projection dispatches, four staging/reduction dispatches,
and one final residual readback in every trial. No CPU-visible intermediate
selects a later dispatch. Its scratch high-water mark is 427,088 bytes and its
complete Metal resource ledger is 202,146,896 bytes, below both 1 GiB and the
device's reported recommended working-set size. All error flags remain zero.

The canonical interleaved measurements are:

| State and variant | Trial walls (ms) | Median (ms) | C4 speedup |
| --- | --- | ---: | ---: |
| Cold C2, 24 waits | 134.667, 105.877, 131.506 | 131.506 | control |
| Cold C4, one wait | 109.990, 81.451, 109.801 | 109.801 | 1.198x |
| Warm C2, 24 waits | 41.081, 39.942, 41.545 | 41.081 | control |
| Warm C4, one wait | 15.206, 14.428, 15.452 | 15.206 | 2.702x |

Every paired cold C4 trial improves on its C2 control, but the median misses
the frozen 2x continuation gate by a wide margin. The cold C4 wait remains
100.584 ms at median around only 8.383 ms of GPU activity. Warm C4 proves the
intended command and staging topology: once pages are resident, the entire
routed transaction is roughly 2.7x faster than projection-at-a-time C2.

C4 deliberately performs zero sparse repairs versus C2's `[6, 4, 3]`, yet all
six candidate trials produce C2's exact routed hash
`6577967c5c847228ca900a03e39279c63359fbaf3102dc1472612b5301c84ace`
and final-residual hash
`112757cb90f05804fd887e7fc4c10563321ba49ed2e9eda792d32f4abfbdd8c3`.
This is strong layer-local evidence that the Metal-native numerical branch does
not regress the current L3 control on this row. It does not repair the shared
source-derived failure: routed relative L2 remains `0.00172562` with maximum
absolute error `1.0`, and final-residual relative L2 remains `0.00163510` with
maximum absolute error `1.0`. No whole-prefix or hosted parity claim follows.

Gate 8 passes at every required phase with 77% minimum free memory,
538,050,560-byte peak RSS, 68,915,200-byte final physical footprint, zero swap
growth, zero new throttled pages, and stable protected services. The harness
cost 219--220 ms per candidate trial and is recorded separately from the
monotonic layer interval; the first two reports are preserved as noncanonical
because they charged that harness asymmetrically or used the pre-repair binary.

The canonical raw report is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0111/trials-003/report.json`,
hash
`47f764370172dff489629bb171d9dad7345f39e37f21622244a63b6f4edfcb14`.
The clean analyzer at
`9013194b632424fbe296fb5edfa8adc7f9f28b98` emitted `analysis-001.json`,
hash
`1940aa4554eedc586ff567c041f62f91d757d5afc88244ef895e71ca22488fc0`.
The updated throughput model hashes to
`cae9784cf9005596c54ab55ba48e51b138bead17b0767cec405bc6998e51e918`.

## Decision

Reject the one-barrier transaction as the promoted cold architecture on the
unchanged internal-SSD source-FP8 representation. Do not construct the full
303 GB expert bank, run another complete token, or advance C4 to a whole-prefix
L3 experiment under this premise. The experiment confirms rather than removes
the cold acquisition floor already bounded by PW-0108.

Retain C4 as the preferred warm/resident routed-layer mechanism and as a
necessary building block for any future verifier that amortizes one acquired
expert union across many positions. Reopen it only after a changed premise—an
exact executable-byte reduction, a named faster storage configuration under
the $500 cap, or a sufficiently wide measured route-coherent transaction—can
clear the cold bound. Argument buffers and indirect command buffers remain
deprioritized because measured encoding is not the limiting category.
