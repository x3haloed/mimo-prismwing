# Experiment program

The program is ordered by information gained per dollar and engineering week.
Later stages are conditional on earlier kill gates.

This file is a prospective queue, not a results log. Executed work receives a
stable `PW-NNNN` record under `experiments/` and follows the promotion and
reversal rules in [WORKFLOW.md](WORKFLOW.md).

## E0 — Checkpoint census and lock

**Question:** What exactly must reside, stream, execute, and remain available for
each modality?

Produce a tensor census grouped into:

- Routed experts by layer and projection.
- Attention, routers, norms, embeddings, LM head, MTP, and dense layer zero.
- Vision/audio encoders and all projectors.
- Tokenizer/processor/custom code.
- KV bytes by context length and attention type.

Calculate exact packed sizes under candidate formats and distinguish stored
bytes from bytes the kernel reads. Establish `B_e` and `B_d` rather than
carrying estimates forward.

**Pass:** every source tensor is assigned exactly once and totals match source
indices and hashes.

## E1 — Official tiny and sampled forward fixtures

**Question:** Can the MiMo architecture be reproduced before optimization?

Create source-derived fixtures for text, image, audio, video, mixed input, MTP,
routers, selected experts, KV updates, layers, and sampled local logits. Begin
with seeded tiny synthetic tensors, then use sampled real tensors from the
pinned checkpoint. Every oracle must be traceable to Xiaomi's published model
and processor semantics; no whole-model official-framework capture is assumed
available.

**Pass:** declared numerical tolerances pass in a scalar reference and initial
accelerated kernels. Incremental decode matches whole-sequence evaluation.

**Kill condition:** unresolved semantic mismatch blocks all performance work.

## E2 — Routing, cache, and speculation laboratory

**Question:** Can routing statistics make physical expert reads disappear?

Collect roughly one million representative token positions with:

- Top-eight IDs and gate scores by layer.
- Token, request, session, domain, language, and modality.
- MTP/DFlash candidates, accepted prefix, and rejected branches.
- Cache and scheduling state.

Calculate:

- Frequency, entropy, Gini, hot-set mass, and reuse distance.
- LRU, LFU, 2Q, TinyLFU, and offline Belady byte-hit curves at 1–10 GiB.
- Session working sets and turn-to-turn overlap.
- Inter-token and inter-layer route predictability.
- `A`, `q`, `U`, `A/U`, and rejected-branch cache pollution.
- Affinity-scheduling gains under explicit latency deadlines.

**Go:** exact M1 caching remains a primary track only if the offline-optimal
6–8 GiB cache approaches the 93–98% hit range required by measured bandwidth,
or if scheduling/speculation changes the requirement materially.

**Kill:** if Belady is far below that range, prefetch becomes latency support,
not a throughput architecture.

## E3 — M1 expert execution and I/O microbenchmark

**Question:** What does one real 13.5 MiB expert cost end to end?

Using synthetic and sampled real weights, benchmark batch 1–512:

- Token-major and expert-major order.
- Fixed-stride `pread`, mmap, internal SSD, and candidate external NVMe layouts.
- Queue depth, copies, Metal buffer ownership, and cold/warm file cache.
- INT4 direct execution and lower-bit candidate formats.
- Bucketing, dequantization, dispatch, GEMV/GEMM, and output reduction.

Report logical weight GiB/s, actual device bytes, FLOP/s, energy, and latency.

**Go:** an implementation track must achieve the throughput assumed by its
system model, not merely a fast storage-only read.

## E4 — Exact canonicalization and compression

**Question:** Are experts more losslessly compressible after accounting for
function-preserving symmetry?

For sampled layers:

1. Form coupled gate-row/up-row/down-column neuron records.
2. Prove and apply only exact permutations/scalings valid for MiMo's expert
   equation and quantized metadata.
3. Align experts with approximate matching, then serialize each transformed
   expert exactly.
4. Measure residual entropy and end-to-end decompression throughput for whole
   experts and executable tiles.

**Go:** pursue an L1 codec if total executable-path improvement is material.

**Kill:** less than 10% transfer improvement or decompression that gives back
the gain demotes lossless coding to checkpoint-size housekeeping.

## E5 — Activation-weighted shared-basis audit

**Question:** Can the expert bank become a small executable program family?

For representative early, middle, late, hot, and rare experts:

- Test ranks 16, 32, 64, 128, and 256 after canonicalization.
- Test matrix-space, coupled-neuron, output-space, and nonlinear expert-family
  representations.
- Evaluate only on held-out routed activations as well as generic activations.
- Measure local expert-output error, layer-state drift, later route divergence,
  next-token KL, modality/task score, executable bytes, and compute.
- Determine whether common bases are evaluated once for the top-eight mixture
  or redundantly inside each expert.

**Go:** begin recovery training only if rank 32–64 or an equivalently compact
program gives a plausible quality path and a fused kernel reduces `f_M`, not
only checkpoint size.

**Kill:** rank 256 or higher for acceptable activation-weighted error makes the
stock-M1 moonshot nonviable.

PW-0115 adds the prerequisite physical envelope. A published-MoBE-shaped form
that leaves MiMo down projections unchanged has a 33.333% bank floor and cannot
meet PW-0045's 25% gate. A deeper all-three-projection family remains eligible
in optimistic accounting; freeze `(r=768,m=4)`, `(512,8)`, and `(128,32)` for
the activation-weighted audit. These are candidate shapes, not quality or
performance results.

PW-0116 supplies the first real routed-activation corpus at layers 4, 24, and
46, but its hosted English trace touches only 69--72 experts per layer and has
thin contiguous validation/holdout expert coverage. It is sufficient for a
pilot and matched controls, never for representation promotion. PW-0117 then
rejects the published nonlinear MoBE form for transaction compute while
retaining only the separately named identity-basis transaction algebra.
PW-0118 proves that the smallest `(128,32)` production parameter and Adam shape
fits safely on MPS; it does not prove that rank is a useful fidelity target.

PW-0119 supplies the stronger independent best-rank control. Rank 128 is
1.27--10.26x worse than rank 768 on the sampled routed expert outputs, while
even rank 768 leaves 56.9--78.3% relative L2 at layers 24 and 46 under ordinary
matrix-SVD fitting. This supersedes the assumption that the smallest
weight-MSE shape should be trained first and makes activation-weighted fitting
the cheapest remaining falsification. PW-0120 therefore preflights the
6,643,793,920-byte semantic parameter/gradient/Adam lower bound for `(768,4)`
under phase-level Gate 8 stops before any rank-heavy fit. It rejects direct
full-state MPS Adam: after parameters and dense backward, the first Adam step
hits the 7.10-GiB allocator ceiling while requesting another 1.50 GiB, and the
immediate release footprint also misses the 4-GiB gate. Do not disable the
watermark or retry this topology. Continue only through a separately frozen
block-coordinate, offloaded, factored-state, or external optimizer path that
preserves the rank-768 representation and held-out objective.

PW-0121 chooses the cheapest such path: optimize one projection of layer-24
hot expert 23 at a time, retain no inactive gradients or state, and compare the
complete three-projection fitted expert against PW-0119's untouched validation
and holdout rank-768 SVD control. It passes: validation relative L2 falls
64.54% and untouched holdout falls 44.81%, while every projection releases to
zero MPS current allocation. This authorizes replication on layer-46 hot
expert 28, not shared-bank construction; depth dependence and the thin English
corpus remain unresolved.

PW-0122 completes the authorized depth replication on layer-46 hot expert 28.
Validation relative L2 falls 65.81% and untouched holdout falls 47.21% against
the independent rank-768 SVD control, with the same bounded release behavior.
This authorizes only a multi-expert shared-basis pilot within one layer. That
pilot must compare against activation-weighted independent controls and retain
at least two experts with non-empty train, validation, and holdout coverage.

PW-0123 strengthens that minimum to the smallest topology that forces sharing:
five layer-46 experts against four bases. It rejects the mechanism. The four
experts initialized with private bases remain competitive, while expert 57—the
first forced to share—misses independent projection controls by 3.98--5.05x
and reaches `0.8114` holdout relative L2, worse than global SVD's `0.7454`.
Equal-expert complete averages pass and would conceal the tail failure.

Do not add bases or merely extend the same run. PW-0116 now limits the next
representation inference: expert 57 has only 17 train placements. Acquire a
broader frozen multilingual/modality activation corpus with substantive same-
layer expert coverage in every split, then repeat the forced-sharing control.
If the tail failure survives representative coverage, kill four-basis sharing
before any full bank, quantized artifact, or kernel.

## E6 — MTP and DFlash verification

**Question:** How much accepted work does speculation buy on the actual runtime?

Measure native MTP and DFlash across block sizes and workloads:

- Draft latency and memory.
- Acceptance distribution, not only mean.
- Target positions `q`, accepted tokens `A`, and unique expert union `U` by
  layer.
- Dense weight reuse, expert weight reuse, target arithmetic, and KV overhead.
- Cache-aware candidate counts and route-overlap shaping with mathematically
  correct verification/correction.

PW-0009 established a prerequisite: the official DFlash bundle's target
weights differ from the pinned base checkpoint. Test its draft against the base
target directly; do not import bundle acceptance claims or substitute its
target. Restrict the shipped verifier to greedy mode until a distribution-
preserving positive-temperature correction path passes statistical tests.

PW-0110 adds the measured cold-storage prerequisite for the unchanged
source-FP8 internal-SSD branch. `q=16` and `q=32` are rejected even with
perfect acceptance and minimum union. A base-aligned route-coherent pool must
support at least `q=137` for the formal 50-TPS branch or `q=94` for the
separately valuable 34.3-TPS horizon, and real `A/U` must meet 136.380 or
93.556 respectively. Do not begin proposer training below that physical bar.

PW-0112 supplies the real wide-route answer: on a frozen base teacher-forced
text suffix, the best `q=94` window reaches only `A/U=46.567`, and `q=137`
reaches `A/U=57.045`. Their otherwise-free ceilings are 17.072 and 20.914 TPS.
Reject proposer training and wide source-FP8 verification on this storage
premise even if the proposer is base-aligned and perfectly accepting. Reopen
only after executable-byte reduction or a different named physical store
changes PW-0110's acquisition floor.

The same trace retains—but does not promote—a bounded exact cache as a
secondary experiment. Four-GiB offline Belady reaches 44.716% hits and a
first-32-trained static frequency cache reaches 29.951% on the following 96
positions, while global LRU remains zero. Any physical follow-up must freeze a
cold end-to-end gain and explain why the remaining 5.232 GB/token of oracle
misses fits the combined branch; it is not a standalone 34.3/50 mechanism.

PW-0113 rejects the deeper exact neuron-canonicalization follow-up to PW-0109.
Expanded per-neuron scale association adds only 3.116%, but aligned residuals
save just 0.340% versus original source bytes and make the optimistic physical
bound 266.790 ms rather than 47.7 ms. Do not expand exact permutation work to
all experts or build a decoder. A learned/common basis, sign symmetry, or
approximate routed-mixture compiler is a distinct experiment and exactness
class.

PW-0114 conditionally retains the named repair-free Metal-native L3 numerical
premise. On one frozen complete incremental token it passes the predeclared
source-derived final-distribution probe with 0.024239-nat chosen-token error,
19/20 top-token overlap, and 0.000578 projected JSD while performing zero sparse
repairs. This does not promote the 75.834-second projection-at-a-time vehicle or
establish near-equivalence: routes differ from source at 20 layers and from the
repaired control at three late layers. Do not broaden the numerical walk until
an independently bounded representation changes PW-0108's executable-byte
premise.

**Go:** use speculation as expert-byte leverage only when `A/U > 1`; otherwise
retain it for dense-weight reuse or discard it if total wall time loses.

## E7 — Candidate companion hardware

**Question:** Can inexpensive local hardware execute complete MiMo layer stages
fast enough?

Before buying a fleet, benchmark one candidate node with real layer shapes:

- Sustained DRAM bandwidth and NUMA placement.
- Direct low-bit expert and attention kernels.
- Full stage throughput at verifier batch sizes.
- Hidden-state network transfer, serialization, and barrier latency.
- Single-stream latency and multi-stream pipeline throughput.

Compare:

1. Entire language backbone on one DRAM-rich server.
2. Contiguous layer stages across a few nodes.
3. Remote expert-only execution as a control.

**Go:** buy/assemble only if one stage demonstrates scaled throughput above the
target with at least 25% headroom and the complete BOM remains within target.

**Kill:** nominal bandwidth without fused-kernel throughput is a failure.

## E8 — Integrated milestones

Integrate only evidence-backed mechanisms, in this order:

1. Correct text runtime.
2. Native vision/audio/video encoders and mixed inputs.
3. Target-faithful expert streaming baseline.
4. Native MTP/DFlash.
5. Proven cache/layout optimizations.
6. Proven exact codec or hardware stages.
7. Modified expert representation only behind an explicit mode and quality
   gate.

At Prismwing 10, 25, and 50, freeze artifacts and rerun the complete validation
protocol. Optimization does not accumulate quality debt between milestones.

## E9 — Embodiment-jump portfolio

When measured component work shows that rearranging the current execution shape
does not close the gap, test alternative physical embodiments under the shared
contract in [EMBODIMENT_JUMPS.md](EMBODIMENT_JUMPS.md):

- [PW-0044](../experiments/PW-0044-route-coherent-phrase-lattice.md) turns
  speculative future time into route-coherent verification width.
- [PW-0045](../experiments/PW-0045-routed-mixture-compiler.md) compiles the
  weighted routed mixture rather than materializing independent experts.
- [PW-0046](../experiments/PW-0046-expert-bank-exception-store.md) treats exact
  expert weights as backing state for a conservative resident program.
- [PW-0047](../experiments/PW-0047-texture-native-weight-codec.md) tests GPU
  fixed-function texture decode as an executable weight representation.
- [PW-0048](../experiments/PW-0048-dram-backbone-appliance.md) maps the complete
  backbone onto one local DRAM-rich appliance rather than remote expert calls.

These are predeclared hypotheses, not evidence. Execute their cheap kill tests
in dependency order, preserve target-faithful and modified branches separately,
and give every combined mechanism a new experiment ID.

## E10 — Base-layer causal transition

[PW-0049](../experiments/PW-0049-real-base-decoder-layer.md) joins the promoted
attention and dynamic source-FP8 MoE work across one real learned base-model
decoder layer. It now passes as the target-faithful correctness baseline and
provides the first causally derived route trace for PW-0044 through PW-0046.

The layer-local fixture must derive routes and selected expert work from its
own learned attention output. PW-0039's frozen post-attention input, routes, and
expert union remain component controls and cannot be substituted for this
causal boundary.

The seeded trace selects 56 unique experts for eight positions (`U=7.0`). Its
bounded expanded-F32 Accelerate embodiment is deliberately not a performance
default; the result advances the correctness ladder and changes the search
priors, not endpoint TPS.

## E11 — Slow complete text endpoint

[PW-0050](../experiments/PW-0050-slow-complete-text-endpoint.md) is the next
causal boundary. It carries a one-token real UTF-8 prompt through the pinned
tokenizer, all 48 source decoder layers, final norm and LM head, then retains
K/V state for one incremental greedy token. This deliberately small raw-text
walking slice must become whole before chat-template expansion or any new
performance mechanism is promoted.

PW-0050 may stream and sequentially materialize selected source experts to fit
the 16 GiB host. A layer prefix, frozen-route replay, fixture-supplied hidden
state, or logits-only probe is not a text endpoint. Its first passing timing is
an end-to-end baseline, not an accepted performance default.

## E12 — Page-stable routed-layer transaction

[PW-0105](../experiments/PW-0105-weight-install-tomography.md) establishes that
expert-scoped validation and global checkpoint invalidation, not Metal
arithmetic, dominate the current cold routed path.
[PW-0106](../experiments/PW-0106-page-stable-metal-ready-routed-layer.md) then
passes its causal gate: a prevalidated page-aligned artifact is 2.601x faster
than copied/global-release control while still copying Metal buffers, and a
real no-copy binding reaches 6.381x. This promotes the physical representation
and lifecycle into the next experiment, not into the runtime default.
[PW-0107](../experiments/PW-0107-two-barrier-routed-layer-transaction.md)
subsequently rejects ordinary two-barrier command aggregation as the cold
solution: it reaches 1.694x warm but only 1.166x cold, with 96.001 ms still
inside two waits around 8.320 ms of GPU activity.
[PW-0108](../experiments/PW-0108-metal-io-layer-acquisition-bound.md) then
tests the promoted acquisition mechanism directly. Three concurrent real
Metal-I/O command buffers reach 58.034 ms cold for the exact 201.376 MB selected
payload, missing the 47.7 ms physical continuation bound. The internal-SSD
shared-event arena pipeline is therefore rejected before construction.
[PW-0109](../experiments/PW-0109-exact-expert-block-canonicalization.md) tests
whether exact 128-neuron symmetry can reduce that payload without changing
source scale topology. Aligned residuals compress to 95.087%, only 0.0433%
better than identity residuals and materially worse than compressing the
unmodified records, so that exact codec branch is rejected.
[PW-0111](../experiments/PW-0111-one-barrier-metal-native-routed-layer.md)
therefore tests the remaining deferred architecture directly: keep all eight
experts' activation staging, projections, SwiGLU, route weighting, reduction,
and scatter inside one named L3 Metal transaction, then wait and read back once.
Its source-derived oracle, existing L3 control, and external target thresholds
remain unchanged.
The completed experiment reproduces C2's exact routed and final-residual bytes
without sparse repair and improves the warm median 2.702x, but cold improves
only 1.198x because 100.584 ms remains in the one wait around 8.383 ms of GPU
work. The unchanged internal-SSD full-bank and token-walk branches therefore
remain rejected; retain the one-barrier mechanism only for a future resident
or wide-amortized transaction.

Do not build an internal-SSD Metal-I/O/compute arena over the unchanged
payload. Reopen this mechanism only for a named faster storage configuration
or after an exact executable representation reduces selected bytes enough to
change PW-0108's bound. Preserve PW-0106 C2, PW-0107 C3, and PW-0108's
three-buffer loader as distinct controls.

Any next exact byte-reduction experiment must change a premise PW-0109 did not
test—such as a representation that replaces the source scale layout or a
learned common basis—and must rederive decode cost against PW-0108 before
implementation. Generic storage compression and block permutation are not
active paths.

**Go:** for a changed physical premise, rederive the acquisition bound before
implementation, reproduce exact bytes and PW-0106's unchanged candidate
output, pass Gate 8, and achieve at least 2x cold complete-layer gain over
PW-0106 C2. Attribute physical reads, page-ins, GPU intervals, queue overlap,
waits, and arena residency.

**Kill:** a fused scheduler that merely moves the 95.9 ms wait or exceeds the
shared-host memory contract does not justify a full-bank artifact. Do not build
the approximately 303 GB bank or rerun a full token until this component gate
passes. PW-0114 has conditionally resolved and explicitly named the numerical
branch; it does not change the physical prerequisite.

## Black-swan budget

No more than 10% of research time before Prismwing 10 goes to black swans:

- GPU texture-format decompression of directly executable expert tiles.
- Progressive bitplanes with exact token-sampling certificates.
- Compute-in-memory or computational-storage devices.
- ANE-resident static expert neighborhoods.
- Certified activation/output memoization.

Each receives a one-week maximum kill test with a predefined success metric.
