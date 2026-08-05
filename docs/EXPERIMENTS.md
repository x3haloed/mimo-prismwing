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

## Black-swan budget

No more than 10% of research time before Prismwing 10 goes to black swans:

- GPU texture-format decompression of directly executable expert tiles.
- Progressive bitplanes with exact token-sampling certificates.
- Compute-in-memory or computational-storage devices.
- ANE-resident static expert neighborhoods.
- Certified activation/output memoization.

Each receives a one-week maximum kill test with a predefined success metric.
