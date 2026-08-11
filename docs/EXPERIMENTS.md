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

PW-0002 now passes for the pinned revision. A complete local SHA-256 receipt
binds all 39 locked files with no omissions, and an independent local Rust
header census assigns all 73,530 tensors. Its category counts and
`315,683,674,448` tensor bytes exactly reproduce the earlier remote-header
census. Preserve these receipts as L0 authorities and fail closed on revision,
layout, count, or byte-total drift; no throughput constant changes.

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

PW-0124 executes the cheaper coverage falsification first. It keeps positions
`168..223` untouched and occurrence-stratifies only the already-unsealed
development prefix, raising expert 57 from 17 to 58 train placements. Expert
57's complete holdout result improves enough to pass the global-SVD gate, but
its shared gate/up/down projection errors remain `4.329--6.817x` their matched
independent controls; all three aggregate projection gates fail too. This
rejects development-split scarcity as the explanation for PW-0123 and makes
broader corpus acquisition solely to rescue this exact four-basis form
unattractive.

PW-0125 moves to PW-0115's structurally distinct balanced branch without
paying for a shared fit. Rank-512 activation fitting improves validation and
untouched holdout by 62.16% and 46.31% over matched SVD, but validation remains
`1.30184x` PW-0122's fitted rank-768 result and misses the frozen `1.25x`
capacity gate; holdout passes at `1.22401x`. Reject `(r=512,m=8)` before a
nine-expert/eight-basis optimizer or runtime construction. Preserve the strong
independent-fit result separately from the failed branch gate.

PW-0126 begins the genuinely direct PW-0045 branch by fitting layer-level
output dictionaries to routed residuals rather than expert matrices. Even an
oracle with true best coefficients fails validation at maximum rank 111:
5.24%, 27.01%, and 38.46% relative L2 at layers 4, 24, and 46 versus the 1%
gate. Training reconstruction is numerical-roundoff exact, so the failure is
generalization rather than implementation or optimization. Holdout remains
sealed. Reject the fixed linear dictionary before coefficient-network or
runtime work; nonlinear/input-conditioned direct compilation remains distinct.

PW-0127 returns to PW-0048 with the reduced $500 cap. A historical used R720
with two E5-2680-v2 CPUs and 512 GiB shows that source-backbone capacity can
fit the budget, but authenticated mandatory matrices need 29.641 billion
operations/token. Even granting every core 3.60-GHz turbo and perfect AVX
issue, the CPU-only ceiling is 38.865 TPS. Reject this class for Prismwing 50.
The 34.3 horizon remains measurement-only at 88.25% of impossible peak, far
too tight for a purchase inference before omitted FP8, attention, KV, NUMA,
and network work.

PW-0128 tests the remaining conventional legacy-accelerator variant without
confusing a favorable decode roofline with the full target. It binds PW-0112's
exact wide route unions and PW-0127's authenticated matrix count to one or two
M40 24-GB cards and one P40 24-GB card. The continuation gate is the 15-second
8K text TTFT requirement: grant all CPUs and GPUs peak FP32 simultaneously and
reject any named configuration whose mandatory matrices for only 8,000
uncached positions already exceed it. Record PCIe and layer-arena decode
ceilings independently; neither theoretical bandwidth nor a card-only market
price is a performance or complete-BOM claim.

The executed ceiling preserves the memo's routed-layer premise but rejects the
named hardware: a three-arena layer pipeline needs at most 2.341 GB, and one
ideal PCIe link would support 68.012--94.953 expert-only accepted TPS across
the frozen wide windows. Full-target prefill is decisive instead. Even with
CPU and GPU advertised peaks added perfectly, one M40 needs 29.0885 seconds,
two M40s 15.6500 seconds, and one P40 18.0299 seconds for only 8,000 mandatory
matrix positions. All miss the 15-second 8K TTFT gate before every omitted
cost. Reject these direct-FP32 configurations; retain the bounded layer
transaction for a faster substrate or changed executable representation.

PW-0151 reopens E7 against the zero-acquisition-cost H11SSL-i/EPYC 7351P host
and photographed 750-W NEX750B. Its impossible CPU-only ceiling is 25.046 TPS;
one P40, one P100, and one V100 each fail the 15-second 8K matrix-only prefill
gate. Two P100s alone survive compute at 12.260 seconds. On the real `q=137`
route and a granted four-by-2.5-GB/s store, however, the ideal serial block is
still 2.483 seconds: 34.3 TPS requires `A>=86/137`, and 50 TPS requires
`A>=125/137`. Conditionally retain this as a narrow pre-purchase envelope, not
a BOM or performance result; cable, airflow, clearance, storage, power, and
base-aligned proposer evidence remain required.

PW-0152 rejects the supplied width-eight and published width-16 DFlash shapes
before training. They require 18 and nine distinct target transactions to span
137 positions because every next block depends on the preceding target's clean
bonus anchor. They therefore cannot meet PW-0151's `A=86/137` or `A=125/137`
single-transaction requirement by ordinary chaining. A 137-node tree for
Prismwing 50 needs depth at least 125 and leaves only 12 off-path nodes. Retain
only a separately named `q>=137` or depth-at-least-125 base-aligned proposer;
its own cheap calibration and complete physical ledger remain prerequisites.

PW-0153 tests a changed physical premise before proposer training. The complete
source tensor payload is 294.003 GiB: five 64-GiB modules are the byte minimum,
although that population is not explicitly enumerated by the H11SSL manual;
six modules are enumerated but unbalanced, and eight are balanced. At an
impossible dual-PCIe-3-x16 nameplate ceiling, resident `q=137` acquisition plus
PW-0151 compute lowers the 34.3/50-TPS prerequisite from `A=86/125` to
`A=32/47`. The physical architecture survives, but the dated procurement
branch is rejected: five captured compatible modules cost `$1,235.95`, already
`$735.95` over the entire cap before accelerators or installation. This is not
a permanent market-wide price bound and authorizes no purchase.

PW-0154 tests a smaller exact residency embodiment that avoids PW-0153's DRAM
cost. After reserving every non-routed source tensor, three layer arenas, and
exact 8K BF16 KV, two P100s have aggregate room for 660 complete experts. A
static set trained only on PW-0112's 87-position prompt avoids 53.045% of the
following `q=137` union bytes and leaves 10.673 GB. One 3.5-GB/s lane is
structurally rejected for Prismwing 50 at only 42.033 perfect-acceptance TPS;
four such lanes lower the 34.3/50-TPS requirement to `A=34/49`. Conditionally
retain two-to-four lanes plus the exact cache, but not as a runtime or purchase:
aggregate-HBM sharding, sustained I/O, prefill, 1M KV, CUDA, and the complete
`$500` BOM remain unproven.

PW-0155 closes the logical install topology but rejects the captured shopping
list as purchase authority. H11SSL-i slots 2/4/6 are x16 and support
`x4x4x4x4`, so two double-width P100s plus one passive four-drive carrier are
logically placeable. The named subtotal is `$403.38`, but tax, destination
shipping, original PSU cables, exact SSD identity, dongle construction,
cooler fit, and sustained reads are unresolved. More seriously, two 250-W
cards plus the 170-W CPU leave only 62 W under the PSU's combined 732-W +12-V
label, while one P100 auxiliary input is specified up to the full 20-A label
of one rail. Preserve the topology as conditional, authorize no purchase, and
measure 8K route coverage before asking for physical validation.

PW-0156 attempts the next cheap falsification: count monotonic distinct
`(layer, expert)` records over an exact 8K causal prefill and reject the
four-lane source-FP8 branch at 9,003 records. It is inconclusive. The primary,
alternate, and all five pre-frozen panel corpora encounter exact eighth/ninth
router-logit ties before the contracted 512-position prefix; the panel stops
are 509, 309, 20, 146, and 423. Because PyTorch does not promise stable tied
`topk` indices, the C++ selection cannot authorize downstream states. No
manifest, coverage result, Gate-8 pass, or endpoint metric was published.
Do not search further corpora. Retain the hardware branch conditionally and
renew this walk only with source-framework route-index authority or a proven
exact equivalent.

PW-0157 closes that exact-equivalent question. The pinned PyTorch 2.13.0 CPU
fixture and Prismwing's libc++ bridge match every tied unsorted index exactly;
the original 512-position control and one-shot K/V-release runtime also match
every route-semantic field. Exact walks through 512, 1,024, 2,048, 4,096, and
8,000 positions touch 2,980, 3,572, 4,456, 4,585, and 4,903 distinct
`(layer, expert)` records. The final count is 4,100 below the 9,003-record
four-lane rejection boundary. Retain only that optimistic 8K storage-capacity
condition; it neither measures the storage path nor repairs PW-0158's complete
two-P100 million-context rejection.

PW-0158 closes a separate full-capability prerequisite before the surviving
two-P100 component envelope can motivate purchase. At exactly one million
positions, ordinary QK and weighted-V arithmetic across the pinned nine global
and 39 sliding layers requires `184,524,643,656,007,680` FLOPs. Granting two
P100s their combined advertised 37.4-TFLOPS FP16 peak continuously still gives
an `82.2302`-minute attention-only floor, versus TARGET's complete 30-minute
prefill limit. Exact BF16 KV is also 23.066 GB; with free streaming of all
non-routed tensors, only 261 complete expert slots remain after the three
arenas. Reject the complete ordinary-dense-attention two-P100 embodiment while
preserving PW-0151/PW-0154 as 8K component evidence. A changed-attention L3/L4
branch or different complete hardware candidate remains logically open and
must satisfy every long-context fidelity and capability gate.

PW-0159 tests the narrow used-Ampere alternative and corrects a precision-rate
conflation before authoritative execution. Dense BF16 with FP32 accumulation
on the 12-GB RTX 3080 is `61.2864` TFLOPS, not 123; mandatory 1M matrices plus
attention therefore need `58.2418` minutes and reject the source-oriented
control. Dense FP16 with FP16 accumulation is an L3 `122.5728`-TFLOPS
diagnostic and narrowly survives at `29.1209` minutes, with no fidelity proof.
The first 4,096 exact route positions plus a perfect 375-expert preload still
force at least three ideal 3.5-GB/s storage lanes for 8K TTFT. The captured
active card/three-drive/adapter subtotal is `$575.00` before tax, rejecting the
dated BOM. Retain only a price-triggered L3 hypothesis below a `$371.72`
delivered-card ceiling before tax; authorize no purchase.

PW-0160 tests whether the only permitted hosted whole-model reference can
actually provide a scoreable true-million-token answer key before
changed-attention validation begins. The local path deterministically renders
and round-trips exactly 1,000,000 pinned tokenizer IDs, but three bounded
Parasail calls return only transient upstream failures: JSON 502, then two
explicit shared-provider-pool 429s after repair and cooldown. This neither
passes nor kills hosted million-token capability. Preserve every error, stop
under the attempt budget, leave the reference path unproven, and require a new
experiment plus stable Parasail availability or explicit new-epoch authority
before retrying.

PW-0161 closes the actual single-card 32-GB Volta alternatives suggested by
the owned host's cleaner 312-W aggregate +12-V nameplate margin. The standard
V100 fails permanently for ordinary dense 1M context: even its favorable
112-TFLOPS L3 Tensor peak plus the EPYC's impossible peak needs `1,899.6029`
seconds, already `99.6029` seconds beyond the complete gate. V100S's
130-TFLOPS ceiling survives at `1,638.0745` seconds but remains numerically
unqualified and cannot hold exact KV, arenas, and all common source weights in
32 decimal GB. Captured active card-only prices are `$679.00` and `$1,054.99`
before tax, independently exceeding the complete cap before cable, cooling,
or storage. Reject both captured purchases and retain only a price-triggered
V100S L3 hypothesis; authorize no hardware or CUDA implementation.

PW-0162 tests the most favorable numerical version of the cheap two-P100
changed-attention premise: retain the source attention probabilities' largest
20% of visible rows, renormalize them in F32, and compare the resulting value
output on all 64 heads at 15 positions in each of the nine global layers. Its
first full observer walk exposed a correctness-fixture defect before producing
a pruning result. `layer_routes_sha256` covered complete route-trace records,
including nondeterministic per-layer `wall_ms`, so a cross-run equality check
failed even when every route was identical. A same-shape observer-disabled
control proves all 24,064 ordered expert rows and route-weight rows bit-exact;
its authenticated manifest hashes to
`480b02816b293ed8a2275e3c2810ee940fa0916db31fd1d730d6331e9f00a025`.
Repair route identity to cover only layer, ordered expert IDs, and ordered
weights; both authorities then hash to
`c0e5c8fd8c72f148895d39fdf38b95e84e93228206563ea49b242f48b0c69872`.
Make no pruning decision from the invalid first observer attempt; rerun the
unchanged numerical oracle under the corrected fail-closed guard. That
corrected full run then failed the semantic guard genuinely: doing allocation,
sorting, renormalization, and candidate reductions between source head
computations changed at least one route. Treat this as a second instrument
defect, not a pruning result. The repaired instrument only copies sampled
Q/K/V and source outputs into preallocated bounded storage during the source
walk, verifies exact routes, and performs all replay and oracle arithmetic
offline. Require bit-exact replay of the captured source outputs before
adjudicating the candidate. The resulting passive-capture run still changed
the semantic route hash after `1,679` seconds. Its failure manifest hashes to
`026f116129543b02285d81e20bb0a3a7746c91623f4403a0aa10f262b9d87189`.
This is still an instrument failure, not a pruning result. Run a same-commit
no-capture control to distinguish capture effects from changed-binary or
Accelerate drift, and require any successor failure to report the actual hash,
expert mismatches, route-weight mismatches, maximum absolute error, and ULP
error rather than failing opaquely. Replace heap-backed capture storage with a
single fixed-offset anonymous mmap and require a same-commit 64-position
no-capture/capture smoke pass before another full walk.
The unchanged-binary no-capture control then completed in `1,677.943614`
seconds and reproduced all `24,064` semantic route rows bit-exactly. Its
manifest hashes to
`9e95643ae0cba8ee9eda2f0447f477d05e839a02e13ff457e80499cbba86bcce`.
This attributes the remaining drift to the capture path rather than the
rebuilt source runtime and keeps the mmap smoke repair justified.

That fixed-offset anonymous-mmap capture passes the 64-position same-shape
smoke: all 3,008 route rows remain exact and the 100% offline replay is
bit-exact across 73,728 captured values. The authority and capture manifests
hash to
`d6b1483b0d6161611f58b2746edcd5f356f503c337bf590fcee2989f3d436f66`
and
`07adc240519642719c49c822aa25a1e7b38581d7ac629ddde5e0a5690e8013aa`.
This authorizes the final 512-position oracle walk; it does not itself decide
the pruning mechanism or establish endpoint TPS.

The production preflight additionally distinguishes the analyzer's parsed-JSON
semantic hash (`c0e5c8fd...c69872`) from the runtime's typed-F32 semantic hash
(`9cf63371...b7a0dc`). Both bind the same authenticated route values under
different numeric canonicalizations. Pin both explicitly and reject any raw
report that does not carry the runtime value.

The authenticated final walk rejects the mechanism. At 20% retained history,
aggregate relative L2 is `0.172375`, the worst layer is `3.025940`, and
head-query p99 is `4.888554`, missing the frozen `0.010000`, `0.020000`, and
`0.050000` limits by wide margins. The exact `21.056139%` two-P100 arithmetic
boundary remains at `0.167131` aggregate error. Raw and analysis evidence hash
to `15c0cb8ab6e5058e6413efeb2a60effd200a8c5e9bc915f708fe030c4f6f4cbe`
and `afc32798c5a474286e3eea65ccd6d32ab05f04921df1bacf5622585cad09d422`.
Kill simple
probability-ranked global-history pruning at this arithmetic fraction; retain
learned/recurrent attention and changed-weight mechanisms as distinct
branches. No throughput-model constant or endpoint TPS changes.

PW-0163 closes the strongest conventional 32-GB AMD PCIe counterexample found
in the current search. MI100's 92.3-TFLOPS BF16 Matrix peak plus the EPYC's
impossible concurrent peak still needs `2,301.8085` seconds for mandatory 1M
matrices plus ordinary attention, permanently rejecting the source-oriented
control before all omitted work. Its 184.6-TFLOPS FP16 peak is an L3-only
arithmetic survivor at `1,155.5143` seconds. Exact KV, arenas, and common
weights exceed 32 decimal GB by `6,221,107,536` bytes. The captured active used
card is `$999.00` before tax, already `$499.00` over the complete cap, and the
owned Debian 13 installation is not a supported ROCm 7.1 MI100 OS. Reject the
captured procurement and source-oriented runtime; retain only a future
price-triggered FP16 L3 hypothesis and authorize no HIP implementation.

PW-0164 closes the strongest NVIDIA Blackwell tier officially launched below
the complete hardware cap rather than treating older Ampere/Volta/CDNA results
as representative. RTX 5060 Ti's advertised 759 AI TOPS is not dense BF16.
Scaling the official same-generation RTX 5070 dense rates by its 36 SMs and
2.57-GHz boost gives `47.3435`-TFLOPS BF16/FP32-accumulate and
`94.7636`-TFLOPS FP16/FP16-accumulate ceilings. With the EPYC's impossible
peak granted concurrently, complete 1M matrices plus ordinary attention need
`4,453.8213` and `2,242.4320` seconds respectively. Both exceed the 1,800-
second gate before every omitted cost, permanently rejecting ordinary-dense
RTX 5060 Ti regardless of future price. Exact 1M BF16 KV alone also exceeds
16 decimal GB by `7,065,559,040` bytes. The 180-W card has favorable PSU
nameplate margin; official `$429` MSRP and a dated `$479.99` out-of-stock row
do not establish a delivered BOM. Preserve RTX 5070+, changed attention, and
modified low-bit modes as separate branches; authorize no purchase or CUDA
implementation.

PW-0165 closes the current in-stock sub-`$500` AMD consumer counterexample.
RX 9060 XT's full 103-TFLOPS dense half-precision Matrix rate plus the EPYC's
impossible concurrent peak still needs `2,064.3998` seconds for complete 1M
matrices plus ordinary attention, missing the entire gate by `264.3998`
seconds before all omitted work. This rejects both source-oriented BF16/F32
accumulation and the favorable dense FP16 L3 diagnostic. AMD's 205-TFLOPS
structured-sparse row would pass at `1,040.9414` seconds, but the RDNA4 ISA
binds it to two zeros per four and unchanged Prismwing weights do not admit
that premise. Exact 1M BF16 KV alone exceeds 16 decimal GB by
`7,065,559,040` bytes. The captured new in-stock card is `$449.99` before
unknown tax and has favorable 160-W/one-8-pin PSU nameplates, but no complete
BOM follows. Reject ordinary-dense RX 9060 XT permanently; preserve explicit
2:4 weight modification, FP8, changed attention, and faster cards separately.

PW-0166 closes the affordable Intel Xe2 counterexample without mislabeling
Arc's official 233-INT8-TOPS row as BF16. Pinned Intel IGC DPAS semantics assign
two operations per BF16 channel versus four per INT8 channel, while the Xe2
scheduler models equal same-size DPAS latency and occupancy independent of
precision. The resulting source-oriented B580 ceiling is 116.5-TFLOPS
BF16/F32-accumulate. Even with the EPYC's impossible peak concurrently,
mandatory 1M matrices plus ordinary attention need `1,826.6923` seconds,
missing the complete gate by `26.6923` seconds before all omitted work.
Exact BF16 KV alone exceeds 12 decimal GB by `11,065,559,040` bytes. The
190-W board has favorable PSU nameplate margin and an official `$249` launch
price, but neither proves installation or a delivered BOM. Permanently reject
ordinary-dense B580; preserve changed attention, modified weights, faster Xe2
products, and multi-card execution, and authorize no purchase or oneAPI work.

PW-0167 tests the older 16-GB Arc A770 rather than assuming Xe2's generation
name makes B580 the stronger source-oriented candidate. Intel publishes 262
dense INT8 XMX TOPS for A770 and a 4,096:2,048 INT8-to-BF16 operation ratio for
Xe-HPG, deriving a favorable 131-TFLOPS BF16 ceiling. With the EPYC's impossible
peak concurrently, mandatory 1M matrices plus ordinary attention need
`1,625.6406` seconds, leaving `174.3594` seconds before omitted costs. A770 is
therefore retained as an arithmetic survivor, not promoted as performance.
Exact 1M BF16 KV exceeds its 16 GB by `7,065,559,040` bytes, so layer-major or
host/storage streaming is required. The owned H11SSL-i lacks native ReBAR,
Intel warns ReBAR is needed for optimal Arc performance, and the owned Debian
13 system is outside Intel's listed client-GPU Linux matrix. Require an active
complete sub-`$500` BOM and reversible installed oneAPI BF16, PCIe, and
ReBAR-off/on component evidence before purchase or implementation.

PW-0168 replaces PW-0167's generic reference-card and sold-market premises
with one authenticated exact active candidate. GUNNIR specifies its A770
Photon 16G OC at 285-W TBP, two 8-pin inputs, and 300x118.5x50 mm. The active
listing shows `$411` plus `$20` shipping, four available, and import fees
included, leaving `$69` before unknown destination tax and installation parts.
The 285-W board plus 170-W CPU leaves 277 W under the PSU's combined +12-V
label, but exact clearance, original-compatible cables and pinout, checkout
total, cooling, ReBAR, supported oneAPI environment, and installed performance
remain unproved. Retain the candidate pending physical and checkout evidence;
authorize neither purchase nor implementation. The authoritative report
hashes to
`dfd12ca7bb331003e28241e1c5eac49c579eecfa90cb5216fb41edb8a297f6bd`.

PW-0169 finds a preferred domestic reference-board candidate. Intel's A770
Limited Edition is 225 W, requires one 8-pin and one 6-pin input, and measures
279.9x126.36x42 mm at its maximum bracket/connector extents. A direct active
used listing shows `$300` plus `$11.71` rendered shipping, leaving `$188.29`
before actual-destination differences, tax, and installation parts. That is
credible BOM room and the 225-W card leaves 337 W of combined +12-V nameplate
headroom, but seller-described working order is not validation and there are
no seller returns. Four authenticated listing images bind the box's 16-GB
`21P01J00BA` label and the Limited Edition card form, not component function.
Retain physical clearance, original EVGA cables/pinout,
actual checkout, cooling, ReBAR, supported oneAPI, and installed performance as
gates. Prefer this listing over PW-0168 but authorize neither purchase nor
implementation. The installed kill threshold is `118.238594` sustained
BF16/F32-accumulate TFLOPS even when all omitted work costs zero, `90.2585%`
of the derived A770 ceiling. Passing only retains the card. The authoritative
report hashes to
`127a898e54f51044bf68bf58f80d071e98b2e10130f2b008a6fe0d313d2d9db3`.

PW-0170 closes the card-only decode-capacity ambiguity before physical work.
After reserving all non-routed source tensors, three bounded arenas, and exact
8K BF16 KV, the A770's 16 decimal GB leaves room for only 25 complete source
experts. A causal prompt-frequency set avoids just 25 of the real `q=137`
suffix's 903 union records (`2.76855%`), leaving `22,100,987,904` source bytes.
Reject the A770 HBM cache as a primary mechanism.

The strongest deliberately impossible envelope grants four independent
3.5-GB/s lanes, the A770's full derived 131-TFLOPS BF16 peak, and the EPYC's
impossible peak. It still needs `A=56/137` for 34.3 TPS and `A=81/137` for 50;
four 2.5-GB/s lanes need `A=77` and `A=113`. Width-eight/16 speculation is
structurally insufficient. Retain only a base-aligned `q>=137` proposer plus
four-lane storage as an unmeasured conditional branch. The card leaves
`$188.29` before tax, carrier/drives, cables, and cooling, so no complete BOM
or purchase follows. The authoritative report hashes to
`c8eba5c4348378177d0d297b8eb4713fd9be71aa2f5a7c2790895c35859af5af`.

PW-0171 tests that remaining procurement premise against exact active storage.
Four quantity-available used Samsung PM981a 256-GB drives, one observed order
shipping fee, and an active quad-M.2 passive-bifurcation carrier total at least
`$208.10`. With PW-0169's `$311.71` card observation, the favorable lower
bound is `$519.81`, already `$19.81` over the complete cap before tax, GPU
cables, or cooling. Reject this exact active BOM and supersede the claim that
`$188.29` is credible complete-BOM room. Capacity and nameplate are adequate:
four drives provide 1.024 decimal TB and Samsung specifies 3,500 MB/s read per
drive, but neither sustained concurrent bandwidth nor platform bifurcation is
measured. Retain only a price-triggered or already-owned-storage reopening;
authorize no purchase. The authoritative report hashes to
`14549b38ee1daee523fd5a76ca9654cdcf7aa6284c651fb36eccac68908b28d3`.

PW-0172 tests PW-0170's slower four-lane alternative rather than generalizing
PW-0171's rejection. Four active exact-part Samsung PM981 256-GB drives cost
`$115.96` with free observed shipping. With the `$39.99` carrier and
PW-0169's `$311.71` card observation, the pre-tax subtotal is `$467.66`,
leaving `$32.34`. The break-even tax rate is only `7.092883%` if cables and
cooling are free, so no complete delivered BOM or purchase follows. The retail
2,800-MB/s specification for the matching base part is 12% above the inherited
2.5-GB/s grant but is not manufacturer or installed performance evidence.
Retain only the conditional `q=137`, `A=113` 50-TPS (`A=77` 34.3-TPS) branch
pending checkout, physical parts, installed storage/A770 measurements, and a
base-aligned proposer. The authoritative report hashes to
`2b38a618c0364ce2c11a7d93b2bf57e357c38d8cc5f3edfc2da954a6795da564`.

PW-0173 tests whether newer released speculators remove PW-0170's proposer
gap. They do not. After granting one target bonus, the published EAGLE-3,
P-EAGLE, AngelSpec DFly, and BASTION configurations have maximum accepted
paths of 9, 6, 8, and 17 tokens respectively. BASTION is strongest but remains
39 tokens below the least demanding retained requirement, `A=56`. Its largest
reported slice mean of 10.60 is 5.283x below `A=56` and 10.660x below the
slow-storage Prismwing-50 requirement `A=113`, but these cross-model ratios are
diagnostic rather than bounds.

Reject all four audited released configurations as direct Prismwing proposers.
Retain only a separately trained or scaled MiMo-specific `q>=137` form as an
unproven research branch; do not infer feasibility, impossibility, endpoint
TPS, or purchase authority. The authoritative report hashes to
`15ec2cfa3ea80a3914ce500f3cb8288a2149cc1948469aeecde04922f6f7a16d`.

PW-0174 closes the current L3 mismatch-acceptance loophole without
generalizing to an unbuilt scaled verifier. Approximate Speculative Decoding's
primary `DSpark-14B-block7` configuration has a favorable maximum path of
eight with a target bonus, 48 below PW-0170's minimum `A=56`. Its request
budget `B=8` is cumulative regret, not proposal depth. Mean accepted length
rises only from 3.85 to 4.20; the 7.78% mean and 15.26% maximum reported
throughput gains are cross-model paper results, not Prismwing TPS.

The paper also does not qualify the changed trajectories for Prismwing's L3
target. It reports over 95% hash divergence on named tasks and a worst task
point change of -1.52 percentage points, while omitting the hosted top-20
distributional gate, native modalities, one-million context, required paired
confidence intervals, and MiMo execution. Reject the released configuration;
retain only a separately scaled MiMo-specific `q>=137` ASD branch with the full
validation protocol as unproven. The authoritative report hashes to
`2a8bbcc3d70740501fea245e33b28313d23447cfdde205c139f86981e4f4dd6e`.

PW-0175 reopens structured sparse prefill as a distinct changed-attention
branch rather than generalizing PW-0162's probability-ranked oracle. The
released GLM-4-9B-1M MInference head map retains a favorable `1.230279%` of
causal pairs at one million positions; charging the last-64-query online index
QK pass raises effective work to `1.237959%`. The independently reproduced
two-P100/EPYC full-system ceiling permits `21.056139%` of ordinary global work.

This arithmetic promotes only a MiMo-specific source-state oracle. Derive
MiMo's patterns per layer/head, reproduce the online selector, and test output,
route, logit, long-text, and native-modality fidelity before kernel work. Do
not copy GLM's map or infer P100/Metal speed. Quest's released sparse path is
decode-only and explicitly leaves prefill dense, so reject it only as the
PW-0158 prefill repair. No runtime, fidelity, endpoint TPS, or purchase is
promoted. The authoritative manifest hashes to
`e5ac56b7f710285cdeb0088f9fa750748ad74cbc68cd6d4dcb627061209a37ab`.

PW-0176 executes that MiMo-specific oracle on a deterministic 65,536-token
layer-0 slice and rejects all combinations of the five released MInference
pairs. The strongest uniform `(1000,6096)` pair uses `20.599935%` effective
work but reaches `0.055171` aggregate relative L2, `0.884388` maximum-position
error, and `0.723112` p99. Even an impossible best-pair choice at every exact
head-query fails at `0.047658`, `0.721474`, and `0.435570`. Therefore no fixed
layer/head assignment over the released pairs can pass; retain only trained
or repaired selectors with different widths or mechanisms as distinct
research. Raw and analysis evidence hash to
`1d6c4b4fd607fee439b170da0e26e4a9f1c380231a6baa47b009a7fd0061c9a9`
and
`3176fed9199aba3d30ac1916d96ce1b8d5b55fbb005561b16b769873097da0da`.
No throughput-model constant or endpoint TPS changes.

PW-0129 returns to the compact modified representation that can actually
change the M1 traffic premise. Evaluate fixed affine group-128 INT4, with INT8
as a quality control, on PW-0116's real source-routed activations at layers 4,
24, and 46. Preserve source routes and compare complete route-weighted layer
outputs, not isolated matrices. Read only positions `0..167` first; the final
56-position pilot holdout stays sealed unless INT4 reaches at most 1%
aggregate and 2% per-layer validation relative L2, no row above 5%, and at
most 60% of source executable expert bytes. Failure kills naive INT4 before a
full bank, recovery training, speculation, or accumulated execution.

The real-activation audit rejects naive affine INT4 without unsealing holdout.
Its 53.112% byte ratio passes the physical gate, but validation routed-output
relative L2 is 4.192%, 11.917%, and 15.461% at layers 4, 24, and 46; aggregate
error is 9.766% and the worst row reaches 17.922%. Source replays are bit-exact.
INT8 improves those layer errors to 0.970%, 2.406%, and 3.551%, but occupies
103.100% of source bytes and still misses the deeper-layer numerical gate.
Do not build the naive INT4 bank or compose it with cache/speculation. A next
quantized branch must change calibration, outlier treatment, training, or the
representation itself.

PW-0130 gives the cheapest calibration branch an intentionally generous
capacity test. Recompute PW-0129's real INT4 outputs, then fit F16 per-expert,
per-output-channel bias-only and affine repairs on the same validation rows
they are evaluated on. The full repair costs only 4 MiB per layer. Kill this
diagonal output-calibration family if even the same-validation affine oracle
cannot reach 1% aggregate, 2% per layer, and 5% per row while monotonically
improving the nested controls. Holdout remains sealed; a capacity pass would
authorize a separately frozen train-only calibration test, not deployment.

The same-validation oracle rejects diagonal output calibration before a
train-only implementation. Affine repair reduces validation error from
4.192%/11.917%/15.461% to 1.153%/2.485%/4.816% at layers 4/24/46, but aggregate
error remains 2.992% and the worst row 6.913%. Its 4-MiB-per-layer cost is only
0.0651% of the source bank, so capacity—not storage—is the failure. The
remaining error requires input-conditioned and/or cross-channel correction.
Holdout remains sealed.

PW-0131 grants the remaining input-dependent/cross-channel error a compact
two-factor repair on each expert's real MoE input. Stack ranks 8, 16, 32, and
56 on PW-0130's same-validation affine oracle, store all factors as F16, and
charge a full 256-expert bank. Select the smallest rank reaching 1% aggregate,
2% per layer, and 5% per row while the combined representation stays below
60% of source bytes and repair arithmetic below 5% of source expert MACs.
Holdout remains sealed. A same-validation pass only authorizes train-only
generalization; rank-56 failure kills this low-rank output-repair family.

The capacity oracle passes first at rank 32: 0.9493% aggregate validation
relative L2, 1.5825% worst layer, and 2.1885% worst row. The combined INT4,
affine, and low-rank representation is 55.2599% of source bytes and repair
arithmetic is 1.0417% of source expert MACs. Rank 56 nearly memorizes the
same-validation slice, underscoring that this is capacity—not generalization.
Proceed to a frozen train-only rank-32 fit; keep holdout sealed.

PW-0132 fits the exact rank-32 affine-plus-low-rank form only on positions
`0..111` and scores frozen parameters on `112..167`. Validation experts absent
from train use visible identity fallback; strict passage requires complete
coverage plus 1% aggregate, 2% per layer, and 5% per row. A bounded near miss
(2%/4%/8%, complete coverage) may justify broader training-corpus acquisition.
Any larger failure rejects this pilot form. Positions `168..223` remain sealed
regardless; only a validation pass can open a separate holdout record.

The train-only result rejects the pilot repair and does not qualify as the
bounded near miss. Aggregate validation relative L2 is 15.033%, the worst
layer is 17.398%, and the worst row is 57.421%. Layer 4 regresses from 4.192%
uncorrected INT4 to 17.398% after rank-32 repair, despite fitting its training
rows to 0.052%. Layer 24 also exposes 15 validation placements for an expert
absent from train, but both fully covered layers independently fail by large
margins. Keep holdout sealed; do not acquire broader data for this mechanism.
Proceed to weight-domain calibration, outlier-aware mixed precision, or a
different executable form.

PW-0133 takes the cheapest weight-domain branch first. Preserve PW-0129's
affine group-128 INT4 core, rank row-by-128 weight-error groups using only
train activation second moments, and restore the top 1%, 2%, 4%, and 6% from
exact source FP8. Score frozen validation with routes unchanged and holdout
sealed. Charge raw FP8 groups, source block scales, U32 ordinals, and
correction MACs; dense correction matrices are oracle machinery, not the
artifact. A strict pass requires 1% aggregate, 2% per layer, 5% per row, at
most 60% source bytes, and at most 10% extra expert MACs. A 2%/4%/8% near miss
may authorize AWQ composition; a larger miss kills this exception-store form
before a sparse Metal kernel or full bank.

The completed audit rejects that exception-store form. At the largest
admissible 6% group fraction, aggregate validation error falls only from 9.766%
to 8.387%, the worst layer remains 13.737%, and the worst row 17.161%. The
artifact already occupies 59.487% of source bytes and adds 6.001% correction
MACs; 7% would exceed the 60% byte gate. The fully covered layers fail
independently of layer 24's 15 fallback placements. Keep holdout sealed and do
not build the sparse kernel or bank. A next weight-domain test must change the
quantization geometry or error propagation—AWQ scaling, GPTQ-style updates,
function-preserving rotations, or recovery training—not merely retain more
groups chosen by this diagonal proxy.

PW-0134 changes the INT4 grid rather than adding exceptions. For each expert,
use only positions `0..111` to search AWQ's activation-mean scale exponent over
`0.00..0.95`. Search an exact shared gate/up input-channel transform and an
exact up/down hidden-channel transform, compose them, quantize all three
weights, and score only `112..167`. Gate the unquantized transform algebra,
record train-absent fallbacks, and keep `168..223` sealed. Charge both F16
scale vectors even when folded into packed weights. A strict 1%/2%/5% pass
authorizes holdout and a packed kernel; a 2%/4%/8% near miss may authorize AWQ
plus exceptions. A larger failure moves to second-order weight updates,
rotations, or recovery training.

The AWQ-style scale family improves every layer but is decisively rejected.
Aggregate validation error falls from 9.766% to 7.745%; layer errors become
2.563%, 8.381%, and 12.614% at layers 4/24/46, with a 17.501% worst row. The
exact transform algebra passes and selected exponents are nontrivial, so this
is not an identity-control failure. Its 53.161% source-byte ratio is excellent,
but quality misses even the near-miss gate. Keep holdout sealed and do not
compose exceptions. Proceed to a mechanism that propagates correlated error,
changes outlier geometry, or trains recovery weights.

PW-0135 tests correlated quantization-error propagation before paying for all
validation experts. On the highest-coverage experts at layers 4/24/46, apply
group-local GPTQ on the fixed MLX affine-INT4 grid, sweeping 0.1%/1%/10%
damping and natural/activation order per projection using only train
activations. Execute the chosen grid values through an unpacked dense oracle,
prove grid membership, compare the identical unpacked RTN control with
PW-0129's packed control, and keep holdout sealed. Continue to a full-layer
audit only if every expert halves validation error, reaches at most 8% L2 and
12% worst-row error, and improves train without changing INT4 bytes or runtime
MACs. Otherwise move beyond this block-local fixed-grid form.

The frozen control is rejected by a narrow but real miss. Experts at layers 4
and 24 pass with validation relative L2 of 3.305% and 6.644%. Layer 46 halves
its error and passes the 12% row gate, but reaches 8.066% against the 8.000%
absolute ceiling. All nine projections select activation order and 0.1%
damping. Keep holdout sealed and do not expand this exact group-local form to
all validation experts. The large improvement makes second-order assignment a
useful premise, but the next contract must change the mechanism—global-Hessian
coupling, a function-preserving rotation, or recovery training—rather than
retroactively relaxing PW-0135's threshold.

PW-0136 reopens cold routed-layer acquisition under a mechanism PW-0106 and
PW-0108 did not test. Derive eight fixed-stride extents from PW-0106's exact
source-FP8 artifact, `pread` them into eight reusable page-aligned allocations
already wrapped by Metal, and sweep 1/2/4/8 bounded workers under interleaved
cold and warm trials. Require exact full-artifact hashes and real physical-read
evidence. Continue to a slot-owned I/O/Metal scheduler only if cold acquisition
clears PW-0108's unchanged 47.7 ms bound with no trial above 57.723 ms. Failure
kills this I/O embodiment for source FP8, not for a later fidelity-qualified
INT4 artifact whose selected bytes are roughly halved.

The explicit-read branch is rejected for the unchanged source-FP8 payload.
Cold medians for 1/2/4/8 workers are 59.094/58.125/58.205/58.515 ms, with
exactly 201,719,808 physical bytes read in every trial. The selected two-worker
median misses the 47.7 ms bound and all of its trials exceed 57.723 ms. Its
near-identity with PW-0108's 58.034 ms Metal-I/O result shows that neither API
nor additional cold-read concurrency removes the internal-SSD/payload floor.
Warm eight-worker `pread` reaches 13.632 ms, but warm pages are not the target
state. Do not build a source-FP8 slot scheduler. Reuse this exact control only
after a numerically qualified representation materially reduces bytes.

PW-0137 changes the one causal feature PW-0135 left untested: retain one full
input-channel Hessian and carry each 128-column block's quantization error into
all later columns. Reuse the original MLX affine-INT4 grids, PW-0135's selected
0.1% damping and activation order, and test only the narrowly failing layer
46/expert 28 on the same sealed train/validation split. Continue only if it
clears the unchanged 8%/12% complete-expert gates, halves validation error,
improves train, and preserves the exact 13,369,344-byte/no-extra-MAC ledger.
A pass authorizes a separately frozen three-expert confirmation; a failure
moves to a genuinely different geometry or recovery mechanism.

The one-expert rescue passes. Layer 46/expert 28 improves from PW-0135's
`0.080659` validation relative L2 to `0.059227`, with a `0.077608` worst row
and `0.033130` train error. It reduces the identical affine-INT4 control by
63.73% while preserving the exact byte and MAC ledger. This establishes that
cross-group Hessian coupling—not a relaxed gate—contains the missing local
capacity. Keep holdout sealed and run the separately frozen three-expert
confirmation before expanding to a layer or executable artifact.

PW-0138 applies the exact PW-0137 mechanism without tuning to PW-0135's
original layer 4/24/46 experts. Every expert must halve validation error, reach
at most 8% L2 and 12% worst-row error, improve train, and be no worse than its
group-local candidate under the unchanged 13,369,344-byte/no-extra-MAC ledger.
The layer-46 rerun must reproduce PW-0137's metrics and assignment hashes
exactly. A pass authorizes only a separately frozen all-validation-expert
audit; holdout and runtime construction remain closed.

The confirmation passes all three experts. Validation relative L2 is
2.216%/5.960%/5.923% at layers 4/24/46 and maximum-row error is
3.615%/7.979%/7.761%; every expert improves train and beats its group-local
candidate. Layer 46 exactly reproduces PW-0137. Proceed to the separately
frozen all-validation-expert audit under the same mechanism and ledger; keep
holdout and runtime construction closed.

PW-0139 expands the frozen mechanism to all 41 experts selected by validation
at layers 4/24/46 and reconstructs each complete route-weighted BF16 layer
output. Expert-routed train positions calibrate 39 experts; the two layer-24
train-absent experts use a declared all-train-position layer fallback. Require
the original 1% aggregate, 2% per-layer, and 5% worst-row gates, exact PW-0138
reproduction, improved calibration outputs, complete route accounting, and the
unchanged physical ledger. Keep holdout sealed; a pass authorizes only a new
holdout audit.

The all-expert audit rejects the frozen candidate. Aggregate routed validation
error is 3.504%; layers 4/24/46 reach 1.013%/4.069%/5.754%, and the worst row
is 8.279%. Coverage, source reconstruction, projection calibration improvement,
PW-0138 reproduction, and physical checks all pass, so the three-expert result
does not generalize to the routed layer population. Keep holdout sealed and do
not build the bank or runtime. A cheap successor may test pooled-Hessian
shrinkage on the deeper low-calibration-count failures before moving to
rotation or recovery training.

PW-0140 tests that lead on the three deepest low-count failures: layer
24/experts 39 and 128 and layer 46/expert 140. Replace their sparse routed
calibration inputs with all 112 layer train inputs while holding the full-
Hessian algorithm, affine grids, damping, ordering, bytes, and MACs fixed.
Continue only if every pooled candidate improves PW-0139, reaches 8% validation
L2 and 12% worst-row error, and improves every pooled projection control. A
pass authorizes only a train-only shrinkage-policy experiment; holdout and
runtime remain closed.

Pooled calibration improves all three experts but is rejected by the uniform
gate. Layer 24 experts 39 and 128 fall to 6.396% and 6.835% validation error;
layer 46 expert 140 falls to 8.336% but misses the 8% ceiling. This confirms
sparse calibration contributes to PW-0139, yet pooled-only calibration does
not qualify the fixed-grid bank and cannot address the better-covered deep
failures. Keep holdout and runtime closed; move to rotation or recovery.

PW-0141 leaves calibration topology and changes the weight geometry. Apply one
fixed model-wide randomized-Hadamard residual rotation to an early control and
well-covered layer-24/46 experts, prove the unquantized algebra exactly, create
new affine-INT4 grids in the rotated basis, and run unchanged full-Hessian
GPTQ. Continue only if both deep experts improve PW-0139 by 25%, reach 5% L2
and 8% worst-row error, while the early control regresses by at most 10% and
the physical ledger remains unchanged. This is a local capacity control, not a
whole-model rotation or runtime authorization.

The fixed residual rotation is rejected. Its unquantized algebra passes near
machine precision, but validation changes only from 2.216%/6.585%/6.744% to
2.224%/6.504%/6.623% across the three experts. Rotated round-to-nearest is much
worse, showing that GPTQ supplies nearly all of the recovered quality. Keep
holdout sealed and do not rotate the checkpoint or build the runtime; proceed
to recovery training rather than another fixed rotation seed.

PW-0142 tests the cheapest direct recovery-training form without changing the
runtime ledger. Reproduce PW-0139's codes for the three well-covered PW-0141
experts, freeze those four-bit assignments, and jointly train only the existing
group-128 scale and bias arrays through each complete SwiGLU expert. Use one
frozen optimizer schedule on routed train positions, stage final grids through
F16, and score untouched validation positions. Continue only if both deep
experts improve by at least 25%, reach 5% L2 and 8% worst-row error, the early
control remains stable, and the 13,369,344-byte zero-extra-MAC ledger is exact.
Keep holdout, code-changing QAT, bank construction, and runtime work closed.

The frozen PW-0142 form is rejected. Layer 4 remains unchanged after F16 grid
staging, while layer 24 worsens from 6.585% to 34.294% validation L2 and layer
46 from 6.744% to 37.572%; neither deep expert improves even on train. Codes,
PW-0139 reproduction, partitions, physical accounting, and Gate 8 all pass.
Reject this fixed-code group-scale/bias schedule without generalizing the
result to code-changing QAT or other recovery representations. Keep holdout and
runtime construction closed.

PW-0143 repairs a receipt portability defect exposed by the PW-0142 preflight.
macOS changed the APFS mount-session device number while every path, byte count,
inode, nanosecond mtime, and receipt hash remained unchanged. Centralize runtime
file-identity validation and treat `st_dev` as a recorded diagnostic rather
than a durable gate. Continue to fail on any size, inode, mtime, status,
receipt-hash, or model-hash change. This is an L1 correctness repair and does
not weaken checkpoint content authority.

The repair passes. All 39 checkpoint files retain exact size, inode, mtime,
and receipt hashes while consistently reporting only the APFS device-number
drift. Twenty-three focused tests pass, the real shard preflight succeeds, and
PW-0142 completes through the unchanged receipt. Promote the centralized
durable identity predicate as a correctness repair; retain device numbers as
diagnostics rather than restart-invalidating gates.

PW-0144 tests the next distinct recovery mechanism. Hold PW-0139's group-128
F16 grids fixed, initialize dimensionless latent offsets around its exact
four-bit codes, and use a frozen straight-through full-expert optimizer that may
change code assignments. Fit only routed train positions for the three
well-covered controls and score untouched validation. Continue only if both
deep experts improve by 25%, reach 5% L2 and 8% worst-row error, the early
control remains stable, train improves, codes change without leaving `[0,15]`,
and the original byte/MAC ledger remains exact. Keep holdout, bank, and runtime
closed.

The PW-0144 schedule is rejected before generalization. The early expert makes
no effective update; the deep experts change 21.45% and 23.33% of codes but
worsen train error to 15.882% and 17.040% and validation to 68.495% and
44.489%. Initial controls, code domain, F16 metadata, partitions, physical
ledger, and Gate 8 all pass. Do not tune this failed schedule on visible
validation or expand it. Grid-changing QAT or a different representation needs
a new frozen cheap gate.

PW-0145 separates optimizer viability from validation. On layer 46/expert 249
train positions only, restart the exact PW-0144 fixed-grid latent form for 32
steps at learning rates `0.0001/0.0005/0.001/0.005`; never load validation.
Select lowest final train error with a lower-rate tie break. Continue only if
train error falls 25%, loss descends, 0--5% of codes change, code/grid
authority passes, and the runtime ledger remains exact. A pass freezes one
schedule for a separate validation experiment; it is not fidelity evidence.

The tested PW-0145 family is rejected. All four rates change zero codes and
leave train L2 exactly at 3.928%; their maximum offsets range from 0.0032 to
0.1598, below the 0.5 rounding boundary. Validation was never loaded. This
isolates a quantized dead zone rather than a fidelity result. Resolve one
threshold-crossing train-only schedule before any new validation experiment.

PW-0146 is the final schedule test for the fixed-grid latent form. On the same
layer 46/expert 249 train slice, run exactly 32 steps at learning rate `0.02`,
predicted to cross the first 0.5 code boundary without PW-0144's multi-bin
explosion. Never load validation. Continue only for 25% train improvement,
descending loss, 0--5% changed codes, exact grid/code authority, and unchanged
runtime accounting. Failure ends schedule search for this parameterization.

PW-0146 rejects the remaining schedule interval. It changes a bounded 2.346%
of codes and reaches 0.628 latent displacement, but train error explodes from
3.928% to 106.296% and loss from 0.00154 to 1.12988. Validation was never
loaded. Combined with PW-0145's no-change dead zone and PW-0144's large-step
divergence, end schedule search for this fixed-grid independent-offset STE
form. Move to grid-changing training or another executable representation.

PW-0147 changes representation rather than repeating recovery schedules. Build
group-128 affine five-bit grids and run unchanged global-Hessian assignment on
PW-0138's three representative experts. The exact prospective artifact is
16,515,072 bytes per expert (`0.656090` of source), or about 198.7 GB for the
routed bank before padding. Continue only if every expert reaches 2% validation
L2 and 5% worst-row error, improves train and its four-bit control, preserves
all authorities, and adds no runtime MACs. A pass opens only an all-validation-
expert audit; no runtime or hardware purchase follows.

PW-0147 rejects that five-bit form. The early expert passes at `1.990%`
validation L2 and `3.427%` worst-row error, but the layer-24 and layer-46
experts reach only `4.722%/5.410%` and `4.316%/4.971%`, respectively. Every
candidate still improves both its five-bit round-to-nearest train control and
its exact four-bit validation control. Thus the added bit has real value, but
not enough capacity for the unchanged `2%/5%` representative gate. Preserve
the physical result—16,515,072 bytes/expert, 198,709,346,304 bytes/bank, zero
extra MACs—and advance only to a separately frozen six-bit control. No
throughput-model constant changes because no executable bank or endpoint was
measured.

PW-0148 freezes the next code-capacity point without changing the numerical
control or fidelity thresholds. Six-bit codes plus F16 affine metadata require
19,660,800 bytes/expert (`0.781059` of source), or 236,558,745,600 bytes for
the routed bank before padding. Continue only if all three experts clear the
same `2%/5%` validation gates, improve six-bit round-to-nearest and immutable
PW-0147 five-bit controls, preserve all authorities, and add no MACs. The bank
fits arithmetically in 256 GiB but leaves only about 35.6 GiB for everything
else, so a fidelity pass remains insufficient to authorize hardware.

PW-0148 rejects the six-bit form. The early expert reaches `1.921%` validation
L2, but the layer-24 and layer-46 experts remain at `4.407%` and `3.819%`.
All worst rows now clear 5%, every candidate improves its six-bit
round-to-nearest train control and five-bit validation control, and all
authorities pass. The fifth-to-sixth bit therefore has diminishing value and
does not resolve deep-expert generalization. Seven-bit codes would consume
255.5625 GiB for routed experts alone, so further bit-width search is physically
ineligible for a 256 GiB companion once required resident state and headroom
are included. Close this affine/global-Hessian width ladder; continue through a
non-affine representation or a separately measured companion arithmetic path.
No throughput-model constant changes because no executable bank or endpoint
was measured.

PW-0149 changes the shared affine-level assumption. Give every 128-weight
row-group a deterministic 16-centroid F16 codebook and apply the unchanged
global-Hessian assignment on PW-0148's three controls. The prospective form is
18,874,368 bytes/expert (`0.749817` of source), or 227,096,395,776 bytes for
the routed bank, with no added matrix MACs. Continue only if all experts clear
the unchanged `2%/5%` validation gates and improve both nonuniform
round-to-nearest train output and immutable six-bit validation. A pass opens
only an all-validation-expert audit; a failure closes this fixed per-group
codebook form without claiming that all vector or learned quantizers fail.

PW-0149 rejects the scalar-codebook form. Validation relative L2 is
`3.376%/6.060%/6.064%` and worst-row error is
`6.881%/8.157%/7.748%`; all three are worse than immutable six-bit affine and
miss both fidelity bounds. Global-Hessian assignment still drives train error
from `9.443%/17.941%/15.747%` down to `0.544%/4.104%/3.287%`, reproducing the
same train floor without validation transfer. Close fixed per-group scalar
level search. Vector/program quantization and source-preserving companion
execution remain distinct; no bank, kernel, hardware, or throughput constant
is authorized.

PW-0177 realizes one such vector form on the onboard M1 rather than judging a
dense reconstruction alone. A row-scaled, 8-bit-index/two-weight Core ML
codebook compresses a real layer-46 expert to 13,140,830 bytes and executes in
1.4222 ms warm median, but fails validation at `15.9577%` relative L2 and
`18.0525%` maximum-row error. Its 503.257-ms model load also kills dynamic
one-package-per-expert switching independently of fidelity. Preserve the
resident compressed-arithmetic result, but continue only through a resident
shared transaction plus a separately trained or activation-aware low-rate
representation; neither more bits nor route-time package loading changes the
1-TPS physical bound.

PW-0178 attacks that low-rate branch in the input direction with a favorable
private capacity oracle. Its two-index-bit layout would reduce 376 expert
executions to 2,365,587,456 code bytes/token and retain only 246,415,360 bytes
of codebooks across all layers, but complete-expert validation error is
`20.7785%`; gate and up alone miss at `11.9019%/8.4431%`. Since private
256-centroid codebooks dominate a shared same-rate fit, kill the single-
codebook family before a shared bank or kernel. Only a physically charged
residual/multi-codebook repair or genuine recovery training remains distinct.

PW-0179 closes the compact low-rank residual option on PW-0178's core. Rank 96
is the last point inside the frozen byte/MAC envelope yet leaves `19.8209%`
complete-expert validation error; even diagnostic rank 128 reaches only
`19.6296%` while capturing `23.28%/23.18%/40.11%` of the three projection
residual energies. The residual is high-rank. Do not add rank until the traffic
advantage disappears or build a kernel for this representation. Only a
non-low-rank trained representation remains distinct.

PW-0180 tests that final local branch by optimizing continuous vector
centroids through complete-expert loss with every compact code fixed. Train
loss falls 55.875%, proving gradient movement, but validation deteriorates to
`34.0665%` complete error and gate/up errors of `123.458%/74.279%`. This is
memorization, not a transferable executable rule. Kill fixed-index centroid
recovery and do not expand it to shared fitting or tune on validation.

PW-0181 closes the requested existing-M1 one-TPS run. Even an impossible 8 GiB
offline-Belady cache plus perfect I/O/compute overlap leaves an exact lower
bound of 1.221235 seconds after adding attention alone (`<=0.818843 TPS`),
before the remaining endpoint. All physically lower-rate representations now
fail their cheap held-out or generalization gates. With hardware sidecars
excluded and expensive broad training unauthorized after failed prerequisites,
there is no evidence-backed next branch. This is a frontier closure, not a
successful endpoint or a new throughput constant.

PW-0182 reopens that closure only for current MLX microscaling formats. MXFP4
exactly matches the INT4 byte envelope and runs a real expert in 0.5875 ms warm
median, but fails validation at `19.3978%`; NVFP4 and affine group-32 also fail.
PW-0183 then rejects projection-sensitive allocation: `3/3/6` bits fits the
envelope but has `25.5800%` complete error, while `4/4/8` still has `12.3779%`
and costs 31.37% more than INT4. Current direct FP4 kernels change arithmetic
speed, not MiMo fidelity.

PW-0184 tests the distinct training-free activation-sparsity premise suggested
by TEAL and WiSparse. Removing the least important 25% of source-weight columns
per token would be physically sufficient to challenge PW-0181's ideal lower
bound, but the real deep expert reaches `10.8212%` validation error even with
weight-aware scoring. Larger sparsities degrade monotonically. Kill direct
channel deletion before a sparse storage layout or kernel.

PW-0185 tests exact prompt-lookup speculation on PW-0112's 137-token target
suffix. The most permissive rule averages only `A=1.087302`; even impossible
`U=1` leaves 1.002496 seconds of miss acquisition per accepted token before
attention and every other cost. Safer minimum n-grams perform worse. Reject
prompt lookup on this trace; Jacobi/lookahead remains a separate proposer.

PW-0186 executes that separate target-generated branch rather than importing a
published Llama result. One authenticated Jacobi shift raises acceptance to
`A=3` at measured `U=2.268617`, or `A/U=1.322392`. PW-0181's favorable physical
model then gives 0.868016 seconds per accepted token before omitted work. This
passes only the prerequisite for a third/convergence iteration; it is not
endpoint TPS.

PW-0187 advances the authenticated chain to `A=5` at `U=2.050532`
(`A/U=2.438392`), promoting a production-shaped wide Metal verifier rather
than another source iteration. PW-0188 then removes its first physical
prerequisite: a real unaligned checkpoint tensor binds directly through a
page-rounded Metal no-copy buffer with explicit offset and exact GPU/CPU byte
samples. The wide path therefore does not require the rejected approximately
303 GB repacked bank; next establish full FP8 projection parity through that
binding.

PW-0189 rejects the current direct projection under PW-0101 source-BF16
semantics because the existing Metal kernel is the previously named L3
arithmetic. PW-0190 then isolates the physical question under that honest
label: original-shard weight and scale mappings copy zero source bytes and
retain `9.13839e-7` projection relative L2 against the readable L3 authority.
Promote a complete no-copy expert only in modified mode; source fidelity remains
an independent blocker.

PW-0191 composes six such original-shard bindings through a complete real
expert. It reproduces PW-0034's output SHA with zero copied source bytes and a
1.078-ms warm median. This promotes heterogeneous direct-checkpoint expert
scheduling under the explicit L3 label; it does not repair PW-0189's source-
BF16 failure or claim accepted-token throughput.

PW-0192 retains that no-copy embodiment through the shared-weight batch-eight
expert transaction. It reaches 0.263 ms per fixture row and 3.881x per-row
speedup with exact reporting of `accepted_tokens=0` and `A=0`. Promote the
heterogeneous route-scheduling probe; do not turn fixture width into acceptance
or endpoint TPS.

PW-0193 executes the real PW-0187 layer-43 route union from original shards:
17 experts and 64 placements pass mixture parity with zero source copies. Its
32.906-ms warm wall exposes a removable scheduler tax rather than a mapping
failure: fixed batch eight executes 136 rows, 112.5% above the 64 real
placements. Promote a correctness-fixtured count-aware shared-weight kernel;
retain zero accepted tokens and no endpoint claim.

PW-0194 proves that merely reducing the logical row count is insufficient:
runtime-bounded loops preserve parity but regress warm wall to 62.600 ms.
Reject that kernel. Test compile-time width specialization once; if eight named
widths cannot beat PW-0193 by 1.5x, retain fixed batch eight.

PW-0195 passes that final scheduler branch. Eight compile-time widths preserve
the exact PW-0193 output and reduce warm layer wall to 19.745 ms, a 1.667x gain
over fixed batch eight. Promote specialized width selection throughout the
wide L3 verifier; source-BF16 and complete endpoint gates remain open.

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

PW-0150 corrects one draft-input omission in PW-0102: the shipped trained mask
embedding has norm `1.479445`, while pinned-base row 151675 has norm only
`0.0000207`. Substituting the authenticated draft-only mask changes the proposal
from `[264,1773,102092,...]` to `[264,11,11,...]`, proving that the old proposal
was not representative of the shipped mask embodiment. It still fails the
first required target token (`11` versus `13`); target token 13 is draft rank
four with a `2.21875` logit gap. Thus `A=1` and, because `U>=1`, `A/U<=1` without
another target walk. Reject the supplied DFlash-8/mask/base combination; retain
base-trained or materially wider proposers as separate branches.

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

PW-0196 establishes that source dynamic-FP8 input semantics and BF16 output
staging are sufficient for byte-exact parity on a real direct-checkpoint
projection. PW-0197's first wide GPU-resident composition nevertheless fails
the unchanged relative gate despite sub-micro absolute error. Preserve both:
the former promotes the semantic boundary, while the latter forbids treating
boundary placement as proof that batched full-expert arithmetic is
backend-invariant.
PW-0198 eliminates backend SiLU evaluation with an exact finite-BF16 lookup and
produces the identical rejected output. PW-0199 then rejects 16/32/64/128/256-
lane reduction trees. Further layer-local numerical work must change the
accumulation representation; ordinary transcendental or lane-count tuning is
closed.

[PW-0203](../experiments/PW-0203-wide-source-jacobi-endpoint.md) completes the
promoted wide-verifier slice. It preserves PW-0187's exact posterior and
`A=5` while executing retained K/V, all 48 layers, direct-checkpoint Metal
experts and shared projections, and all eight LM-head rows. The correctly
labeled `metal_native_l3` run is 22.743 seconds or `0.21985` accepted TPS with
27.508 GB of physical reads.
This rejects the current `q=8` internal-SSD source-FP8 embodiment for one TPS
and supersedes all layer-only extrapolations. Do not rerun this endpoint for an
ordinary kernel or monitoring tweak. Reopen only after a held-out-passing
executable-byte reduction or a MiMo-specific proposer changes `A/U` enough to
rederive the full physical bound. Hardware sidecars are outside this branch.

## Black-swan budget

No more than 10% of research time before Prismwing 10 goes to black swans:

- GPU texture-format decompression of directly executable expert tiles.
- Progressive bitplanes with exact token-sampling certificates.
- Compute-in-memory or computational-storage devices.
- ANE-resident static expert neighborhoods.
- Certified activation/output memoization.

Each receives a one-week maximum kill test with a predefined success metric.
