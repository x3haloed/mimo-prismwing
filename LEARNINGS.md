# What we know so far

This is a living evidence ledger. It separates observations from deductions and
open hypotheses. Numbers are engineering estimates until a Prismwing experiment
replaces them with checkpoint- and device-specific measurements.

Repository observations below were checked on 2026-08-04 against:

- TurboFieldfare commit `7a99f2a635e3adf7ed0720b882d2edb600f2f0da`.
- Swiftlet commit `d0cf7021b0544bf1ba4f264c592386a54bc49a00`.

## 1. Starting MiMo model

Working architecture facts for XiaomiMiMo/MiMo-V2.5:

- Approximately 311B total and 15B active parameters.
- 48 transformer layers, 47 of them MoE layers.
- 256 routed experts per MoE layer; top eight selected per position.
- Hidden size 4,096 and expert intermediate size 2,048.
- Three MTP modules usable for speculative decoding.
- Approximately 39 sliding-window and nine global-attention layers; SWA window
  128.
- No dense shared expert comparable to Gemma or Qwen hybrid MoEs.
- Native vision, audio, and video input pathways around the language backbone.

With groupwise INT4 plus metadata at roughly 4.5 effective bits/weight:

- One expert is `3 × 4096 × 2048` parameters, or approximately 13.5 MiB.
- The routed bank is approximately 158.6 GiB.
- A cold token touches approximately `47 × 8 × 13.5 MiB = 4.96 GiB` of
  expert weights.
- The remaining language spine has been estimated near 4 GiB, but this must be
  itemized. Embeddings are gathered rather than entirely scanned, and the LM
  head, routers, attention, MTP, norms, and modality projectors have different
  traffic behavior.
- Arithmetic is roughly 30 GFLOP per ordinary token before speculative
  overhead. At 50 output TPS, an efficient verifier is already in the
  1.5–1.8 TFLOP/s class; 100 TPS exceeds the ordinary floating-point class of a
  base M1 unless effective active computation is reduced.

The original cold-streaming design therefore has independent problems in
capacity, SSD traffic, executable unified-memory traffic, compute, and 47-stage
causality.

PW-0002 later pinned the published source representation as block-scaled FP8,
not the candidate INT4 representation used by the initial design estimate.
From the source index and pinned 128×128 scale layout, one routed expert occupies
25,171,968 bytes: three 4096×2048-equivalent FP8 matrices plus three f32 scale
grids. The 47×256 routed bank therefore occupies 302,869,118,976 bytes, and a
cold source-FP8 token selects 9,464,659,968 bytes (8.815 GiB). These figures are
index/config-derived until the full safetensors header census closes PW-0002.

The earlier 13.5 MiB expert and 4.96 GiB cold-token figures remain useful only
for the proposed groupwise-INT4 embodiment. They are not source-checkpoint
measurements. Machine-readable status and provenance live in
`spec/throughput-model.json`.

## 2. What TurboFieldfare established

[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) is strong evidence
for the *shape* of a small-memory MoE runtime:

- Keep common weights, attention, routers, norms, shared expert, embeddings,
  KV state, and scratch resident or bounded.
- Repack routed experts into fixed-stride, per-layer files.
- Read selected expert blobs explicitly into bounded, reusable Metal buffers.
- Use LFU eviction with recency tie-breaking and allow the OS file cache to act
  as an unbudgeted second chance, not as correctness-critical memory.
- Preserve source quantized values during repacking rather than dequantizing and
  requantizing.
- Bind installation artifacts to manifests, layout validation, source
  fingerprints, and hashes.
- Treat prefill and decode as different execution regimes.
- Validate primitives and kernels against simple references with explicit
  tolerances.

Direct references:

- [System design](https://github.com/drumih/turbo-fieldfare/blob/main/docs/SYSTEM_DESIGN.md)
- [Optimization journey](https://github.com/drumih/turbo-fieldfare/blob/main/docs/OPTIMIZATION_JOURNEY.md)
- [Expert streaming implementation](https://github.com/drumih/turbo-fieldfare/tree/main/Sources/TurboFieldfare/Infrastructure/Streaming)
- [Validation lessons](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/summaries/09-validation-and-measurement-lessons.md)

TurboFieldfare's production Gemma path has properties MiMo lacks: only 30 MoE
layers, smaller expert blobs, and a resident shared expert whose compute can
overlap routed-expert I/O. Its success does not predict MiMo throughput by
simple scaling.

## 3. What Swiftlet established

[Swiftlet](https://github.com/leonickson1/Swiftlet) independently extends the
same principle to Qwen hybrid MoEs on macOS and iOS:

- Fixed-stride `.qpack` expert containers allow a miss to become one `pread`.
- A bounded pool of shared Metal buffers prevents virtual-memory thrash.
- Quantized bytes are consumed directly by Metal kernels.
- LFU plus recency is practical on mobile memory budgets.
- The runtime has CPU references, small fixtures, per-kernel comparisons,
  incremental-decode tests, and byte-verifiable containers.
- The current README reports 4.5–5 TPS for a Qwen3-Next 80B-A3B 4-bit model on
  an M5 Mac with roughly 4.3 GB peak RAM, and 7–11 TPS for a 35B-A3B model.
- Reported cache hit rates from 43–70% did not change throughput much in that
  smaller workload; the implementation reports dispatch rather than I/O as the
  current bottleneck.

Direct references:

- [Swiftlet overview and measurements](https://github.com/leonickson1/Swiftlet/blob/main/README.md)
- [Bounded expert cache](https://github.com/leonickson1/Swiftlet/blob/main/Sources/SwiftletCore/ExpertCache.swift)
- [Qpack implementation](https://github.com/leonickson1/Swiftlet/blob/main/Sources/SwiftletCore/Qpack.swift)
- [Metal shard and resident-weight handling](https://github.com/leonickson1/Swiftlet/blob/main/Sources/SwiftletCore/MetalShardStore.swift)

MiMo's estimated 4.96 GiB cold expert traffic per output is about 6.3 times the
roughly 0.79 GiB/token implied by the earlier 80B comparison. MiMo also has more
active parameters and no shared expert. Swiftlet proves implementation
feasibility, not a 50-TPS MiMo path.

## 4. Throughput model

Use separate factors for separate tiers:

- `B_e = 4.96 GiB`: one cold token's routed expert representation.
- `B_d`: executable dense/spine bytes read per verification pass; provisional
  value 4.0 GiB.
- `f_C`: stored expert-bank capacity factor.
- `f_S`: SSD-transfer factor.
- `f_M`: executable-memory factor.
- `h`: byte-weighted demand cache-hit rate.
- `A`: accepted output tokens per verification pass.
- `U`: unique one-token-equivalent expert sets consumed by that pass.
- `q`: positions evaluated by the target verifier.

Approximate storage traffic:

```text
SSD GiB/output = B_e × f_S × U × (1 - h) / A
```

Approximate executable-memory traffic:

```text
memory GiB/output = (B_d + B_e × f_M × U) / A
```

Target arithmetic is closer to `q/A` ordinary-token computations; expert
weight overlap can improve GEMM efficiency but does not remove the per-position
nonlinear computation.

Consequences:

- Prefetching hides latency but does not change the SSD equation.
- Speculation reduces expert bytes only when `A` grows faster than `U`.
- A compact on-disk codec that expands before GEMV improves `f_S` but not `f_M`.
- Cache hits avoid SSD reads but the cached weights still cross unified memory.
- When expert traffic becomes small, `B_d/A` and compute become binding.

## 5. Committee contributions worth preserving

Three independent model analyses converged on several useful reframings.

### 5.1 Expert-major rather than token-major execution

Bucket ready `(expert ID, activation, gate)` records, load each expert once,
evaluate every waiting activation, and scatter outputs. This is target-faithful
and turns GEMV toward GEMM. It can produce perhaps 10–20 exact aggregate TPS on
the M1 at very large batches, but a complete bank sweep at 5 GiB/s takes about
32 seconds, so it is not interactive single-session throughput.

### 5.2 Move activations to weights

An expert consumes an activation of roughly 8 KiB and produces a similarly
small vector while reading about 13.5 MiB of weights. Sending activations to a
DRAM-resident compute node is therefore more sensible than sending expert
weights to the Mac.

The preferred topology gives a node complete contiguous layer ranges—attention,
router, experts, KV, and norms—so network barriers occur only at stage
boundaries. If one affordable high-memory server can run the whole backbone,
that is simpler still. A remote expert-only appliance retains 47 sequential
round trips and cannot overlap adjacent layers of the same stream.

The committee's 40–80 TPS estimates for inexpensive DDR4 servers are
unverified and optimistic. Nominal DRAM bandwidth is not fused low-bit GEMV
throughput. Benchmark one layer before purchasing a system.

### 5.3 Exact neuron canonicalization

A SwiGLU expert's 2,048 coupled gate-row/up-row/down-column units can be
permuted exactly. Canonically align these units across experts before lossless
delta coding or lossy shared-basis analysis. This is analogous to motion
compensation before residual coding and is a cheap, target-faithful experiment.

Arbitrary gate rescaling is not exact because SiLU is not homogeneous. Any
normalization transform must be proved for the precise expert equation.

### 5.4 Shared bases plus expert residual programs

The highest-upside modified-model path is to express experts as resident common
bases plus small directly executable residuals, then recover quality by
distillation. Approximate bank sizes for three rank-`r` residuals per expert at
4.5 bits are:

| Residual rank | Approximate routed bank |
| ---: | ---: |
| 16 | 1.9 GiB |
| 32 | 3.7 GiB |
| 64 | 7.4 GiB |
| 128 | 14.8 GiB |
| 256 | 29.6 GiB |

Shared bases, metadata, the spine, KV, encoders, and buffers are additional.
Rank 16–32 would create a comfortable on-M1 bank; rank 64 might fit only at
short contexts. Singular-value energy alone is not sufficient: measure
activation-weighted expert-output error and downstream route/logit divergence.

### 5.5 Draft-side freedom

A draft model, route-overlap bias, cache-aware candidate count, or shadow expert
bank can be approximate while the overall sampler remains target-distribution
preserving—provided the verification and correction algorithm is mathematically
exact and uses the actual proposal probabilities required by that algorithm.

This is the safest place for creative approximations. It does not make target
verification or its expert traffic free.

### 5.6 Multimodal prefill

A large image/video/audio prefill can collectively touch nearly every expert.
Even expert-major prefill then approaches one complete 158.6 GiB bank sweep.
Decode TPS and multimodal time-to-first-token must therefore be measured
separately.

## 6. Our viability assessment

- Exact interactive 50–100 TPS on the base M1 through SSD streaming is not
  credible.
- Exact expert-major batching is credible for aggregate offline throughput but
  not interactive latency.
- Exact hardware assistance should move complete layers or the whole backbone
  to DRAM-rich compute. It remains kernel- and compute-limited after storage is
  solved.
- Fifty TPS on the M1 itself requires a profound executable representation
  change plus MTP/DFlash acceptance around four or better. It is physically
  conceivable but high risk.
- One hundred TPS on the M1 is likely arithmetic-limited even with negligible
  expert bytes.
- Expert-only aggressive quantization is better motivated than sub-INT4 spine
  quantization; shared components affect every token and routing decision.
- ANE expert execution, texture codecs, progressive exact sampling, and
  compute-in-storage are black-swan experiments, not architectural pillars.
- Native MTP and the published MiMo-V2.5-DFlash artifact make speculative
  decoding the first runtime multiplier to measure, not proof of a local
  speedup.

## 7. Highest-information unknowns

1. Exact checkpoint census and executable `B_d`.
2. Byte-weighted cache curves, including offline Belady, for 2–8 GiB.
3. Conditional route entropy, reuse distance, and session working sets.
4. Joint distributions of acceptance `A`, verifier width `q`, and route union
   `U` for MTP and DFlash.
5. Real M1 throughput for one 13.5 MiB expert at batch 1–64, including
   dequantization and dispatch.
6. Lossless residual entropy after exact neuron canonicalization.
7. Activation-weighted rank/error curves across representative layers and
   rare experts.
8. Native vision/audio/video token counts and multimodal prefill traffic.
9. OpenRouter endpoint stability, provider quantization, and logprob support for
   every modality.
10. End-to-end low-bit expert throughput on any candidate companion hardware.

The experiment plan converts these unknowns into explicit kill gates.

## 8. Workflow evidence

The inspired repositories contribute a development method as well as runtime
ideas. This was not fully captured in the first Prismwing specification.

TurboFieldfare documents a repeated measured loop across 103 curated
experiments: profile the complete token path, isolate the largest share using
production-shaped conditions, make one focused change, apply the right
correctness oracle, then return to an interleaved end-to-end comparison. It
preserves negative results, conditional wins, correctness repairs, and reversed
decisions. Particularly important lessons are that warm `mmap` results did not
predict cold expert reads, attractive cache simulations could be neutral in a
full decode, first-run gains disappeared after thermal-balanced interleaving,
and reductions in allocations or dispatches did not necessarily increase TPS.

Swiftlet shows a complementary endpoint-first workflow. Its plan begins from
multiple executable answer keys, an explicit reuse/do-not-port map, and an
up-front memory budget. It then climbs from deterministic tiny fixtures to a
slow CPU forward pass, per-layer teacher fixtures, Metal parity, whole-model
decode, and only then performance work. Layer-local fixture inputs prevent
accumulated floating-point drift from obscuring the layer that introduced a
semantic error. Its public commit history is mostly a consolidated
implementation, so `PLAN.md`, fixture generation, and tests are stronger
workflow evidence than commit archaeology.

Prismwing adopts both patterns in [the research and delivery
workflow](docs/WORKFLOW.md) and keeps actual experiment history separately from
the prospective plan under [the experiment ledger](experiments/README.md).

## 9. OpenRouter reference viability

PW-0001 established that, on 2026-08-04, Parasail was the only discovered
OpenRouter MiMo-V2.5 endpoint advertising both `logprobs` and `top_logprobs`.
A request pinned to Parasail with fallbacks disabled and all parameters
required returned 20 alternative logprobs at every visible text token position.

The initial request exhausted its completion budget on hidden reasoning and
returned no scored text. Explicitly disabling reasoning produced the expected
visible completion and complete top-20 payload. Reference fixtures must
therefore pin reasoning disabled; accepting a successful HTTP response without
checking visible token-level evidence would be a false pass.

Subsequent PW-0001 probes established the same evidence path for single-image,
multi-image, audio, video, and mixed image/audio inputs. Each response named
Parasail, contained visible output, and supplied exactly 20 alternatives for
every scored token position. The tiny synthetic tasks also received
semantically correct responses.

Parasail is therefore promoted as the initial reference provider. This proves
reference-path viability, not a frozen final epoch or future availability; the
endpoint remains an external single-provider dependency and requires drift
canaries.

## 10. First executable MiMo semantic

PW-0003 confirms from the pinned Xiaomi source that `noaux_tc` routing uses
sigmoid scores plus correction bias for group/expert selection, but gathers and
normalizes the original uncorrected sigmoid scores for mixture weights. The
expert equation is `down(silu(gate(x)) * up(x))` and token output is the
weighted sum of selected expert outputs.

A reproducible two-token tiny fixture now defends that distinction in an
independent Rust f64 scalar implementation. This is component evidence only;
FP8 decoding and sampled-real tensor parity are not yet proven.

PW-0004 subsequently crossed the first sampled-real representation boundary.
The pinned MTP shard stores quantized projection weights as safetensors
`F8_E4M3` with one f32 inverse scale per 128×128 block. An independent Rust
E4M3FN bit decoder exactly reproduced 32 real library-decoded values, and block
dequantization matched within `1e-9` in f32.

An exhaustive fixture subsequently matched PyTorch's f32 output bits for all
256 byte patterns, including subnormals, signed zero, maximum finite values,
and signed NaNs. This supersedes the statement that FP8 format edge cases were
unproven.

PW-0005 then exercised four real MTP projection rows at the production input
width of 4,096, crossing all 32 column-scale blocks per row. The dependency-free
Rust scalar GEMV matched safetensors/PyTorch f32 matmul within `2e-7`. This
supersedes the absence of production-width accumulation evidence for a small
row slice, but full matrices and accelerated kernels remain unproven.

PW-0006 crossed the first real accelerator boundary. An MSL kernel consumed the
same production-width raw FP8 rows and scale blocks directly on the Apple M1
GPU, with maximum absolute error below `8e-9` against the frozen oracle. This
supersedes the statement that accelerated kernels were wholly unproven for the
slice. The one-thread-per-row kernel is correctness machinery only; full expert
shapes, parallel reduction, fused SwiGLU, and end-to-end speed remain unproven.

PW-0007 measured that transparent kernel on a full 16,384×4,096 real MTP
projection. Warm GPU median was 9.618 ms, or 6.498 GiB/s of logical FP8 weights.
Because the matrix contains eight routed-expert gate-projection byte
equivalents, the measured bandwidth implies only about 0.737 routed-only TPS
for a cold source-FP8 ordinary token before all other work. The kernel missed
its 10 GiB/s kill threshold and is rejected as a performance architecture while
retained as a correctness oracle. Parallel column reduction is the next test.

PW-0008 combined parallel reduction, a 256-entry FP8 decode table, and loops
aligned to the source 128-column scale blocks. The selected 64-lane kernel
repeated at 37.39 and 38.14 GiB/s between stable 6.49–6.50 GiB/s controls, a
5.82× gain with unchanged correctness. It is promoted as the accelerated FP8
projection baseline, not an endpoint performance default.

The result makes a hard dependency visible. At 37.765 GiB/s, source-FP8 routed
bytes alone permit only 4.284 ordinary-token equivalents per second. An
otherwise-free 50 TPS endpoint would still need `A/U >= 11.67`. Even applying
the same bandwidth to the estimated INT4 representation requires `A/U >= 6.56`.
Real dense/spine traffic and compute make both necessary conditions optimistic;
MTP/DFlash acceptance and route-union measurement are now a primary risk
frontier.

PW-0009 prevents a tempting checkpoint substitution. The official
MiMo-V2.5-DFlash repository bundles the same 73,081 target tensor assignments
and equal per-shard payload sizes, but 48 deterministic payload samples across
all 16 target shards differ from the pinned `XiaomiMiMo/MiMo-V2.5` revision.
Its bundled target is therefore a distinct weight set, not merely a
safetensors-header repack. The base checkpoint remains authoritative and
published DFlash acceptance cannot be transferred to it without measurement.

The shipped DFlash block verifier is target-preserving for temperature-zero
argmax: it accepts the consecutive matching draft prefix and inserts the
target's first mismatch token. Its positive-temperature branch independently
samples draft and target tokens without speculative-sampling correction, so it
is not proven target-distribution preserving. The downloaded DFlash draft is
an L2 candidate only for greedy decoding until a correct sampling verifier is
implemented and tested.

PW-0010 closes the published DFlash-8/source-FP8 branch on the measured M1
kernel without waiting for a whole-model runtime. DFlash commits at most eight
tokens per target pass, while a non-empty pass has `U >= 1`. Its maximum
possible `A/U` is therefore 8, below PW-0008's source-FP8 requirement of
11.6705. Even perfect acceptance, perfectly identical routes, and zero cost for
everything else cap routed-only throughput at 34.275 TPS.

The groupwise-INT4 estimate is not killed by that structural bound, but its
remaining window is narrow: at the same measured bandwidth and perfect
eight-token acceptance it needs `U <= 1.21895`, equivalent to only about 9.75
unique experts per layer across all eight positions. This supersedes the looser
claim that DFlash acceptance alone was the next unknown; joint `A/U` is the
decisive quantity, and source FP8 has already failed it for the published block
size.

PW-0011 replaces the hypothetical 4.5-bit bandwidth projection with a real
directly executable candidate. Symmetric signed INT4 with one f32 scale per
row×128 group occupies 12.75 MiB per routed expert and 4.68164 GiB per cold
ordinary token. A real 16,384×4,096 M1 Metal projection repeated at 1.0710 and
1.0705 ms between FP8 controls, a 1.53× projection-time gain and 31.009 GiB/s
of physical INT4-plus-scale traffic.

That gain does not rescue DFlash-8 as a Prismwing 50 argument by itself. The
measured INT4 path needs `A/U >= 7.5488`; perfect eight-token acceptance permits
only `U <= 1.05977`, or about 8.48 unique experts per layer across the entire
block. Its routed-only ceiling is 52.989 TPS before dense work and all runtime
overhead. A complete, faithful endpoint below 50 would still be a valuable
project result and milestone; these idealized bounds are branch-selection
evidence, not achieved throughput or a reason to abandon delivery.

The embodiment is also explicitly L3. Four real production-width rows show
9.84% relative L2 projection error versus source FP8, despite 0.9961 cosine
similarity. This does not predict whole-model quality, but it prevents the
kernel's exact agreement with its quantized oracle from being mistaken for
target fidelity.

PW-0002's pinned remote-header census supersedes the provisional 4 GiB shared
spine estimate. Source tensors contain 5.6762 GiB of attention/norm weights,
1.1641 GiB each for the LM head and token embeddings, 0.1875 GiB for dense
layer zero, and 0.1836 GiB of router matrices. Embeddings are gathered rather
than fully scanned during decode, but a DFlash target verification that emits
all eight posterior logits scans at least the attention, dense-zero, router,
LM-head, and small correction tensors once per pass: 7.21145 GiB before routed
experts.

Combining that source shared floor with PW-0011's INT4 experts gives 11.8931
GiB per idealized `U = 1` DFlash-8 pass, or 1.48664 GiB per output at perfect
`A = 8`. Fifty TPS would require 74.33 GiB/s across all of those heterogeneous
kernels. This is still a necessary traffic model, not an endpoint measurement;
the local full-path implementation must determine attainable throughput and
is valuable even when it does not reach 50.

PW-0012 shows why bytes and `A/U` are necessary but insufficient. Applying one
INT4 projection to eight positions takes 4.423 ms in Prismwing's best readable
kernel, not the 1.071 ms batch-one time. Across three projections and 47 layers,
that current kernel corresponds to only about 12.8 routed-only accepted TPS at
perfect `A = 8, U = 1`.

MLX 0.31.2's optimized affine-INT4 quantized matmul improves the same real
batch-eight projection to repeatable 2.668–2.694 ms and roughly 400 GFLOP/s.
The equivalent optimistic routed-only diagnostic is 21.16 accepted TPS. Its
affine four-row relative L2 error is 4.09%, better than the symmetric candidate
but still L3. A fused real-expert path may improve the diagnostic, and a real
endpoint around 10–25 TPS remains valuable; neither point converts this
component measurement into endpoint throughput or fidelity.

PW-0013 establishes the first real M1 storage artifact. `PWEXPRT1` copied an
actual 4,096×2,048 routed-expert down projection and its scale grid without
conversion, bound them to the complete locked source-shard SHA-256, aligned
both payloads, and passed independent verification. The format refuses
overwrite and detects a one-byte mutation.

This promotes the container schema, not a complete expert or runtime. The real
gate/up tensors live in the paired shard still being acquired; once present,
the same format can carry all six weight/scale tensors and feed the fused
expert experiment.

PW-0014 replaces the down-projection shape assumption with an actual routed
tensor. Eight copies of the real 4,096×2,048 layer-43/expert-32 down matrix run
at 2.634 and 2.598 ms for batch eight, 2.4% faster than the equal-byte gate/up
proxy. The refined idealized routed-only diagnostic is 21.33 accepted TPS at
perfect `A = 8, U = 1`.

The actual down fixture also increases fidelity concern: four deterministic
projection outputs have 15.51% relative L2 error after affine INT4. This is not
representative activation or whole-layer evidence, but it makes layer-local
real-activation validation a hard gate rather than follow-up polish.

PW-0015 replaces three isolated projection calls with the first complete real
routed expert: actual layer-43/expert-32 gate and up qmatmuls, SwiGLU, and the
actual down qmatmul. MLX affine INT4 repeats at 1.3445 and 1.3124 ms for eight
positions on one expert. A conservative sequential schedule for the same eight
experts across all positions (`A=8, U=1`) corresponds to 16.02 routed-only TPS
across 47 layers. This supersedes PW-0014's 21.33 TPS concatenated-projection
proxy as the more causally complete diagnostic, while remaining neither an
optimized schedule nor endpoint throughput.

The complete expert differs from its source-FP8 computation by 15.96% relative
L2 at batch one and 15.48% at batch eight, with cosine near 0.988. The fixed
synthetic input cannot predict accumulated model quality, but the candidate
crosses the experiment's 10% caution threshold. Affine INT4 therefore remains
conditional until real router-selected activations, whole-layer state, and
eventually hosted distributional gates pass.

PW-0015 also proves a useful acquisition path: exact tensor byte ranges can be
materialized from a pinned Hugging Face revision and hashed locally, avoiding a
34.37 GB transfer on the critical path. This is not full checkpoint closure;
the remote whole-file SHA is still only its locked LFS identity until the local
file completes and hashes successfully.

PW-0016 extends that path through the actual layer-43 noaux_tc router and all
nine heterogeneous experts selected across an eight-position deterministic
block. The observed union is only nine unique experts (`U=1.125`), and MLX's
selected sets match the source-derived router exactly. This is encouraging for
DFlash-8's stringent route-reuse requirement, but one synthetic input at one
layer is not a route-union distribution and cannot promote the branch.

The complete routed block, including router scores, normalized weights, exact
selected-position expert batches, and weighted summation, repeats at 9.906 and
9.756 ms per layer. Reusing that layer cost across all 47 routed layers gives a
17.31 routed-only TPS diagnostic. It supersedes PW-0015's sequential
single-expert extrapolation, but still excludes every non-MoE endpoint cost and
uses a fixture-specialized dispatch schedule.

Affine INT4's weighted block output has 17.02% relative L2 error against source
FP8 with 0.9855 cosine. Error did not cancel through routing and aggregation;
it increased from the complete single-expert fixture. Current affine INT4 is
therefore a performance substrate and falsification candidate, not a
quality-qualified target embodiment.

PW-0017 supersedes the tempting interpretation that PW-0016's `U=1.125` might
be representative. The earlier input amplitude was 0.01, so learned correction
bias dominated routing. With independently RMS-normalized inputs, layer 43's
median unique-expert count across eight positions is 54 at zero correlation,
23.5 at correlation 0.9, 12 at 0.99, and nine at 0.999. Only identical inputs
reliably produce exactly eight unique experts.

This synthetic sweep does not estimate real DFlash routes; its result is that
the narrow low-union regime is extremely correlation-sensitive. Actual
hidden-state route traces are now required before inserting a favorable `U`
into any throughput claim. The fixed DFlash block size and measured kernels
remain useful bounds, but synthetic route reuse cannot rescue them.

PW-0018 reverses the assumption that intermediate affine precisions offer the
natural fidelity/performance tradeoff on M1. On a complete real expert, 5-bit
and 6-bit are both slower than 8-bit. Affine 8-bit reduces source-FP8 relative
L2 error from INT4's 15.48% to 0.912%, while its two interleaved medians average
only 4.6% slower than INT4. The optimized kernel shape matters more than packed
byte count for this warm component.

On the heterogeneous routed block, affine 8-bit repeats at 11.146 and 11.113
ms and 1.026% relative L2, compared with INT4's 9.906/9.756 ms and 17.02% error.
Its 47-layer routed-only diagnostic is 15.29 TPS. The representation is 3.1%
larger than source FP8, so it is not embodiment compression; it is a
quality-oriented use of MLX's executable substrate. It is selected for a
predeclared validation gate, not yet promoted for target fidelity.

PW-0019 validates that selection under criteria committed before execution.
Three paired process orders give mean medians of 10.1155 ms for INT4 and
11.1265 ms for INT8, a 9.995% slowdown versus the fixed 20% limit. Every route
and fixture check passes; INT8 block relative L2 remains 1.026% (limit 2%) and
cosine 0.999947 (minimum 0.9998).

Affine INT8 is therefore the default quality-oriented research substrate.
This is not target promotion: the evidence is still one synthetic-input MoE
block, the representation is L3 and larger than source FP8, and accumulated
whole-layer/logit behavior is unknown. INT4 remains available for compact
performance experiments but is not the default quality candidate.

PW-0020 rejects direct reuse of Atomic Chat's llama.cpp TurboQuant fork for
MiMo, while preserving it as valuable implementation evidence. The compiled C
formats are 34, 50, and 68 bytes per 128 values for Turbo2/3/4. With MiMo's
192-wide K padded to 256 and 128-wide V, the exact one-million-token hybrid KV
footprints are 3.590, 5.279, and 7.179 GiB, versus 22.524 GiB for FP16. This
supersedes paper-level raw-bit estimates that omitted block overhead and K
padding.

The blocking mismatch is topological, not theoretical: cache allocation and
query rotation correctly make K 256-wide, and Metal dispatch uses that padded
dimension, but the fork has no `dk256_dv128` flash-attention specialization.
Its visible `dk192_dv128` specialization cannot be selected by that path.
Direct reuse is killed; a minimal corrected port is the active branch.

Exact WHT preserves a deterministic MiMo-shaped attention score vector to
`1.05e-7` relative L2, but low-bit quantization is not automatically
fidelity-safe. On the same synthetic causal-attention fixture, output relative
L2 is 52.2% for Turbo2, 22.7% for Turbo3, and 23.0% for Turbo4. These are
component diagnostics, not model-quality estimates; real attention activations
and hosted-reference logits remain mandatory.

PW-0021 repairs the direct-reuse blocker with a Prismwing-owned runtime-compiled
Metal path at the actual K=256/V=128 shape. The kernel performs query WHT,
packed Turbo3/Turbo4 dequantization, online causal softmax, weighted V
accumulation, and inverse WHT. It matches an independent scalar reference from
context 17 through 8,192 at worst `8.33e-7` relative L2 with intact guard bytes.
This promotes an accelerated correctness reference, not a runtime default.

The single-thread schedule is decisively too slow: at context 8,192 its warm
GPU medians are 933.8 ms for Turbo3 and 879.4 ms for Turbo4. This branch is
killed for performance and retained as the oracle for a parallel reduction.
Synthetic quantization error also remains material across contexts, with
Turbo3/Turbo4 output relative L2 generally around 20–30%; real activations are
still the fidelity gate.

PW-0022 replaces that serial schedule with an associative 32-lane online
softmax reduction. At context 8,192, two paired process orders improve mean GPU
median from 931.5 to 31.86 ms for Turbo3 and from 883.2 to 29.99 ms for Turbo4:
29.24× and 29.45× gains. The result is stable across order and comfortably
passes the predeclared 8× gate.

The parallel merge remains numerically faithful to the scalar reference: worst
relative L2 is `3.46e-6` through context 8,192, including a 17-token
nonmultiple-of-32 case. This promotes the 32-lane synthetic component schedule,
not endpoint throughput. Multi-head GQA scheduling, attention projections,
RoPE, KV append, all layers, and real activations remain outside the slice.

PW-0023 closes the scheduling gap for MiMo's 64 Q heads: global attention maps
them to four KV heads (16:1) and SWA maps them to eight (8:1). All 8,192 output
elements agree with scalar GQA at worst `2.29e-6` relative L2. The complete
synthetic attention core is therefore causally real, but its first schedule is
not performance-ready.

At context 8,192, a Turbo4 global layer costs 125.5 ms GPU median and a
128-token SWA layer costs 3.46 ms. Applying those component costs across nine
global and 39 SWA layers is roughly 1.26 seconds before projections, MoE, MTP,
or endpoint work. This makes multi-head attention a measured bottleneck. The
next optimization must share each KV-head scan across its GQA query group;
attention can no longer be omitted from throughput reasoning.

PW-0024 adds three source-required semantics to that executable path: partial
RoPE covers 64 of 192 Q/K dimensions with base 10,000,000 globally and 10,000
for SWA; V is scaled by 0.707 before cache quantization; and each SWA Q head has
a sink logit that adds denominator probability mass with zero value numerator.
Metal matches the scalar source equations at worst `4.09e-7` relative L2.

These semantics now advance to the real transformer-layer fixture. Learned
QKV/output projections, norms, sink biases, and actual layer inputs remain
unavailable until the common `model_pp0_ep0_shard1.safetensors` payload is
acquired or selectively materialized.

PW-0025 removes PW-0023's redundant per-Q-head KV dequantization. A threadgroup
per KV head loads eight-token packed tiles once, then shares them across 16
global or eight SWA query simdgroups. Two paired process orders improve global
context-8,192 GPU medians by 8.96× for Turbo3 and 8.68× for Turbo4, with worst
scalar-relative L2 only `1.46e-6`.

Turbo4 now costs 13.02 ms per global attention core at context 8,192 and 0.360
ms per 128-token SWA core. Applying those component medians across nine global
and 39 SWA layers gives roughly 131.22 ms, down from 1.264 seconds. This is a
material architecture promotion, but remains attention-only and synthetic.

PW-0026 advances attention to learned tensors using the complete local MTP
file: actual input RMSNorm, fused FP8 QKV with block scales, learned SWA sinks,
and BF16 output projection. The Metal packed-cache result matches its scalar
reference at `2.28e-7` relative L2, proving the implementation path. Selected
MLX QKV values match float64 scalar dots within `1.92e-6`.

Uniform Turbo4 is not fidelity-safe by assumption. On deterministic
production-width hidden states, it changes learned attention output by 18.58%
relative L2 and the projected 4,096-wide sublayer by 19.43%. The next fidelity
branch should test higher-precision K and mixed K/V cache formats before
whole-layer promotion.

PW-0027 shows that neither cache side alone explains that error. Source K with
Turbo4 V leaves 15.49% projected error; Turbo4 K with source V leaves 13.58%.
Both sides materially contribute, and upgrading K alone is not a sufficient
quality branch on this learned fixture. The next search must sweep joint K/V
precision before optimizing another packed kernel.

PW-0028 supersedes uniform Turbo4 as the default learned-KV fidelity branch.
On the exact PW-0026 fixture, joint WHT-affine projected error falls
monotonically from 20.98% at 4-bit to 10.00% at 5-bit, 4.34% at 6-bit, and
1.058% at 8-bit. Turbo4 remains 19.43%; its Lloyd-Max representation is not
equivalent to affine4 and neither 4-bit choice is quality-safe by assumption.

WHT-affine8 is now the accelerated-implementation candidate, not a target
default. Charging an FP16 scale plus packed codes per 128 values gives an exact
modeled maximum-context hybrid KV footprint of 13.725 GiB, 39.0625% below
FP16's 22.524 GiB but 1.912 times Turbo4. One deterministic context-17 MTP
sublayer does not establish accumulated model fidelity or endpoint speed; the
shared-KV Metal kernel and then base-layer/hosted gates remain mandatory.

PW-0029 makes the selected cache representation executable without paying the
naive byte-ratio penalty. Across two process orders at global context 8,192,
WHT-affine8 shared-KV Metal averages 13.258 ms versus Turbo4's 13.477 ms, even
though it reads 12.85 MB versus 6.75 MB. Signed-byte dequantization is simple
enough to offset the extra packed traffic in this component schedule; that is
not a general bandwidth claim.

Correctness remains tight from global context 128 through 8,192 and SWA-128:
worst synthetic scalar-relative L2 is `1.44e-6`. The independently generated
learned MTP fixture passes at `2.80e-7`, while retaining PW-0028's 1.058%
projected error versus source. Nine global plus 39 SWA cores give a 133.80 ms
8K-context attention-only diagnostic. Affine8 is promoted into the complete
layer branch, but neither this timing nor one MTP sublayer establishes endpoint
TPS or target fidelity.

PW-0030 closes the first complete learned decoder-block causal path. The actual
8,192-value affine8 Metal attention result flows through the layer-zero BF16
output projection, attention residual, pre-MLP RMSNorm, learned source-FP8
16,384-wide dense SwiGLU, and final residual. Scalar spot checks for fused QKV
and all three MLP projections are within `1.92e-6` absolute error.

On the deterministic MTP final-token fixture, candidate/source relative L2 is
0.984% at attention, 0.461% after projection/residual, 0.429% after RMSNorm,
0.627% at the MLP output, and 0.620% at the final block state. This promotes a
complete MTP correctness reference, not model fidelity: MTP is not a base
layer, and one block cannot reveal accumulated error, MoE routing changes, or
hosted-logit behavior. EP0 remains the required base-layer transition.

PW-0031 establishes the Rust runtime's first direct source-checkpoint
authority. A read-only memory map validates the complete safetensors header,
known dtypes, shape-derived byte counts, overflow-safe offsets, payload bounds,
ordering, and non-overlap before returning an immutable tensor view. Duplicate
JSON keys and ambiguous or malformed layouts fail closed.

Native views of four real MTP tensors match an independent Python raw-range
oracle exactly in dtype, shape, offsets, bytes, and SHA-256. Hashing the
60.8-MB QKV tensor through the full CLI takes 0.17–0.18 seconds with warm OS
cache; that diagnostic is not storage-cold latency or model throughput. Future
native paths should consume this one authority rather than adding another
safetensors parser.

PW-0032 advances that authority into real native computation. The Rust binary
projects the exact PW-0026 normalized hidden state through the complete learned
14,848×4,096 fused FP8 QKV tensor and its mapped 116×32 scale grid without
copying or pre-decoding weights. All 14,848 outputs match the MLX oracle at
`1.15e-6` relative L2 and `1.67e-5` maximum absolute error; Q/K/V scalar-row
checks are within `1.32e-6`.

The readable single-thread path repeats byte-identically and takes 0.30 seconds
warm (0.69 seconds first recorded), including validation, output `fsync`, hash,
and JSON. This is the production native correctness reference, not the
accelerated default or a TPS result. Its stable mapped-byte contract is now the
oracle for a Metal or pinned-MLX projection path.

PW-0033 puts that exact authority behind Rust-owned Metal execution. The
production 14,848×4,096 fused QKV projection matches the readable Rust output
at `1.14e-6` relative L2 and `1.57e-5` maximum absolute error, and repeated
complete processes produce byte-identical output. Layout and source semantics
remain in the shared Rust validator; Metal receives explicit immutable buffers
and dimensions.

With five warmups and 30 serialized resident-buffer measurements, two process
medians are 1.622 and 1.678 ms. The first is 184.95× faster than PW-0032's
300 ms whole-command diagnostic, but that ratio is deliberately asymmetric:
the baseline includes startup, mapping, validation, output `fsync`, hash, and
JSON. This promotes the accelerated projection primitive, not a complete layer,
token path, or endpoint TPS result.

PW-0034 composes three of those validated projections with an independently
fixed F32 SwiGLU into the first Rust-owned, target-faithful complete routed
expert. The actual layer-43/expert-32 gate/up/SwiGLU/down output matches an
independent Torch source-FP8 oracle at `4.70e-7` relative L2 and `4.46e-11`
maximum absolute error. Repeated processes are byte-identical.

Two resident-buffer complete-expert medians are 1.021 and 1.079 ms at batch
one. Serially repeating the first cost for eight experts across 47 routed
layers yields only a 2.605 routed-only token-position/s diagnostic before any
non-MoE work. That kills naive batch-one serial source-FP8 execution as the
performance schedule, not the faithful primitive: heterogeneous expert
batching, route reuse, and MTP acceptance remain the decisive next mechanisms.

PW-0035 shows that merely flattening batch and output rows does not realize
that reuse. The source-FP8 batch-eight expert is correct (`1.63e-6` relative L2
versus Torch), but two medians are 5.131 and 5.111 ms. Per-position improvement
over PW-0034 is only 1.59–1.60×, failing the predeclared 2× and 4 ms gates.

The kernel assigns each position to a separate threadgroup, so all eight
positions reread the same weights. Its idealized perfect-reuse routed-only
diagnostic is just 4.15–4.16 TPS before non-MoE work. This schedule is rejected;
the next batching mechanism must share weight tiles across positions inside a
threadgroup or delegate to a tuned GEMM primitive.

PW-0036 realizes the missing mechanism. One row threadgroup decodes each FP8
weight once and accumulates all eight positions in 2 KiB of threadgroup memory.
Across paired process orders, candidate mean median is 1.9348 ms versus 5.1324
ms for PW-0035 control, a stable 2.653× complete-expert gain. Per-position cost
is 0.24185 ms, 4.221× better than PW-0034 batch one.

Every candidate/control output is byte-identical and remains `1.63e-6`
relative L2 from independent Torch source FP8. The idealized perfect-reuse
routed-only diagnostic rises to 10.997 TPS, still before non-MoE work and under
an unrealistically favorable `A=8`, `U=1`. This promotes the shared-weight
batch component; uneven heterogeneous route batches are now the next gate.

PW-0037 closes that gate for the frozen PW-0016 route fixture. One Rust-owned
Metal command sequence gathers and executes nine exact source-FP8 experts,
applies source route weights, and scatter-adds all 64 real placements. Seven
experts have batch eight, one has five, and one has three; fixed batch-eight
execution exposes 12.5% padding rather than hiding it.

Two whole-MoE medians are 16.145 and 16.158 ms, giving a 10.539 routed-only
accepted-TPS diagnostic at this fixture's `A=8`, `U=1.125`. Complete output is
`1.71e-6` relative L2 from independent Torch source FP8. This replaces
PW-0016's faster 9.83 ms affine-INT4 number as the target-faithful component
cost; the older path's 17.02% error keeps it modified/conditional.

The first native attempt also exposed a real artifact fact: an expert weight
and its scale may live in different selected shard artifacts. Six independent
tensor authorities per expert are now explicit and fail closed. Routing remains
fixture-static; native noaux-tc selection is the next causal boundary.

PW-0038 crosses that boundary for the exact layer-43 fixture. A Rust-owned
Metal F32 `256×4,096` projection shares each weight across eight positions,
then Rust applies sigmoid, correction-biased top eight, uncorrected-score
gather, and normalization. Every selected expert set equals the independent
Torch result and route-weight error is at most `1.49e-8`; repeated canonical
route artifacts are byte-identical.

Two warm medians are 0.3246 and 0.3479 ms, passing the predeclared 1 ms router
gate with substantial headroom. The prior belief that route IDs and weights
must remain frozen runtime inputs is superseded for this fixture. The remaining
gap is composition, not router semantics: the new decision authority must now
causally feed PW-0037's gather/expert/scatter path in one timed command before a
dynamic routed-layer claim is warranted.

PW-0039 completes that composition. Every timed request dispatches the exact
router, derives the heterogeneous schedule in Rust, rewrites gather and routing
buffers from those decisions, and then executes the nine source-FP8 experts and
weighted scatter. The output remains `1.71e-6` relative L2 from independent
Torch and repeats byte-identically.

Two integrated medians average 17.1149 ms, only 0.9636 ms (5.97%) above
PW-0037's frozen-schedule control. The corresponding fixed-fixture routed-only
diagnostic is 9.945 TPS at `A=8`, `U=1.125`. The earlier limitation that native
router output did not drive expert execution is superseded. Representative
decode routes, a complete base transformer layer, and endpoint TPS remain
unmeasured; this result promotes the causal component, not those extrapolations.

PW-0040 falsifies expert-union phase parallelism as the missing performance
mechanism. Despite exact expert-major indexing and byte-identical complete
output, two paired candidate medians average 17.8157 ms versus 17.1049 ms for
PW-0039 control—a 4.16% slowdown rather than the required 20% gain. Keeping
each expert's gate/up/SwiGLU/down work temporally local is better on this M1
fixture than dispatching each phase across the whole union.

The diagnostic implementation also duplicates packed and serial buffers,
raising reported Metal buffers from 232.5 MB to 463.2 MB and peak process
footprint to 903.8 MB. Deduplicating those buffers would improve embodiment but
would not rescue the failed timing gate. The union-parallel schedule is
rejected; the PW-0039 schedule remains the executable default.

PW-0041 also rejects exact-F32 hot-cache matrix execution as the missing
backend. The expanded MLX path is faithful (`1.77e-6` relative L2), but paired
medians average 21.6533 ms versus 17.2763 ms control, a 25.33% slowdown. Batch
eight does not amortize the tuned matrix path enough to offset F32 traffic and
dynamic scheduling.

The embodiment trade is unfavorable too: 226.5 MB of exact selected source
tensors become 906.0 MB of F32 experts, with 919.8 MB MLX peak allocation and a
6.59 s cold install. This kills the planned C++ bridge branch for F32 expert
matmul on M1. Direct source-FP8 remains the promoted representation and backend.

PW-0042 rejects the narrower hypothesis that separate small gate/up dispatches
cause the direct-FP8 deficit. Fusing them into one 4,096-row dispatch remains
byte-exact, but paired medians average 17.4116 ms versus 17.0549 ms control, a
2.09% slowdown. Kernel-launch count and projection-grid size are not the
decisive bottleneck at this workload.

PW-0040 through PW-0042 jointly kill three superficially plausible ways to
make the same arithmetic more matrix-like. Further performance work should
change the inner FP8 reduction/data path or an explicitly validated fidelity
mechanism, not continue reshuffling identical projections.

PW-0043 tests the available M1 SIMD-group matrix unit rather than another grid
reordering. It improves numerical parity to `2.08e-7` relative L2, but paired
medians average 22.4906 ms versus 17.0986 ms control, a 31.54% slowdown. The
exact 8×8 tile requires 512 decode/synchronization steps along K for each
projection; that coordination cost dominates its faster accumulation.

The exact tile design is rejected, not the general existence of matrix units.
A future cooperative TensorOps path with wider tiles or a genuinely reused
decoded-tile cache would be a different mechanism. On the present M1/macOS
substrate, PW-0039 remains the best faithful native MoE implementation.

PW-0049 closes the next causal boundary: a frozen production-width layer-43
input now flows through learned source-FP8 QKV, source-faithful SWA,
post-attention residual/norm, native noaux-tc routing, all selected source-FP8
experts, weighted scatter, and the final residual under one Rust authority.
Two independent final-code processes are byte-identical and pass the final
gate at `3.64e-7` relative L2 and `2.384e-6` maximum absolute error.

The experiment supersedes two assumptions. First, sequential FP32 RMS
reduction was not numerically stable enough at this boundary; an f64 reduction
with the prescribed FP32 variance preserves semantics and materially improves
parity. Second, PW-0039's nine-expert `U=1.125` fixture is not representative:
the seeded real attention path selects 56 experts for eight positions,
`U=7.0`, with a healthy `7.41e-4` top-k margin.

The correctness embodiment expands only that selected source-FP8 union and
uses Accelerate SGEMM. Its routed MoE medians are 231.154 and 227.888 ms with
about 7.06 GB resident state; QKV is another 272--274 ms. This is decisive
evidence against treating expanded F32 as a performance default. PW-0049 is
the oracle and trace source for the predeclared topology/embodiment jumps, not
an endpoint TPS result.

PW-0050 run 001 exposed a checkpoint-layout exception before executing its
first full-model projection. Every full-attention fused QKV weight is
`[13568,4096]`, but its scale grid is `[108,32]`, not the generic `[106,32]`.
The mismatch occurs on exactly the nine full-attention layers. The 108 scale
rows preserve fused semantic segments: 96 Q block rows, two rows for each of
four 192-wide K heads, and one row for each of four 128-wide V heads. Generic
row-block validation must remain strict; full QKV requires a separately named,
head-aware scale mapping. The first endpoint attempt failed closed before
substantial memory growth and is preserved rather than omitted.

PW-0050 run 002 validates the shared-host safety gate and exposes a second
embodiment boundary. The repaired endpoint reached layer 24, then stopped on
the fixed 8 GiB process-footprint limit. System memory still reported 86% free,
swap had not grown from its 1,845.44 MiB baseline, and VM throttling remained
zero, so global pressure alone would have missed the accumulating process
residency. Dropping expanded matrices and calling malloc pressure relief was
insufficient because clean pages from the 17 persistent checkpoint mappings
remained resident. The next test must explicitly release those file-backed
pages between phases; it must not raise the 8 GiB/4 GiB footprint limits or
reinterpret the stopped walk as endpoint evidence.

PW-0050 run 003 proves that checkpoint-page release works but that a decoder
layer is still too coarse a resource-lifetime boundary on this 16 GiB host.
After layer 23, whole-mapping `MADV_DONTNEED` and malloc relief reduced physical
footprint to 650,410,688 bytes, while the historical peak had already reached
8,655,618,048 bytes—62.6 MiB over the fixed 8 GiB stop. Global memory remained
healthy and swap did not grow. The belief that layer-boundary release alone is
sufficient is therefore superseded: decoded matrix storage, allocator slack,
and newly faulted source pages must be released after each matrix operation so
their residency cannot accumulate within a layer.

PW-0050 run 004 supersedes the narrower belief that issuing the same
`MADV_DONTNEED` hint more frequently would force bounded mapped-file residency.
Matrix-boundary hints delayed the stop to 463 seconds but clean mapped pages
still accumulated until peak residency reached 8,723,333,120 bytes. The phase
cleanup then reduced current footprint to 365,902,912 bytes. Darwin documents
`MADV_DONTNEED` only as a near-term access expectation, whereas
`msync(MS_INVALIDATE)` explicitly invalidates cached mapped data. The next
candidate must test that stronger primitive under the same limits; frequency
alone is not the mechanism.

PW-0050 runs 005 and 006 establish the first complete, bounded native text
walk. `msync(MS_INVALIDATE)` before `MADV_DONTNEED` changes mapped-file release
from a reclaim hint into an effective phase boundary on this Darwin host. Two
clean processes executed all 48 layers twice with retained K/V, produced the
same `[122046,13]` (`瀛.`) output, and had an identical normalized semantic
trace hash. Complete wall times were 288.914 and 287.776 seconds; the resulting
~0.00694 accepted TPS is a slow correctness diagnostic, not a performance
default. Peak residency remained about 4.02 GiB, phase footprint below 3 GiB,
and host safety signals stayed clean.

The causal and embodiment boundaries are now real, but whole-model semantic
parity is not yet established. The surprising `瀛.` continuation means
determinism and component fixtures cannot substitute for accumulated hosted
logit comparison. `5345aa6` is promoted only as the M2 walking foundation;
target-faithful labeling remains conditional on an identical-prefix hosted
reference gate.

PW-0051 rejects OpenRouter's legacy raw-completions surface as the cheap
identical-prefix bridge for MiMo on Parasail. Although OpenRouter accepted a
standard `prompt` request at `/api/v1/completions`, its transformed upstream
body contained chat `messages` and omitted `prompt` while still calling
Parasail's completions endpoint. Parasail returned HTTP 400 before inference.
This is reproducible API-boundary evidence, not a behavioral model result. The
hosted comparison must use the supported chat surface and local native
multi-token prefill over the frozen checkpoint template.

PW-0052 freezes the first directly usable whole-model comparison for the M2
endpoint. Pinned Parasail, reasoning disabled, returned `Hello!` with exactly
20 logprob alternatives at both positions. The request consumed 27 prompt
tokens, so a single causal batched prefill is the appropriate local mechanism;
27 serial whole-model walks would add no semantic authority and would be an
unnecessary storage pass. Hosted capture success establishes the answer key,
not local parity.

PW-0052 local run 001 decisively rejects the direct-F32 whole-model numerical
mode. At the identical frozen 27-token chat prefix, local greedy output `.3`
disagreed with hosted `Hello!` at both positions. The hosted chosen-token
logprob errors were 13.5370 and 8.0002 nats, so this is not a near-tie or a
readability judgment. Causal cache lengths, full layer execution, and shared
host safety all passed.

The next semantic repair is now source-directed rather than speculative. The
checkpoint's FP8 config declares `activation_scheme: dynamic` with 128x128
weight blocks. The DeepSeek weight-format authority and compressed-tensors
scheme define that combination as dynamic per-token-per-128-channel activation
quantization. PW-0050/PW-0052 instead dequantized weights and multiplied raw
F32 activations. Component F32 fixtures validated that diagnostic arithmetic,
but they could not establish the omitted production activation semantics across
48 layers.

PW-0053 implements that declared dynamic activation semantic exactly against a
PyTorch 2.13.0 byte-level E4M3FN fixture, but rejects its omission as the
primary cause of the hosted mismatch. The frozen chat continuation changed
from `.3` to `. -`; hosted-token logprob error worsened from 13.5370 to 13.7936
nats at the first position and improved from 8.0002 to 7.4905 at the second,
with top-1 agreement still 0/2. The quantizer remains a correctness repair,
not a successful parity repair. The next source-directed semantic gap is BF16
execution-boundary rounding: existing component gates deliberately validated
F32 diagnostic arithmetic and therefore cannot authorize 48-layer accumulated
behavior for a BF16 model.

PW-0054 realizes the pinned model's explicit BF16 tensor boundaries and keeps
the F32 router/normalization/softmax internals distinct. Exact PyTorch
conversion and causal-attention fixtures pass, but the hosted repair is mixed:
the first chosen-token error improves from 13.7936 to 12.8016 nats while the
second worsens from 7.4905 to 7.8176, with output `.3` and top-1 agreement 0/2.
BF16 boundary omission is therefore superseded as the primary explanation,
while the source-authorized casts remain a correctness repair. Further full
walks now require a line-by-line source/runtime semantic discrepancy, not an
untethered numerical tweak.

PW-0055 corrects another pinned-source detail: text RoPE performs two BF16
multiplications and a BF16 addition, rather than one combined F32 expression
followed by a cast. Exact PyTorch fixtures pass and the chat output changes
materially to ` a.`, but hosted movement is again mixed: first-token error
improves from 12.8016 to 12.5231 nats while second-token error worsens from
7.8176 to 9.5774. The staging remains a correctness repair and is rejected as
the primary mismatch. Whole-model output sensitivity now makes further blind
full walks low-value; the next gate must localize the first real layer-0
intermediate divergence against independent PyTorch BF16/dynamic-FP8
semantics.

PW-0056 replaces output-only speculation with a real 27-position layer-local
trace. Embedding, RMSNorm, dynamic-FP8 fused QKV, partial RoPE, scaled V,
attention, BF16 projection, both residuals, and dense SwiGLU can now be
compared at named, hash-bound boundaries under the shared-host contract. The
gate caught both an attention shape-schema error and an incomplete oracle
operation: the first Python trace omitted the pinned BF16 max-subtraction
before F32 softmax. Those failures were preserved rather than normalized away.

PW-0057 uses Accelerate vForce exponential evaluation and the corrected source
operation order. Final layer-0 comparison 005 has maximum relative L2
`2.85e-6`, maximum absolute error `7.63e-6`, minimum BF16 equality 99.9959%,
and bit-exact probabilities through final residual. Dense layer 0 is
provisionally cleared; the belief that a layer-0 projection/layout error is the
primary hosted mismatch is superseded. The next localization target is routed
layer 1 with learned SWA sink and dynamic expert selection, not another blind
whole-model walk.

PW-0058 clears that next structural boundary. The production Rust path
causally recomputes dense layer 0, then matches the independent oracle
bit-for-bit through routed layer-1 SWA QKV, theta-10,000 RoPE, learned sinks,
attention, projection, residual, post-normalization, and F32 router logits.
All 27 eight-expert sets are exact, with maximum per-expert route-weight error
`2.54e-8` against the `5e-7` gate. The belief that SWA attention or noaux-tc
routing is the first hosted-divergence mechanism is superseded. The next rung
is execution of only the causally selected real experts; the 4.245-second,
686 MB Rust trace is diagnostic and changes no throughput-model constant.

PW-0059 clears the complete first routed decoder layer. Across 28 independently
selected experts and 216 placements, gate/up, BF16 SwiGLU, and down projections
are bit-exact. The weighted scatter differs by only `2.09e-8` relative L2 and
one BF16 quantum maximum because independent route weights differ at `2.54e-8`;
the final layer state is bit-exact. Dynamic gather, source-FP8 expert execution,
scatter, and routed residual are superseded as primary hosted-mismatch causes.
Together dense layer 0 and routed layer 1 exercise every model semantic
category, so another isolated layer is lower value than a serial layer-final
whole-model oracle/native trace. The 15.199-second Rust diagnostic changes no
throughput constant.

PW-0060 performs that serial trace and localizes the first accumulated failure
to layer 2. Embedding and complete layers 0–1 are bit-exact; layer 2 retains
exact expert sets but reaches `1.78e-6` maximum route-weight error and `0.0625`
maximum final-state error at 99.452% BF16 equality. Later states and routes
compound from there, so output-only whole-model repairs were addressing a
downstream symptom. A separate oracle failure also exposed real cross-shard
expert weight/scale placement at layer 43; tensor authority must resolve weight
and scale independently rather than requiring co-location. Both 48-layer walks
passed the shared-host contract with no swap growth or throttling. The next
diagnostic is layer-2 substage localization from the bit-exact layer-1 final,
not another full walk. These cold correctness walls change no throughput-model
constant.

PW-0061 identifies the first layer-2 operation difference as softmax
denominator reduction, not exponential evaluation or routing. Centered scores
are exact; one of 25,920 BF16 probabilities differs when Rust accumulates the
19-value row forward. Reverse F32 accumulation matches PyTorch exactly on that
row and preserves exact probability payloads on the complete real layer-0,
layer-1, and layer-2 corpora. The single probability quantum amplifies into
router-weight and final-state failure, so sigmoid and scatter changes would be
downstream repairs. Gate denominator order directly before repeating layer 2.

PW-0062 confirms reverse denominator accumulation removes that first mismatch:
layer 2 becomes bit-exact through router logits and every selected expert
tensor. The remaining nine final BF16 differences originate in one-ULP router
sigmoid differences; route weights differ only `2.22e-8`, but the strict final
gate correctly preserves the discrepancy. Denominator order is promoted as a
correctness repair; vectorized PyTorch sigmoid is the next isolated boundary.

PW-0063 clears layer 2 completely and supersedes the narrower belief that
matching sigmoid values alone would do so. SLEEF U10 makes all 6,912 router
scores exact, but PyTorch's unsorted top-k output order also controls the
eight-value vector reduction used to normalize route weights. Reproducing its
libc++ `std::nth_element` order and four-lane sum makes every route-weight F32
bit and all 21 captured tensors through final residual exact. The bounded run
peaked at 729 MB with 81% system-free memory, no swap growth, no throttling,
and all protected services healthy. This is a correctness repair, not a
throughput result; the next cheapest falsification is a repeated full-prefix
trace against the existing frozen oracle.

PW-0064 advances the accumulated exact frontier through layer 3. Layer 4 is
now the first failing boundary: expert selections remain exact, but 16 of 216
route-weight F32 values differ by at most `5.93e-5`, and the layer-final BF16
state has 99.1093% equality with `0.0625` maximum error. Later divergence is
downstream. The full walk also validates the strengthened shared-host
contract: repeated phase cleanup returned residency near 152 MB, the LM head
peaked at 3.945 GB and ended at 2.687 GB, free memory stayed at 81%, and swap,
throttling, and protected-service health remained clean. The next diagnostic
is a layer-4 substage trace from exact layer 3, not another full walk.

PW-0065 localizes layer 4's first actual difference to two attention
probabilities after bit-exact centered scores. The earlier vForce/reverse-sum
path was exact on layers 0–3 by corpus coincidence, not because it reproduced
the PyTorch CPU kernel. PyTorch uses SLEEF vector exponential, four-lane ARM
accumulation, horizontal reduction, reciprocal, and multiplication; replaying
that order makes all complete real layer-0 through layer-4 probability corpora
exact. The first formal BF16 gate failure is downstream at post-attention
RMSNorm. Gate the true softmax operation order on both failing rows before
repeating layer 4; routing and expert changes remain unjustified.

PW-0066 promotes the pinned PyTorch ARM softmax order. SLEEF exponentials,
four-lane accumulation, horizontal reduction, one reciprocal, and
multiplication reproduce all 101,952 real BF16 probabilities across layers 0,
1, 2, and 4. Layer 4 is exact again from its post-attention residual through
final state, with bit-exact routes and weights. A one-value attention-output
delta and five projection values remain below `0.000977` and disappear at the
BF16 residual boundary; preserve them as a separate accumulation diagnostic,
but they do not fail the layer or justify holding the accumulated frontier at
layer 3. The next cheap discriminator is another frozen full-prefix replay.

PW-0067 exposes a fail-fast gap in the scalar SLEEF port before producing a
new frontier result. The polynomial was correct on normal outputs, but its
single exponent-bit construction cannot represent negative `q + 127` and
panics where SLEEF's two-stage `vldexp2` deliberately creates subnormals. The
12-minute run remained memory-safe and produced no manifest. Gate subnormal
exponential and saturated sigmoid values before repeating; this failure says
nothing about attainable throughput or the next divergent layer.

PW-0068 implements SLEEF's two-factor `vldexp2` and clears exact PyTorch bits
through minimum subnormal, underflow, overflow, and saturated sigmoid cases.
The full walk then completes safely and advances the exact accumulated
frontier through layer 6, with exact route order and every route-weight F32 bit
for layers 1–6. Layer 7 is the first failure: 12 BF16 state values differ by at
most `0.0625`, and eight route weights differ by at most `1.70e-6`; expert sets
remain exact. The next diagnostic is a layer-7 substage trace from exact layer
6, not another full walk.

PW-0069 localizes layer 7 to one BF16 attention score, not routing or expert
execution. From an exact layer-6 input, PyTorch and Rust remain bit-exact
through QKV, RoPE, values, and sinks; position 22/head 12/source 17 differs by
one BF16 quantum after the query/key dot and scale. It changes two probabilities
and survives the post-attention residual in 12 of 110,592 values, reproducing
PW-0068's final error and route-weight envelope. The Rust trace safely peaked
at 723 MB RSS, returned to 125 MB, retained 81% free memory, and caused no swap
growth, throttling, or service loss. Isolate PyTorch's aarch64 BF16 dot-product
accumulation order on that exact pair before any repair or full walk.

PW-0070 proves the layer-7 score mismatch came from reduction topology. The
old forward and PyTorch-source four-lane F32 sums differ by only two ULPs but
round to adjacent BF16 values. A hash-bound real fixture gates that boundary,
and the promoted four-lane reduction makes all 21 layer-7 captures bit-exact;
expert selection/order is exact and route-weight serialization differs by only
`7.43e-9`. The 118.330-second replay peaked at 720 MB RSS, returned to 124 MB,
retained 81% free memory, and caused no swap growth, throttling, or service
loss. The exact accumulated frontier is now through layer 7; use one frozen
full-prefix replay to find the next boundary.

PW-0071 advances the bit-exact accumulated frontier through layer 10. Layer 11
is the first actual divergence—only five BF16 values, `8.83e-7` relative L2,
and `0.015625` maximum error—while layer 14 is the first layer-final threshold
failure after that error accumulates. Route weights remain within their strict
gate through layer 11 and first fail at layer 12; expert sets remain exact
through layer 18. Treating layer 14 as the causal boundary would skip the first
arithmetic defect, so localize layer 11 from exact layer 10. The first full walk
under normative Gate 8 peaked at 4.168 GB RSS in the LM head, ended at 2.904 GB,
retained at least 80% free memory, and caused no swap growth, throttling, or
protected-service loss.

PW-0072 localizes layer 11 to one global-attention score. The first oracle
attempt usefully failed closed on an SWA-layout assumption; layer 11 instead
has 4 KV heads, a 13,568-row full-QKV layout, no sinks, and the 10M RoPE base.
With that topology corrected, all captures through query/key/value are exact.
Position 22/head 3/source 16 differs by one BF16 quantum because PyTorch's
specialized reduced-precision GEMV dot uses eight vector accumulators and a
pairwise reduction tree, whereas the PW-0070 repair models the simpler
four-lane fallback. The one score reproduces all five layer-final differences.
Gate the specialized vector reduction on this pair before another arithmetic
change or full walk.

PW-0073 promotes PyTorch's specialized global-attention BF16 vector dot. Eight
four-element F32 accumulators, pairwise reduction, ARM horizontal addition,
and the source's vector/scalar tails clear every layer-11 capture exactly; route
serialization differs only `7.57e-9`. A focused width-192 fixture alone was not
enough: the full suite caught the omitted tail path on a tiny global-attention
case before any real replay, and the repair was withheld until that path also
matched. The final 180.887-second trace peaked at 749 MB RSS, retained 81% free
memory, and caused no swap growth, throttling, or service loss. The exact
accumulated frontier is now through layer 11.

PW-0074 advances the bit-exact accumulated frontier through layer 12. Layer 13
is the first actual divergence—21 of 110,592 BF16 values, `1.63e-6` relative
L2, and `0.015625` maximum error—while layer 14 remains merely the first formal
layer-final failure. Route weights first exceed their strict gate at layer 13;
expert sets remain exact through layer 18. The safe full walk peaked at 3.838
GB RSS in the LM head, ended at 2.710 GB, retained at least 77% free memory,
and caused no swap growth, throttling, or protected-service loss. Localize
layer 13 from the frozen exact layer-12 input rather than changing downstream
routing or repeating another full walk.

PW-0075 localizes layer 13 to attention-value reduction after every centered
score and probability matches. Exactly one of 221,184 attention values differs:
the forward F32 sum lands on a BF16 tie, while PyTorch's specialized vector-tail
topology lands two F32 ULPs lower and rounds to the oracle value. That one
quantum causes the layer's downstream router-weight and final-state errors;
expert sets remain exact. The 213.801-second production trace peaked at 750 MB
RSS, returned to 132 MB, retained at least 79% free memory, and caused no swap
growth, throttling, or service loss. Gate the real 25-element pair before
reusing the specialized dot helper for attention values.

PW-0076 promotes the source-pinned specialized vector-tail topology for BF16
attention-value dots. A hash-bound 25-element fixture discriminates its result
from forward accumulation; the full suite additionally caught and protected
the SWA sink/value-row boundary. The corrected real layer-13 replay makes all
21 captures bit-exact, preserves exact expert sets/order, and holds route-weight
serialization error to `2.60e-8`. It safely peaked at 754 MB RSS, returned to
133 MB, retained at least 82% free memory, and caused no swap growth,
throttling, or service loss. The accumulated exact frontier is ready for a
full-prefix replay beyond layer 13; this correctness result changes no
throughput constant.

PW-0077 confirms the accumulated state is bit-exact through layer 13. Layer 14
is both the first actual divergence and first formal layer-final failure: 396
BF16 values differ, equality remains 99.6419%, relative L2 is `9.82e-6`, and
maximum error is `0.0625`. Route weights first exceed their strict gate at
layer 14; expert sets remain exact through layer 18. The 797.596-second safe
walk peaked at 3.946 GB RSS in the LM head, ended at 2.681 GB, retained at
least 82% free memory, and caused no swap growth, throttling, or service loss.
Localize layer 14 from exact layer 13 rather than changing downstream behavior.

PW-0078 localizes layer 14 to post-attention RMSNorm after the complete
attention residual matches bit-for-bit. PyTorch's contiguous-inner cascade sum
produces an F32 variance one ULP below the prior high-precision reduction on
position 1; its inverse differs by two ULPs and tips 41 weighted outputs across
BF16 boundaries. Router and expert differences are downstream. The 230.205-
second Rust trace peaked at 747 MB RSS, ended at 376 MB, retained at least 82%
free memory, and caused no swap growth, throttling, or service loss. Gate the
real row and pinned cascade topology before changing RMSNorm production code.

PW-0079 promotes PyTorch's contiguous-inner F32 cascade for RMS variance. The
hash-bound 4,096-value row distinguishes it from the prior F64 reduction and
gates the exact variance and inverse bits. The repaired layer-14 replay makes
all 21 captures bit-exact, preserves exact expert sets/order, and holds route-
weight serialization error to `1.70e-8`. It safely peaked at 715 MB RSS,
returned to 137 MB, retained at least 82% free memory, and caused no swap
growth, throttling, or service loss. A full-prefix replay can now advance the
accumulated frontier beyond layer 14; no throughput constant changes.

PW-0080 advances the accumulated bit-exact frontier through layer 18, showing
that the RMS cascade repair clears four additional layers without intervention.
Layer 19 is both the first actual and formal failure: 190 BF16 values differ,
equality is 99.8282%, relative L2 is `3.07e-5`, and maximum error is `0.25`.
Route weights first fail at layer 19; expert sets remain exact through layer 24.
The 799.595-second safe walk peaked at 4.169 GB RSS in the LM head, ended at
2.904 GB, retained at least 82% free memory, and caused no swap growth,
throttling, or service loss. Localize layer 19 from exact layer 18.

PW-0081 localizes layer 19 to one SWA attention score. The discriminating
width-192 pair requires PyTorch's specialized eight-vector BF16 dot topology;
forward and four-lane sums round to the adjacent BF16 value. The earlier
layer-7 pair did not distinguish four-lane from specialized at BF16, so its
narrower topology inference is superseded, not erased. One score changes two
probabilities and ultimately 190 final values; routing differences are
downstream. The 303.484-second safe trace peaked at 709 MB RSS, returned to
139 MB, retained at least 82% free memory, and caused no swap growth,
throttling, or service loss. Gate the real pair before unifying score dots.

PW-0082 promotes the pinned specialized eight-vector topology for every BF16
attention score dot. A new layer-19 SWA fixture genuinely discriminates it
from four-lane reduction, while the preserved PW-0070 pair confirms both
topologies happen to round to its oracle BF16 value. The repaired layer-19
replay makes all 21 captures bit-exact, preserves exact expert sets/order, and
holds route-weight serialization error to `2.21e-8`. It safely peaked at 747
MB RSS, returned to 138 MB, retained at least 82% free memory, and caused no
swap growth, throttling, or service loss. The accumulated frontier is ready
for replay beyond layer 19; no throughput constant changes.

PW-0083 advances the accumulated bit-exact frontier through layer 28, showing
that the unified score-dot repair clears nine additional layers without
intervention. Layer 29 is both the first actual and formal failure: only 20 of
110,592 BF16 values differ, equality is 99.9819%, relative L2 is `6.25e-6`,
and maximum error is `0.0625`. Route weights first fail their strict gate at
layer 29, while expert sets remain exact through layer 46. The 800.724-second
safe walk peaked at 4.171 GB RSS in the LM head, ended at 2.909 GB, retained
at least 72% free memory, reduced swap use, and caused no throttling or
protected-service loss. Localize layer 29 from exact layer 28.

PW-0084 localizes layer 29 to one softmax probability after every incoming,
RMSNorm, QKV, RoPE, value, and centered-score bit matches. Position 22, head
15, source 20 rounds to adjacent BF16 probabilities in PyTorch and Rust; that
single quantum causes nine attention-value differences and ultimately all 20
PW-0083 final-state differences. Router and expert arithmetic are downstream,
and expert sets remain exact. The 466.678-second safe Rust trace peaked at 750
MB RSS, returned to 149 MB, retained 82% free memory, and caused no swap
growth, throttling, or service loss. Freeze the 23-value row and discriminate
exponential, denominator, and normalization order before changing softmax.

PW-0085 corrects the earlier PW-0066 interpretation of ARM `vaddvq_f32`:
four accumulated F32 lanes reduce low against high as `(lane0 + lane2) +
(lane1 + lane3)`, not adjacent pairs. The layer-29 fixture proves all SLEEF
exponentials were already exact and discriminates the denominators by one ULP.
The repaired layer replay makes all 21 captures bit-exact, preserves exact
expert sets/order, and holds route-weight error to `1.93e-8`. It safely peaked
at 746 MB RSS, returned to 144 MB, retained at least 83% free memory, and
caused no swap growth, throttling, or service loss. The exact accumulated
frontier is now through layer 29; run one frozen full-prefix replay next.

PW-0086 advances the accumulated bit-exact frontier through layer 33. Layer 34
is the first actual divergence—only 6 of 110,592 BF16 values, `4.43e-7`
relative L2, and `0.0078125` maximum error—while layer 36 is merely the first
formal final-state failure after propagation. Route weights first fail at
layer 34; expert sets remain exact through layer 43. The 781.393-second safe
walk peaked at 3.942 GB RSS in the LM head, ended at 2.674 GB, retained 83%
free memory, and caused no swap growth, throttling, or protected-service loss.
Localize layer 34 from exact layer 33 rather than skipping to layer 36.

PW-0087 localizes layer 34 to one attention value after scores and
probabilities match bit-for-bit. PyTorch's vector-by-25×128 BF16 matrix path
uses generic four-part GEMM reduction and lands one F32 ULP below a BF16 tie;
Rust's specialized contiguous dot lands on the tie and rounds upward. The
PW-0076 pair did not distinguish those topologies, so its narrower inference
is superseded. That single quantum reproduces all six PW-0086 final-state
differences; routing is downstream. The 543.499-second safe trace peaked at
720 MB RSS, returned to 154 MB, retained 83% free memory, and caused no swap
growth, throttling, or service loss. Gate the discriminating pair before
changing attention value-by-matrix reduction.

PW-0088 promotes generic four-part BF16 GEMM reduction for attention
value-by-matrix accumulation while retaining specialized contiguous dots for
scores. The discriminating layer-34 fixture selects generic reduction; the
preserved PW-0076 fixture proves its earlier pair could not distinguish the
two. The repaired layer replay makes all 21 captures bit-exact, preserves
exact expert sets/order, and holds route-weight error to `2.88e-8`. It safely
peaked at 738 MB RSS, returned to 155 MB, retained 83% free memory, reduced
swap use, and caused no throttling or service loss. The exact accumulated
frontier is now through layer 34; run one frozen full-prefix replay next.

PW-0089 closes the complete accumulated transformer prefix: embedding, all 48
layers, final RMSNorm, route weights, and expert sets/order are bit-exact. The
only remaining local mismatch is the LM-head projection—45 of 152,576 F32
logits, 99.9705% exact, `5.25e-5` relative L2, and `0.03125` maximum error.
Both captured hosted-chosen token logits are exact, but the full vector still
fails its unchanged gate. The 799.549-second safe walk peaked at 3.938 GB RSS,
ended at 2.680 GB, retained at least 81% free memory, and caused no swap
growth, throttling, or protected-service loss. Freeze the exact final-norm
input and localize LM-head arithmetic; no transformer change is justified.

PW-0090 clears every formal full-prefix gate but falsifies its own exact
LM-head operation premise. A specialized BF16 dot removes 25 of 45 logit
differences, leaving 20 values within tolerance; the oracle actually widens
BF16 inputs and weights to F32, performs the full matrix multiply, then rounds
to BF16. Complete wall time remains 799.776 seconds, so projecting only the
last prompt row is not yet a full-path performance result. The safe run peaked
at 3.925 GB RSS, ended at 2.662 GB, retained at least 81% free memory, and
caused no swap growth, throttling, or service loss. Reject the specialized-dot
LM-head authority and test the source-faithful one-row F32 matrix path without
relaxing the now-cleared formal gates.

PW-0091 establishes complete local full-prefix parity. Applying the existing
F32 matrix backend to only the authoritative last normalized row reproduces
the frozen oracle bit-for-bit for embedding, all 48 layers, final RMSNorm, all
152,576 logits, route weights, and expert sets/order. Matrix shape—not a
missing specialized arithmetic kernel—was the final boundary. The 798.639-
second run offers no promotable speed gain, but safely peaks at 3.928 GB RSS,
ends at 2.667 GB, retains at least 81% free memory, and causes no swap growth,
throttling, or protected-service loss. The correctness frontier now covers the
complete prefill and next-token logit path; move to generation semantics and
end-to-end cold/warm throughput rather than further layer localization.

PW-0092 establishes the first repaired real incremental text walk. Two clean
processes deterministically emit `[264, 13]` (` a.`), with byte-identical full
logits and route traces, and retain every layer cache from length 27 to 28
while the second step consumes only one token. Step-one logits and routing are
exactly PW-0091's source-checkpoint authority. The frozen hosted service instead
emits `[9707, 0]` (`Hello!`); its first-token logprob differs by 12.5385 nats,
so source parity does not imply hidden hosted-serving parity and local
semantics must not be tuned to manufacture it. The retained-cache token still
takes 158.5--158.6 seconds: 151.6 seconds across layers and roughly 7.0 seconds
outside them. Exact route/header replay partitions 17,207,905,152 logical
source bytes (16.026 GiB), 1,179 FP8 matrix expansions, and 376 expert
executions into that one-token step; the larger 27-token prefill accounts for
the remainder of the 84.18 GB two-step ledger. The repeated weight load and
FP8-to-F32 expansion path is now the primary embodiment bottleneck. Both runs
safely peak near 4.37 GB RSS, finish near 3.1 GB, retain at least 78%
memory-pressure headroom, and cause no swap growth, throttling, or
protected-service loss. Verify incremental state independently, then profile
and compress physical weight work before claiming TPS.

PW-0093 rejects byte identity between a 28-row PyTorch whole-sequence pass and
PW-0092's one-row retained-cache Rust step. The exact token prefix is proven,
but route-weight drift at the appended position starts at layer 1, unsorted
expert order first changes at layer 3, the first expert-set change occurs at
layer 11, and final logits have only 8.3309% equality, `0.0246957` relative L2,
and `0.5` maximum error while preserving greedy token 13. This does not yet
prove bad K/V: matrix backends can use row-count-dependent reduction
topologies, and earlier exactness covered matching 27-row shapes. The safe
702.630-second oracle peaked at 3.879 GB RSS, ended at 219 MB, retained at
least 63% memory-pressure headroom, and caused no swap growth, throttling, or
service loss. Compare PyTorch-28 with Rust-28, then Rust-28's last row with
Rust 27+1; preserve the rejected direct equivalence and do not weaken it.

PW-0094 proves the 28-row source path itself is exact: PyTorch-28 and Rust-28
match bit-for-bit for embedding, all 48 layer finals, final norm, every route,
and all 152,576 logits. The Rust-28 versus Rust 27+1 mismatch is therefore
real, beginning as route-weight drift at layer 1 and reaching expert-set drift
at layer 11, but still confounds retained K/V with one-row matrix reduction
topology. The 785.198-second Rust trace moved 67.099 GB logically, peaked at
4.154 GB RSS, ended at 2.890 GB, retained at least 74% memory-pressure
headroom, and caused no swap growth, throttling, or service loss. A trace-only
schema now admits exactly the frozen prefix plus token 264 while the production
endpoint rejects it. Build an independent PyTorch 27+1 cached oracle next; do
not infer cache correctness from whole-sequence equivalence alone.

PW-0095 independently clears retained-cache semantics. A separate PyTorch
implementation performs the exact 27-token prefill, retains source K/V in all
48 layers, then propagates only token 264 at absolute position 27. All caches
reach 28 positions with the pinned nine-global/four-head and 39-SWA/eight-head
schedule. Its prefill and incremental expert sets/order exactly match Rust;
maximum route-weight errors are `2.97e-8` and `2.08e-7`. The complete one-row
logit vectors are byte-identical and hash to
`e86670ade50a8c02be5451f9233a65e6b982e80d09f8fd38b41c2d8e3ea2526a`,
choosing token 13.
Thus PW-0093's whole-sequence mismatch is row-count-dependent source matrix
arithmetic, not K/V corruption; equal-shape comparisons are the correct cache
gate. The 804.493-second oracle safely peaks at 4.170 GB RSS, ends at 421 MB,
retains at least 71% memory-pressure headroom, grows swap by less than 0.4 MB,
and causes no throttling or protected-service loss. Incremental correctness is
closed for the walking slice; profile and compress the 17.208 GB/1,179-expansion
one-token embodiment next.

PW-0096 profiles the two immutable PW-0092 endpoint reports without another
model run. The 158.52/158.61-second retained-cache tokens vary by only 0.059%.
Routed layers consume 149.40/149.53 seconds, 94.25% of complete token wall;
full and SWA routed layers both average about 3.18 seconds. Experts are 55.00%
of the 17.208 GB logical source path but cause 1,128 of 1,179 FP8 matrix
expansions (95.67%) through 376 executions. Layer zero is about 2.1 seconds and
the non-layer/LM-head remainder about 7.0 seconds. Promote repeated routed
expert FP8-to-F32 expansion/execution as the primary embodiment bottleneck.
Integrate the validated source-FP8 Metal expert path as an explicit candidate
before spending work on attention or K/V compression; no performance default
or accepted TPS changes yet.

PW-0097 validates the first source-faithful one-row Metal expert embodiment.
After a fail-closed attempt exposed a missing BF16 round between SiLU and its
`up` product, the repaired executor matches 4,094/4,096 widened-BF16 outputs,
with `1.78e-5` relative L2 and `2.98e-8` maximum error, in two byte-identical
processes. Median complete expert cost is 6.7615/6.7661 ms including dynamic
FP8 activation staging, all real tensor-buffer installations, three dispatches
and waits, BF16 boundaries, CPU SwiGLU, readback, and buffer destruction. This
is a 58.75--58.79x component gain over PW-0096's 397.5 ms/expert attribution,
while retaining at least 78% free memory, peaking below 68 MB RSS, causing no
swap growth or throttling, and preserving all protected services. Promote the
bounded source-FP8 Metal executor only as the next routed-layer candidate;
one expert under warm OS cache is not endpoint TPS or a performance default.

PW-0098 rejects direct generalization of PW-0097's 64-lane Metal reduction to
a complete routed row. Native source-exact routing is bit-exact and the bounded
eight-expert path runs in 55.90/55.87 ms—56.90x faster than PW-0096's CPU
routed-layer attribution—but final BF16 identity is only 92.2363% and relative
L2 `9.59e-4`. Seven experts individually match at least 4,092/4,096 BF16
values; expert 182 alone falls to 2,992/4,096 despite all six raw tensor ranges
matching the verified checkpoint byte-for-byte. The failure is accumulation-
topology sensitivity at BF16 boundaries, not routing, scatter, storage, or
memory pressure. The rejected run peaks at 253 MB, returns to 30 MB, retains
79% free memory, and causes no swap growth or throttling. Localize and repair
only uncertain expert-182 rows; do not promote the fast path or weaken gates.

PW-0099 repairs the PW-0098 numerical failure without expert- or row-specific
policy. Expert 182's decisive up value lies one F32 ULP from a BF16 midpoint;
the wrong neighbor changes a dynamic-FP8 SwiGLU group maximum and amplifies one
boundary error into 1,104 down differences. A fixed value-derived four-ULP
predicate source-reduces only uncertain rows: three gate, three up, and nine
down rows across the routed slice, decoding 172,032 bytes. Interleaved frozen
controls remain incorrect at 55.09/55.60 ms; repaired candidates run at
55.18/55.60 ms, produce identical bytes, and reach 99.9756% BF16 identity,
`5.26e-5` relative L2, and `2.98e-8` maximum error. Independent expert 32 also
passes with two down repairs. Gate 8 retains 79% free memory, peaks below 258
MB, returns below 31 MB, and records no swap growth, throttling, or protected-
service loss. Promote sparse boundary repair only as a complete-token candidate;
component success is not endpoint TPS or a default.

PW-0100 rejects complete-token promotion of PW-0099's sparse-repaired Metal
executor. The real retained-cache walk still chooses token 13 and has exact
routes at its first failed layer, but accumulated state crosses the unchanged
gate at layer 4: `0.001635` relative L2, `1.0` maximum error, and 97.876% BF16
identity. Final norm and logits also fail. The token takes 75.726 seconds—only
2.09x faster than the 158.5-second CPU control and far slower than the 20-second
candidate gate—despite 55 ms warm component results at layer 43. Thus two PW-
0099 assumptions are superseded: a fixed four-ULP midpoint repair does not
generalize through all accumulated layer distributions, and warm repeated
routed-row timing does not predict cold complete-token expert installation.
The failed walk safely retains 79% free memory, peaks at 4.311 GB RSS, releases
to 3.062 GB, and causes no swap growth, throttling, or protected-service loss.
Localize layer 4 and measure cold install/I/O separately before another full
walk; do not widen correctness thresholds or promote the endpoint.

PW-0101 falsifies the tempting explanation that PW-0100 merely needs a wider
BF16-midpoint threshold. On the exact source layer-4 MoE input, the bounded
Metal path reproduces PW-0100's layer-final failure exactly. Expert 245 gate
row 1798 lands precisely at midpoint `0x40808000` and is selected by the fixed
predicate, but the one-row Accelerate repair returns BF16 `0x40800000` while
the authoritative full 2,048-row projection returns `0x40810000`. Changing
only that gate value creates one SwiGLU mismatch and fans out to 233 down
mismatches (`0.001278` relative L2, max error `4.0`); restoring it makes the
entire 4,096-value down projection bit-exact. Thus sparse row selection is
sound on this failure, while row-count-dependent correction topology is not.
Do not widen the selector. A future repair must prove full-shape-equivalent
boundary decisions without silently paying full-projection work; this numerical
repair remains separate from PW-0100's 75.7-second physical bottleneck.

PW-0102 Phase A verifies the complete pinned DFlash draft payload rather than
trusting transfer metadata. The 2,936,121,080-byte file hashes to its locked LFS
identity `29e60c5d876e1c2e5f11b03244d52e2fe4a2f05c2c6f4c2d5aa15dd971ebc0d5`;
all five auxiliary artifacts and all 63 BF16 tensor names, shapes, and dtypes
also pass. The artifact contains five nonzero per-layer attention-sink tensors,
while its published Hugging Face class registers only the other 58 tensors and
ignores nested value scale. The earlier belief that Hugging Face also ignored
partial RoPE is superseded: Transformers 4.57.6 consumes the exported factor
`0.5`, creates 64-wide rotary factors, and the wrapper fails when applying them
to 128-wide Q/K heads in its first layer. Pinned SGLang `2fc5572`, the deployment
runtime named by Xiaomi's newer DFlash release, explicitly uses
`rotary_dim=head_dim`, as well as unscaled values and no sink. A Mac reference
adapter may therefore normalize only the partial-RoPE factor to `1.0`, must name
that SGLang-semantic adaptation in evidence, and must preserve the unmodified
HF failure. Treat the sink weights and value-scale config as
exported-but-unused; do not apply that label to partial RoPE. The immutable artifact audit
hashes to `e67b0106aa2c26a091f1fef0661a4ccc408389f2bc5d1bab9ed42e46a6e898c6`;
it retained at least 79% free memory, peaked at 222 MB RSS, returned below 151
MB physical footprint, and caused no swap growth, throttling, or service loss.

PW-0102 Phase B attempt `draft-001` preserves the unmodified published-HF
failure at the first attention layer. Its error is the expected 128-versus-64
rotary dimension mismatch; its failure manifest hashes to
`f43cba92b87b2d0c2d2b8603ac974df8d6ee6b898f209b0c001039b251e1b149`,
and it stopped before any full target walk. The last
safety boundary retained 78% free memory, used 286,560,832 bytes of physical
footprint, caused no swap growth or throttling, and retained every protected
service. A separately identified `draft-002` retry is warranted solely to
execute the already-pinned SGLang full-head semantics through the HF reference
adapter; it is not evidence from the unmodified HF wrapper.

PW-0102 Phase B then passes twice under that explicitly named adapter. The cold
manifest hashes to
`cfae209566f433933097e1b4ca97f25e4019dab33851f5f46b294c5ab7709959`
and the warm repeat to
`0094235cbee8a19138b812a1edc40420925a198180f5cf81e9c644d14b31d5c6`.
Both runs are byte-identical at all five draft layers, the final hidden capture,
the complete seven-row logits capture, and proposal IDs
`[264, 1773, 102092, 102092, 102092, 1773, 1773, 1773]`. Cold versus warm
physical reads (3,901,050,880 versus 26,480,640 bytes) explain the 65.14-second
versus 1.39-second draft-forward difference and reinforce that cached draft
latency is not an endpoint claim. Both runs pass every shared-host safety stop
and release below 282 MiB physical footprint. The one contracted target
verification walk is now authorized.

PW-0102 closes the pinned DFlash artifact/base-checkpoint trace after the one
authorized full target walk. The target first posterior token is 13 while the
draft first suffix token is 1773, so no draft suffix token is accepted and
formal `A=1` counts only the target anchor. The 47 routed layers install 878
layer-local unique experts across width eight: mean 18.680851 experts/layer,
`U=2.3351063829787235`, and `A/U=0.42824601366742593`. This is below even the
minimum routed-byte leverage gate of 1 and far below PW-0011's otherwise-free
INT4 requirement 7.548793. The earlier planning possibility that the published
draft might amortize base-target expert traffic is superseded for this pinned
trace. Its verified target pass moves 22,100,987,904 logical expert bytes and
29,844,290,432 total logical source bytes for one accepted anchor, before
charging draft work. Do not optimize or repeat this proposal as a Prismwing-50
path. A base-trained or materially different proposer would be a new branch,
not a reversal without new acceptance evidence.

The immutable Phase C manifest hashes to
`cb30738d5a79d7d85587a68b53f876a59101d5ca09bbc7c895daaf501954f4d3`.
It reproduces all PW-0091 prefill layer captures/routes and its full logit hash,
then produces target posterior IDs `[13, 15, 18, 481, 15, 481, 15, 15]`.
Post-prefill wall time is 272,841.507 ms, a single-trace diagnostic 0.003665
accepted token/s rather than endpoint TPS. Gate 8 passed all 103 boundaries:
minimum free memory 71%, peak RSS 4,044,210,176 bytes, maximum physical
footprint 203,508,736 bytes, at most 1 MiB transient swap growth, zero new
throttled pages, no protected-service loss, and 161,221,376 bytes physical
footprint after final buffer release.

PW-0103 rejects the pinned checkpoint's native MTP path on the same causal
trace without spending another target walk. Pinned SGLang confirms the exact
input transition: rotate the prompt IDs left, append target anchor 264, pair
those embeddings with the target hidden states before final norm, then run the
selected MTP layer. With PW-0091 layer-47 states, MTP layer zero proposes token
100730 while the independently proven target token is 13. The correct token
ranks 175th (logit 7.84375 versus 12.0625 top), so the failure is not a greedy
near-tie. The earlier possibility that the checkpoint-native draft would avoid
DFlash's cross-revision mismatch is superseded for this trace. Do not build the
three-layer MTP scheduler or authorize a target verifier until new semantic
evidence changes this first complete logit vector. The immutable manifest
hashes to
`65404539dc1b0f0e5b8cf0a0962b1b65fcd5e5fdcfe15ae2f1fd5ebdd49992a7`;
Gate 8 passed all 11 boundaries and released to 154,553,152 bytes physical
footprint.

PW-0104 rejects a 6--8 GiB exact expert cache as the primary throughput
mechanism on the authenticated PW-0091 causal trace. Its 27 positions produce
10,152 accesses to 2,353 distinct layer-local experts. At 8 GiB (341 equal-size
expert payloads), an offline Belady oracle with impossible knowledge of the
entire future reaches only 60.037431% hits, leaving 4,057 misses and
102,122,674,176 logical source bytes; this is 32.962569 percentage points below
the predeclared 93% minimum. Causal lifetime LFU reaches 36.830181%, while LRU
gets no hits because 341 slots cannot span the 376 accesses between token
boundaries. The earlier planning possibility that replacement or prefetch
could turn a 6--8 GiB exact cache into the main Prismwing-50 mechanism is
superseded for this trace. Prefetch may still hide latency, and a larger
hardware-resident cache changes the premise. This short text trace does not
replace E2's eventual multimodal million-position corpus or establish a
universal cache rate. The immutable manifest hashes to
`7e88f6613f5a3f84970763f90ce357cbdff77e499f2f3673c4482829b918ab17`.

PW-0105 supersedes the assumption that PW-0100's 75.7-second token primarily
measures Metal projection arithmetic or buffer copying. A causal, profiled
repeat differs in wall by only 0.4638% and partitions 76.077 seconds into
40.561 seconds routed MoE, 28.673 seconds other layer work, and 6.843 seconds
outside layers. Within routed MoE, repeated tensor/scale validation and page
acquisition consume 16.790 seconds, while the safety-oriented
`release_matrix_transients` invalidates all checkpoint mappings after every
expert and consumes 21.012 seconds. All 1,128 source-buffer copies total only
0.773 seconds; synchronous waits total 0.816 seconds and contain 0.404 seconds
of GPU-active time. Thus the M1 GPU is active for only 0.995% of routed wall,
and 97.115% lies in the named layer-transaction target categories. Promote a
prevalidated, page-stable, layer-scoped runtime artifact/no-copy/async branch;
do not expect `bytesNoCopy` alone to clear 2x. The old global invalidation was
a correct bounded-memory vehicle, not a viable expert-scale lifecycle.

The profiled experts move 9,526,915,072 physical read bytes for
9,464,659,968 installed source bytes, proving that the per-expert release policy
defeats page reuse. Gate 8 still passes with 77% minimum free memory, 4.345 GB
peak RSS, 3.091 GB post-release footprint, and no swap growth, throttling, or
service loss. The result does not promote the rejected L3 arithmetic: it
reproduces token 13 and the layer-4/final-logit failures. Nor is routed
transaction fusion sufficient by itself: 35.516 seconds remains in the current
non-MoE layer/outside-layer path even under an impossible zero-cost routed MoE.
The raw report hashes to
`49c1f85b24e8864d43a3a901de9c7c40e8745a4427599248bd937abba4ce3e11`;
canonical analysis hashes to
`26d649f8babbf00a21bace7c522fab178992d092972ffc55ffb076ac033b1150`.

PW-0106 confirms that PW-0105 found an architectural defect, not mere
instrumentation overhead. On one authenticated real routed layer, a lossless
201.720 MB page-aligned artifact with one layer-scoped mapping reduces cold
wall from a 785.196 ms copied/global-release median to 301.831 ms while still
copying every Metal source buffer, a 2.601x gain. Binding the identical mapping
through Metal's real no-copy API reduces cold wall again to 123.053 ms, 6.381x
versus control and 2.453x versus artifact/copy. Its exact byte-read device probe
passes, and all 18 trials preserve identical expert diagnostics, routed bytes,
final-residual bytes, routes, weights, and repair counts. Thus two beliefs are
now superseded: the safetensors execution layout is not an innocuous substrate,
and `bytesNoCopy` is not merely the 1.9% copied-buffer interval once page
acquisition is moved out of CPU tensor scanning and into the GPU-visible
transaction.

The remaining cold no-copy shape is equally important. Source-buffer creation
is only 0.425 ms and GPU execution 8.291 ms, but synchronous waits total 95.784
ms while 201.720 MB is read. A 47x component extrapolation is still about 5.78
seconds/token before non-MoE work. Promote a bounded asynchronous whole-layer
transaction with retained GPU intermediates and overlap, not a full-bank
artifact or endpoint default. The full 47-layer, 256-expert source-FP8 bank is
about 303 GB (282.6 GiB), so build it only after the next gate. PW-0101's L3
arithmetic failure remains unchanged and separate. Canonical raw evidence
hashes to
`fb0a1cf0e9dba0d3941a5d9786e4867fe04ea21dcd81469d986928fdaada9232`;
analysis hashes to
`635e26fb8060c216e917423c6052a3cb42865bc81a27cf2bc7b4322ce2b7edfc`.

PW-0107 supersedes the narrower belief that collapsing projection barriers is
enough to realize the page-stable layer's cold gain. On the same authenticated
layer-4 row and artifact, reducing 24 command buffers, commits, and waits to
two preserves every expert diagnostic, repair count, routed byte, and final
residual byte. It improves the genuine warm median from 40.358 ms to 23.821 ms
(1.694x), proving that command topology is material once pages are resident.
The cold median improves only from 134.570 ms to 115.447 ms (1.166x), one
paired cold candidate regresses, and 96.001 ms remains inside the two waits
while GPU execution totals 8.320 ms. Thus ordinary no-copy aggregation moves
physical acquisition into fewer waits but does not overlap it with compute.

Reject command aggregation as the promoted cold mechanism and advance a
bounded Metal-I/O/compute-overlap branch with reusable arenas and measured
queue overlap. Do not build the approximately 303 GB expert bank or repeat a
full token yet. Gate 8 passed with 77% minimum free memory, 568,229,888-byte
peak RSS, 122,327,104-byte final footprint, zero swap growth or throttling, and
stable services. Raw evidence hashes to
`39d2a678212a7d98aee33396119928c0e9c2baa7aa4e9f5a19c63ce0fd005bd2`;
clean analysis hashes to
`bc2299248006b349eb2a6a9cee4c5b1a715968fbc9bf118a3d6c9aec702165e2`.

PW-0108 rejects internal-SSD Metal-I/O overlap before building a speculative
arena scheduler. The real M1 API loads all 48 authenticated layer records
directly into a bounded shared Metal buffer with exact bytes and complete
statuses. Three concurrent command buffers improve cold acquisition from a
72.875 ms median to 58.034 ms (1.256x) and warm acquisition from 30.263 ms to
14.782 ms (2.047x). However, the cold result misses the predeclared 47.7 ms
continuation bound by 10.334 ms. Since about 10 ms of unchanged CPU work
survives and only 8.320 ms of GPU work can overlap, Phase B cannot reach the
57.723 ms complete-layer 2x gate on the internal SSD even under ideal overlap.

The earlier broad premise that Metal I/O might eliminate the surviving cold
floor is superseded for this storage and unchanged 201.376 MB selected-byte
representation. Preserve the loader as a conditional hardware/control path,
but do not build its shared-event scheduler. A faster named device or a
lossless executable-byte reduction changes the premise and requires a new
contract. Gate 8 passed with 78% minimum free memory, 419,610,624-byte peak
RSS, 10,503,936-byte final footprint, zero swap growth or throttling, and stable
services. Raw evidence hashes to
`6f7d816b4f39c00b967642bdf300e7baea8563a5fca593ab5d0943b5df047d68`;
clean analysis hashes to
`5281fd36c06e2a2e5767918bbb63f0fe33cbec4a1478b4281806d6fdf56ac43d`.

PW-0109 rejects the exact symmetry available without changing the checkpoint's
128-by-128 source-FP8 scale topology. Deterministic bijective alignment of the
16 groups of 128 SwiGLU neurons per expert is exactly reversible across all 48
selected tensors, but aligned XOR residuals compress to 95.087% at fast zstd:
only 0.0433% smaller than identity-delta and 8.167% larger than the unmodified
87.908% control. The aligned fast stream misses the 25% byte-reduction gate and
its optimistic acquisition-plus-decode bound is 245.804 ms rather than 47.7
ms. Thus block permutation does not expose meaningful exact cross-expert
structure on this route.

Do not expand this mechanism to all experts or build a decoder. Arbitrary
single-neuron canonicalization changes the representation premise because it
mixes current scale blocks; a learned/common basis is also a separate branch.
Gate 8 passed with 78% minimum free memory, 508,133,376-byte peak RSS,
82,200,640-byte final footprint, zero swap growth or throttling, and stable
services. Raw evidence hashes to
`9e0f15f65269d1b5c53536f18cda62df039d13ed19f48242f3eef91966b43bab`;
clean analysis hashes to
`a298ae0b3022fa5f22e06a573af9d1bfdc9471eb33c8fafcd7e664cf26d0b12d`.

PW-0110 supersedes the warm-kernel speculation-width prior with the measured
cold-storage bound. The best exact eight-expert Metal-I/O acquisition takes
58.034 ms per routed layer, or 2.727590 seconds across 47 layers at impossible
minimum `U=1`. With all dense weights, compute, drafting, KV work, correction,
and overhead free, `q=32` can reach only 11.732 accepted TPS. The valuable
34.3-TPS horizon requires `A/U >= 93.556` and therefore at least `q=94`; formal
50 TPS requires `A/U >= 136.380` and at least `q=137`. Any real union or
rejection raises those widths.

Reject `q=16` and `q=32` before proposer training or verifier construction on
the unchanged source-FP8 internal-SSD premise. PW-0044 now requires a
base-aligned candidate pool at least 137 positions wide for the formal branch,
not the rejected supplied DFlash or native MTP. This is necessary evidence, not
proof that `q=137` is memory-fit, accurate, or fast. The immutable analysis
hashes to
`844047de4d009d0d7bd6f803e56e097ee6efce66e6a0c2c7d96315962a5cd8b6`.

PW-0111 resolves the deferred one-barrier Metal-native routed-layer premise.
The real M1 candidate keeps dynamic FP8, BF16 staging, SwiGLU, all 24 source-
FP8 projections, route weighting, deterministic reduction, and scatter inside
one command buffer with one wait and one final residual readback. Despite
omitting C2's 13 sparse repairs, it reproduces C2's exact routed and final-
residual hashes on the authenticated layer-4 row. This supersedes the belief
that the CPU sparse-repair topology is required to preserve the current L3
result on that row, but it does not repair the shared source-derived layer
failure or establish accumulated/hosted parity.

The warm median falls from 41.081 ms to 15.206 ms (2.702x), proving that a
routed layer is the correct CPU-visible compute transaction once weights are
resident. Cold falls only from 131.506 ms to 109.801 ms (1.198x); the median
one-wait interval is still 100.584 ms around 8.383 ms of GPU activity. Thus the
stronger transaction topology does not invalidate PW-0108's cold acquisition
bound. Reject full-bank construction and another token walk on the unchanged
internal SSD/source-FP8 premise; retain C4 for a future wide verifier, exact
byte-reduced artifact, or named faster storage condition. Gate 8 passed with
77% minimum free memory, 538,050,560-byte peak RSS, 68,915,200-byte final
footprint, zero swap growth or throttling, and stable services. Raw evidence
hashes to
`47f764370172dff489629bb171d9dad7345f39e37f21622244a63b6f4edfcb14`;
clean analysis hashes to
`1940aa4554eedc586ff567c041f62f91d757d5afc88244ef895e71ca22488fc0`.
