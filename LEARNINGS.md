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
