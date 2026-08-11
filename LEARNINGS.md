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

PW-0002 pins the published source representation as block-scaled FP8, not the
candidate INT4 representation used by the initial design estimate. The final
local Rust census independently assigns all 73,530 tensors and reproduces the
remote-header totals exactly: `315,683,674,448` tensor bytes,
`315,693,004,496` safetensors-file bytes, and `9,330,048` header/padding bytes.
The separate local verification receipt binds all 39 files to the pinned
revision by byte count and SHA-256 with no missing files.

One routed expert occupies 25,171,968 bytes: three
4096×2048-equivalent FP8 matrices plus three f32 scale grids. The 47×256
routed bank therefore occupies 302,869,118,976 bytes, and a cold source-FP8
token selects 9,464,659,968 bytes (8.815 GiB). The local census hashes to
`82a5916a13d3859b7ad47bea41c5827733b0eb05c3e5b2bb32d6e9244ad4bc17`;
the verification receipt hashes to
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`.
PW-0002's L0 gate is closed for this revision, and the existing throughput
constants remain unchanged because local evidence confirms them exactly.

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

PW-0112 supersedes the remaining premise that a sufficiently wide,
base-aligned proposer could amortize unchanged source-FP8 expert acquisition
on realistic target routes. A frozen 137-token hosted suffix prefix produces
`U=2.401596` at `q=137`, so even perfect acceptance gives `A/U=57.0454` and an
otherwise-free 20.914 accepted-TPS ceiling on PW-0110's cold floor. At `q=94`,
the best of 44 sliding windows reaches only `A/U=46.5665` and 17.072 TPS. Both
are below even the separately valuable 34.3-TPS horizon. Reject proposer
training and a wide verifier until executable-byte reduction or a different
physical store changes this premise; route-coherent candidate selection cannot
change the routes of tokens that are actually accepted.

The same trace refines rather than reverses PW-0104's cache result. Four GiB
(170 equal-size layer-experts) reaches 44.7162% under offline Belady across 128
continuation positions, while global LRU remains at zero and a static cache
trained on 32 positions reaches 29.9507% on the following 96. Adjacent route
sets are identical 57.8379% of the time, but the trace still touches 895
distinct layer-experts and the Belady oracle leaves 5.232 GB of logical misses
per token. Retain route/frequency residency only as a secondary conditional
mechanism requiring a cold physical wall gain; it remains far below the 93%
primary-mechanism hit requirement and is not a Prismwing 34.3/50 path by
itself. The raw route manifest hashes to
`584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e`;
analysis hashes to
`e93d930549ee9fe761d7fc98bf59642088b3eb9f41c712968f8df26d5b2c8b98`.

PW-0113 closes the deeper exact neuron-permutation premise left open by
PW-0109. Replicating source scales per neuron and carrying a U16 inverse
permutation costs only 3.1161%, and all eight selected real experts reconstruct
every source tensor byte. Nevertheless, deterministic 2,048-neuron similarity
assignment makes fast aligned residuals 11.0447% larger than the expanded
unmodified control and 0.6035% larger than identity-delta. The result is only
0.3399% smaller than original source bytes, and its optimistic
acquisition-plus-decode bound is 266.790 ms rather than 47.7 ms.

Reject both block-level and individual-neuron exact canonicalization as the
selected-route executable-byte mechanism. Independently trained experts do not
expose useful shared residual structure under these function-preserving
permutations. Do not expand to all experts or a runtime decoder; a sign
symmetry, learned/common basis, approximate representation, or modified expert
compiler is a separately named premise. Raw evidence hashes to
`f6cb7d8510d2076b35db074a5c6a0511fff7c047effa0dcbb6fe7a146f7aea6a`;
analysis hashes to
`5dfb78f1e32b206050e98754cbcfdfbbf4be2960715954e47465bb882aa51a21`.

PW-0114 resolves the numerical branch left separate by PW-0100/PW-0101 and the
one-barrier result. Across the frozen complete incremental token, disabling all
`[129, 170, 250]` sparse gate/up/down repairs preserves source argmax token 13,
reduces source-chosen absolute logprob error from the repaired control's
0.028671 to 0.024239 nats, retains 19/20 source top tokens, and reduces projected
JSD from 0.000679 to 0.000578 nats. The repair-free candidate therefore passes
the predeclared single-position L3 distribution gate with zero repair bytes.

This supersedes the stronger internal assumption that source-framework
BF16-identical layer states or sparse source-topology repair are always
necessary for a useful final distribution. It does not establish
near-equivalence: the candidate first fails layer-final parity at layer 4,
reaches 3.1774% final-layer relative L2, differs from source routes at 20
layers, and differs from the repaired control's selected experts at three late
layers. Retain repair-free Metal-native arithmetic only as a conditional L3
premise to test with a representation that independently changes the cold-byte
bound. The projection-at-a-time vehicle remains rejected at 75.834 seconds for
one diagnostic incremental token, with zero accepted tokens and no TPS claim.

Gate 8 passed both full processes with 77% minimum free memory, at most
4,430,184,448-byte peak RSS, at most 3,203,014,784-byte post-release footprint,
zero swap growth or throttling, and stable services. Raw control and candidate
evidence hash to
`24622adf564d840880ea44163edbb3c98905d4914b2bc4153cb21151fc58281e`
and `16aaaded5cb082e5672f1a18b132fa52665375117dadb07c0c628fcc76b3b43f`;
clean analysis hashes to
`14866caa426287a61d9ed91a441ff6937465da4e542114ba29b773726b332fa6`.
The updated throughput model hashes to
`021b8688d3cea29da310d3360ff03ad2d261112801956660ca98a566fb9b86ac`.

PW-0115 rejects direct adoption of published MoBE for PW-0045 on a shape bound,
not a quality claim. MiMo's gate, up, and down expert projections have exactly
equal source bytes, while MoBE keeps down unchanged. Even free gate/up factors
therefore leave a 33.333% routed-bank floor, already above the frozen 25% gate.

A deeper whole-mixture factorization of all three projections remains
physically eligible under deliberately optimistic one-byte factors/bases and
resident shared work. Of 48 enumerated rank/basis pairs, 36 clear necessary
bank, selected-stream, compute, and 4 GiB resident-basis bounds. The frozen
activation-audit set is rank-heavy `(r=768,m=4)`, balanced `(512,8)`, and
basis-heavy `(128,32)`. The balanced form models 13.281% of source bank bytes,
12.5% of selected streamed bytes, 37.5% of projection work, and 2,365,587,456
resident basis bytes across 47 layers.

This is an eligibility envelope only. It omits factor quantization metadata and
has no learned artifact, activation-weighted error, route stability, kernel,
wall time, output, accepted token, or TPS. Continue to real routed activations
before training or implementation. Clean analysis hashes to
`41cc9b745561a09073902ba65354889d6b87e7d8716aea4db85940cbafc9c67a`.
The updated throughput model hashes to
`770ae7e017db648ac329b2964b55f3a7589c1f42330e7b02500dc02c2e0b3b23`.

PW-0116 establishes a real, exactly reconstructible activation pilot for the
three PW-0115 shared-basis shapes. Across the frozen 224-position PW-0112 path,
layers 4, 24, and 46 each preserve all 1,792 routed placements and reproduce
both the weighted routed output and final residual bit-for-bit from captured
expert-down rows, schedules, and weights. All 48 route identities and
deterministic source-work counters reproduce the prior authority. The raw
manifest hashes to
`b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`.
Independent validation hashes to
`6007e93aa9cc280d20cab3db0f72851ad9f9722e9f225c07c3c1309cc5ef5e08`.

The pilot also sharpens the corpus limitation: the three layers touch only
69–72 experts total, 26–37 appear at most twice, and contiguous validation and
holdout partitions cover only 10–16 experts per layer versus 66–72 in train.
This supersedes any assumption that the convenient hosted English trace can
serve as representative representation evidence. It authorizes only the
frozen activation-weighted pilot audit and matched controls; any positive
result requires a broader multilingual, long-context, modality, common/rare,
and untouched corpus before promotion. Gate 8 passed at 70% minimum free
memory, 942,702,592-byte peak RSS, zero swap growth/throttling, and stable
services. No throughput-model constant or endpoint TPS changes in PW-0116.

PW-0117 separates the published activated MoBE representation from the only
all-projection basis form that can realize Prismwing's routed-layer transaction.
The released trainer applies SiLU or tanh after each expert's coefficient-
weighted basis combination. That nonlinearity does not commute with basis
combination: deterministic counterexamples differ by 0.132106 and 0.552382.
Even before activation, softmax, materialization, or memory traffic, the three
frozen shapes require 206.25%, 237.50%, and 209.375% of source projection
multiplications. Reject the published activated form as a transaction-compute
architecture without disputing its storage/quality results.

With identity activation, gate/up bases evaluate once on the common input and
down experts factor in transposed orientation so their route-weighted latents
reduce before shared output bases. F64 fixtures prove both reordered forms to
within `4.441e-16`. Including coefficient mixing, `(768,4)`, `(512,8)`, and
`(128,32)` require 37.537%, 37.549%, and 28.174% of source multiplication work,
so all remain physically eligible but entirely untrained. Continue only the
separately named `identity-basis-mixture-compiled` form to weight-space fitting
and PW-0116 activation evaluation. Analysis hashes to
`337b57c43638025673eb494eecfc87445468d21b9a1ce384952b72f6fa47a910`.
The updated throughput model hashes to
`a914eb9949ae201d109ca2c107088687bf9f3101b67fd17b0dddd5551300c7ad`.

PW-0118 removes the local optimizer-memory blocker for the smallest-memory
identity-basis shape. A production `(r=128,m=32)` projection across all 256
experts contains 83,894,272 F32 trainable values. Full Adam state reaches
1,342,505,216 bytes of current MPS allocation and 2,167,029,760 driver bytes,
while a real hot/rare source-tile loss falls 70.742% in five steps. Gate 8
passes at 67% minimum free memory, 2,408,975,168-byte maximum process physical
footprint, zero swap growth/throttling, and stable services; MPS current
allocation returns to zero after release.

This authorizes streamed source-weight fitting on the M1, not a quality or
speed claim. The fixed tile can be memorized and does not establish layer
convergence, shared-basis reconstruction, activation fidelity, executable
quantization, or inference performance. Preserve full PW-0116 validation and
holdout gates, and preflight the much larger rank-heavy optimizer separately.
Raw evidence hashes to
`9d96d71f21f68c249b10422ec0fb479ec905874a93a5631c2488b3fc90e53c9c`;
analysis hashes to
`25f71aabb3d66f3142c8ff8447c451c61d7f527b79a02638b256916fe0db778e`.
No throughput-model constant or endpoint TPS changes in PW-0118.

PW-0119 establishes the independent per-expert rank control that any shared
identity-basis candidate must beat. Across six authenticated hot/rare experts,
the source PyTorch oracle reproduces all PW-0116 expert-down BF16 values
bit-for-bit. Rank error improves monotonically, but the early-layer result is
not representative: rank-768 routed-output relative L2 is only
`0.01977--0.02078` at layer 4, yet `0.70977--0.78318` at layer 24 and
`0.56943--0.71427` at layer 46. Rank 128 is 1.27--10.26x worse than rank 768
on the same outputs.

This supersedes the convenient assumption that a source-weight MSE fit at the
small `(r=128,m=32)` shape is the right first representation test. PW-0118
proved only that its optimizer embodiment fits. Preflight rank-heavy optimizer
memory, then use rank-768 activation-weighted fitting on the frozen corpus
before allocating a shared bank. Even poor SVD cannot finally kill that path,
because it minimizes global matrix Frobenius error rather than error on routed
activations. Raw evidence hashes to
`3e7729dfff3d9ab6793d8e74d29ad20bb3c877bea328ae53d9325737c717c8fb`;
analysis hashes to
`166f56b0b56c82099520acd6696647d8bc350b52d5b33d8649d51a7971cf7a34`.
Gate 8 passes with 81% minimum free memory, 1,036,451,840-byte peak RSS, zero
swap growth/throttling, and stable services. No throughput-model constant or
endpoint TPS changes in PW-0119.

PW-0120 rejects direct full-state MPS Adam for the rank-heavy `(r=768,m=4)`
identity-basis shape. Its 415,237,120 F32 parameters occupy 1,660,948,480 bytes;
after a real forward and dense backward, MPS current allocation reaches
3,321,909,504 bytes and driver allocation 4,306,124,800 bytes. The first Adam
step then reaches 7.01 GiB and is refused while requesting another 1.50 GiB
against the safety-capped 7.10-GiB maximum. Disabling the watermark is not an
eligible continuation.

The bounded rejection preserves live host safety—42% minimum free memory,
zero swap growth/throttling, and stable services—but immediate post-cleanup
physical footprint remains 5,502,110,784 bytes and independently fails the
below-4-GiB release gate even though MPS current allocation returns to zero.
This kills the allocation topology, not activation-weighted rank-768 fitting.
The next optimizer must keep inactive blocks gradient-free and explicitly
bound state through block-coordinate, offload, factored-state, or external
training. Raw evidence hashes to
`8e1a597fc5f15e98fffe2afb0e14964777b7fc5251e5bcb8bf60ae8923d5b2db`;
analysis hashes to
`4fce122f9887f7c103c635337c235767fe66de63372c80219f8b745a191c4a50`.
No throughput-model constant or endpoint TPS changes in PW-0120.

PW-0121 rescues activation-weighted rank-768 fitting from PW-0119's poor
global-SVD result without reopening PW-0120's full-state allocator topology.
On layer-24 hot expert 23, sequential projection fitting reduces complete
expert relative L2 from `0.710381` to `0.251869` on validation (64.54%) and
from `0.684958` to `0.378045` on the untouched holdout (44.81%). Train error
falls 85.18%, so substantial corpus specialization remains visible, but both
predeclared 25% continuation gates pass.

The active projection uses only 4,718,592 parameters and returns MPS current
allocation to zero after each fit. Gate 8 passes at 68% minimum free memory,
1,684,082,816-byte maximum physical footprint, zero swap growth/throttling,
stable services, and a 307,990,080-byte final footprint. This supersedes the
belief that matrix-SVD error alone makes deeper rank-768 experts unpromising;
it does not establish shared-basis quality, broad-corpus generalization, rare-
expert behavior, or a runtime artifact. Replicate at layer 46 before any shared
fit. Raw evidence hashes to
`04388f2704607657fecd5304d2533585e7ee6389080f3e77e5658a9875da05fb`;
analysis hashes to
`6f3c7e8d9ddd25db65dc35cb888a98349bfa89b538cc33be7a2e0ffe5e3c6d17`.
No throughput-model constant or endpoint TPS changes in PW-0121.

PW-0122 replicates PW-0121's activation-weighted rank-768 result at layer 46,
where PW-0119 had different spectra and output scale. For hot expert 28,
complete expert relative L2 falls from `0.572330` to `0.195667` on validation
(65.81%) and from `0.545815` to `0.288128` on untouched holdout (47.21%). Train
error falls 87.17%. All three projection validation objectives improve by
84.02--93.00% without holdout selection.

The result makes the next uncertainty sharing, not depth-general independent
fitting. Any shared-basis pilot must beat or approach these activation-weighted
independent controls rather than compare only with global SVD, and it needs at
least two experts with non-empty train/validation/holdout coverage in the same
layer. Gate 8 passes at 69% minimum free memory, 1,696,583,680-byte maximum
physical footprint, zero swap growth/throttling, stable services, and a
312,348,032-byte final footprint. Raw evidence hashes to
`e05a6a5551e1ef8cb2f5593e0aa44f05a16d1c667dc531a3942e56b20196b50d`;
analysis hashes to
`5b5a21be9438e81e9b05a155ca365cd0dc4180be1b06a18a873362e88f60e0eb`.
No throughput-model constant or endpoint TPS changes in PW-0122.

PW-0123 rejects the first identity-basis topology that actually forces sharing.
With five layer-46 experts and four bases, the four experts initialized with a
private basis remain near their independent activation-weighted controls. The
fifth, expert 57, misses gate/up/down validation NMSE by `3.983x`, `5.053x`,
and `4.863x`. Its shared holdout relative L2 is `0.811428`, worse than both its
independent fitted `0.524725` and global-SVD `0.745367` controls.

Aggregate validation and holdout ratios still pass at `1.112x` and `1.138x`.
This is direct evidence that a favorable mean can hide the exact rare/tail
failure prohibited by the target. The prospective representation remains
physically attractive—`19.336%` of source projection bytes under the named
FP8-factor hypothesis and `37.537%` of source multiplications—but those are not
achieved runtime facts because fidelity failed before quantization or kernels.

The unchanged-duration branch is unattractive: expert-57 validation errors
fall only about 1--3% over the final 30 steps while remaining roughly 3--5x
over gate. The stronger unresolved premise is corpus coverage, since PW-0116
provides expert 57 only 17 training placements. Preserve this rejection and
require broader multilingual/modality, route-stratified activation evidence
before another sharing fit. Gate 8 passes at 69% minimum free memory,
1,957,483,392-byte maximum physical footprint, zero swap growth/throttling,
stable services, and a 269,864,960-byte final footprint. Raw evidence hashes to
`e0f682e77d3f9ca79b762fae52534820af963b3a0478d5d4fa9944694ce5bbc2`;
analysis hashes to
`4d4469184eda8717a12643a58b111d0a4fd6ac72585eb6aaabcfc6c187ab6438`.
No throughput-model constant or endpoint TPS changes in PW-0123.

PW-0124 rejects the cheapest remaining coverage explanation for PW-0123. It
preserves the original positions `168..223` holdout and redistributes only the
development prefix by per-expert occurrence, raising expert 57 from 17 to 58
training placements. Its complete holdout relative L2 improves to `0.515999`
from global SVD's `0.745367`, clearing the 25% improvement gate, while complete
validation and holdout equal-expert ratios pass at `0.953x` and `1.136x`.

That positive composed result does not rescue the representation. Shared-to-
independent aggregate projection NMSE is `2.291x` gate, `3.158x` up, and
`1.856x` down. Expert 57 remains the decisive tail at `5.241x`, `6.817x`, and
`4.329x`; expert 28 also reaches `1.568x` on up. The old assumption that the
forced fifth identity failed chiefly because of the contiguous development
split is therefore superseded. Do not acquire a broad corpus solely to rescue
this exact four-basis/rank-768 form. This does not reject independent fitted
experts, other exact sharing codecs, or layer-transaction execution.

The physical ledger remains only prospective at `19.336%` of source projection
bytes and `37.537%` of source multiplications. Gate 8 passes at 70% minimum free
memory, 1,958,122,368-byte maximum physical footprint, zero swap growth or new
throttled pages, stable services, and zero final MPS allocation. Raw evidence
hashes to
`086cd06b66aa79117e44f3b17e3f1b18b751640d1696e3ce6f3045a769586077`;
analysis hashes to
`a6c98d0469e2e788e5c54833975277ebcffa822a3d0b426a8bb39dbf3606d32a`.
No throughput-model constant or endpoint TPS changes in PW-0124.

PW-0125 rejects PW-0115's balanced `(r=512,m=8)` branch at the independent
capacity rung, before sharing. On layer-46 expert 28, activation-weighted rank
512 cuts matched SVD relative L2 by 62.16% on validation and 46.31% on the
untouched holdout, reaching `0.254728` and `0.352673`. This reinforces the
earlier lesson that routed activations are far more informative than matrix
SVD for these deeper experts.

Capacity relative to the already-working rank-768 fit is the limiting result.
Rank 512 is `1.30184x` rank 768 on validation, missing the frozen `1.25x` gate,
although holdout passes at `1.22401x`. Do not build the nine-expert/eight-basis
sharing optimizer merely because the miss is narrow: sharing can only add a
constraint to an independent form that already missed its predeclared gate.
The result does not reject different objectives, nonlinear or learned
representations outside this identity family, or the retained warm routed-
layer transaction.

Gate 8 passes at 76% minimum free memory, 642,355,264-byte maximum physical
footprint, zero swap growth or new throttled pages, stable services, and zero
final MPS current allocation. Raw evidence hashes to
`916ab149169a518d68eace66f2a6d857679c8e6e5e1777f604c904f0179b08e0`;
analysis hashes to
`b49bfe3082cc2a81ba87c717f9f493f22b7fb9204b6b586699bcce559c1b8fe8`.
No throughput-model constant or endpoint TPS changes in PW-0125.

PW-0126 rejects the first direct routed-mixture compiler form without training
a coefficient network or reading holdout. A layer-level mean plus linear output
dictionary at centered rank 111 reconstructs all 112 training residuals to
roughly `2e-15` relative L2, yet validation remains `0.052437` at layer 4,
`0.270086` at layer 24, and `0.384575` at layer 46. At layer 24, positions
whose routes use only training-seen experts are worse than the training-unseen
slice, so categorical route novelty is not the sole cause.

The old premise that routed residuals might occupy a small fixed linear output
subspace is superseded. Its physical algebra is extraordinarily favorable—a
rank-111 F32 dictionary is only `0.028476%` of the source layer bank and oracle
synthesis is `0.225830%` of source mixture multiplications—but perfect
coefficients cannot repair its validation error. Do not build the coefficient
predictor or executor. Nonlinear or input-conditioned bases remain separate
PW-0045 mechanisms, and the untouched holdout remains available for them under
a new contract.

Gate 8 passes at 79% minimum free memory, 172,902,528-byte maximum physical
footprint, zero swap growth or new throttled pages, and stable services. Raw
evidence hashes to
`7a36bba9d8e6fc24cce802341ecfd56933aa05f7f4c07471004662ac414a5ffe`;
analysis hashes to
`e940d38d84a43332a408b41d6d6f005e9bf24bd3c5950dd61ecfc8d15bf6b1bc`.
No throughput-model constant or endpoint TPS changes in PW-0126.

PW-0127 rejects the plausible under-$500 CPU-only R720 class for Prismwing 50
on authenticated arithmetic, not a market or nominal-bandwidth hunch. Mandatory
checkpoint matrices require `14,820,573,184` MACs or `29,641,146,368`
operations per ordinary target token. This includes all attention projections,
the dense layer-0 MLP, 47 routers, top-eight experts, and LM head while omitting
attention scores, KV work, normalization, nonlinearities, FP8 decoding, NUMA,
networking, and sampling.

Two E5-2680-v2 CPUs have an intentionally overstated 1.152-TFLOP/s ceiling when
all twenty cores are granted 3.60-GHz maximum turbo and perfect 16-SP-op/cycle
AVX issue. Mandatory matrices alone cap this fantasy machine at `38.8649 TPS`;
50 TPS requires 128.65% of peak. The 34.3-TPS horizon is not formally
impossible but requires 88.25% before every omitted cost, so retain it only for
a borrowed-node stage and never as purchase evidence. Ordinary selected expert
bytes alone also cap impossible dual-socket bandwidth at `12.6154 TPS`, barely
above PW-0048's 12.5 pre-purchase gate before dense traffic.

The result kills neither PW-0048 generally nor a complete under-$500
accelerated/modified system. It does prove that cheap high-capacity DDR3 is not
the missing Prismwing-50 mechanism by itself. Gate 8 passes at 79% minimum free
memory, 179,292,288-byte maximum physical footprint, zero swap growth or new
throttled pages, and stable services. Raw evidence hashes to
`6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140`;
analysis hashes to
`5a44e66114b51e2b241acb26fcb2c58280fc2a823314b273c0761d58c27ff113`.
No throughput-model measured constant or endpoint TPS changes in PW-0127.

PW-0128 separates two questions that nominal GPU specifications can easily
conflate. The user's routed-layer transaction premise is physically correct
for a legacy 24-GB accelerator: across the frozen PW-0112 continuation, the
largest layer union is 31 experts, and three complete source-expert arenas need
only `2,340,993,024` bytes. The global `q=137` union is 22.730 GB but never
needs to be device-resident at once. Under an impossible PCIe 3.0 x16 ceiling,
expert-only traffic corresponds to `68.012--77.510 TPS` at `q=94` and
`94.953 TPS` at `q=137`. These are physical diagnostics, not endpoint rates.

The full-capability latency gate, not decode residency, kills the named
hardware. Authenticated mandatory matrices require `29,641,146,368`
operations per position. Granting the dual CPUs 1.152 TFLOP/s and every GPU
its advertised peak simultaneously, only 8,000 uncached positions take
`29.0885 s` on one M40, `15.6500 s` on two M40s, and `18.0299 s` on one P40.
All exceed the 15-second 8K TTFT limit before attention scores, KV, FP8 decode,
transfers, routing, networking, or any utilization loss. Reject these direct-
FP32 configurations before CUDA work or purchase.

Do not generalize this rejection to the transaction architecture itself. It
remains a retained fit for a faster substrate, an L1 executable codec that
changes mandatory work, or a named modified low-bit branch. The market ledger
also never became a BOM: the historical server plus one observed M40 subtotal
is `$453.75`, while required GPU kit, storage, networking, shipping, tax, and
cooling remain unpriced. Gate 8 passes at 79% minimum free memory,
19,170,816-byte maximum physical footprint, zero swap growth or new throttled
pages, and stable services. Raw evidence hashes to
`12a177721d520864bd628ad99b9388cfe9c467bb7ad3706a1329536ce293611a`;
analysis hashes to
`e7ed1e57d7058af7328e0ba48425bb755c8476d87eb596d3b6d869870c8420d8`.
No endpoint TPS or measured throughput constant changes in PW-0128.

PW-0129 replaces the synthetic-activation warning around affine INT4 with a
real source-routed validation result. On PW-0116 layers 4, 24, and 46, every
source capture and route authenticated; prefix reconstruction and three
independent dynamic-FP8/BF16 expert replays were bit-exact. The candidate was
then streamed one expert at a time through the complete gate/up/SwiGLU/down
transaction with unchanged source routes.

Affine group-128 INT4 does satisfy the embodiment ledger: 13,369,344 bytes per
expert is `0.531120` of the 25,171,968-byte source record. Fidelity is
decisively inadequate. Validation routed-output relative L2 is `0.041919` at
layer 4, `0.119174` at layer 24, and `0.154606` at layer 46. Aggregate error is
`0.097661`, and the worst validation row is `0.179215`. Train errors show the
same depth trend, so the result is not a validation-only coverage artifact.
The final 56 positions remain sealed.

Affine INT8 is an informative monotonic control: validation errors fall to
`0.009703`, `0.024061`, and `0.035508`, but the artifact occupies `1.030998`
times source-FP8 bytes and still misses the deeper-layer 2% gate. Faster MLX
arithmetic does not make it an embodiment compression. Do not build or compose
the naive INT4 bank merely because its ideal traffic could approach the 34.3
horizon. Any successor must introduce calibration, outlier-aware mixed
precision, recovery training, or a structurally different executable form and
must earn a new frozen validation gate.

Gate 8 passes at 78% minimum free memory, 226,265,920-byte maximum physical
footprint, zero swap growth or new throttled pages, and stable services. Raw
evidence hashes to
`1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306`;
analysis hashes to
`6d7f75d8b65ccd0ba2fe5c3767e2f2e2a4841c4a859749dbcab8289c7c29b673`.
No endpoint TPS or measured throughput constant changes in PW-0129.

PW-0130 determines how much of affine INT4's real-activation error is a static
per-expert output-channel distortion. It recomputes every PW-0129 packed
artifact and baseline exactly, then grants bias-only and scale-plus-bias F16
repairs fitted on the same validation rows they score. This is a deliberately
noncausal capacity upper bound, not a deployable calibration protocol.

Static repair removes most of the error but not enough. Baseline validation
relative L2 at layers 4/24/46 is `0.041919/0.119174/0.154606`; bias-only reaches
`0.017171/0.030419/0.055696`; affine reaches
`0.011530/0.024850/0.048155`. The affine aggregate remains `0.029916` versus
the 1% gate and its worst row is `0.069135` versus 5%. All nested improvements
are monotonic, and the full F16 repair costs only 4,194,304 bytes per layer or
`0.000651` of the source bank.

The failed premise is now specific: a static diagonal transform after the
complete INT4 expert cannot supply sufficient fidelity, even with validation
leakage. Do not spend evidence or engineering budget on train-only output
scale/bias calibration. The remaining error is input-dependent and/or
cross-channel; viable successors must change weight-domain quantization,
retain mixed-precision outliers, add low-rank residual capacity, or perform
recovery training. The final 56 positions remain sealed.

Gate 8 passes at 78% minimum free memory, 221,186,688-byte maximum physical
footprint, zero swap growth or new throttled pages, and stable services. Raw
evidence hashes to
`b011bd5ced8787df62f4380aeeccab9a35aef8b8ab15541207bcd99e35727994`;
analysis hashes to
`18df3de03834e9725c1b472f196d1e67700d9cdd1c8f18f07e5a9c8d6604bd46`.
No endpoint TPS or measured throughput constant changes in PW-0130.

PW-0131 establishes that compact input-conditioned cross-channel capacity can
repair affine INT4 locally, unlike PW-0130's static diagonal transform. On the
same validation rows used for fitting, errors decrease monotonically at ranks
8/16/32/56. Rank 32 is the first full gate pass: aggregate relative L2
`0.009493`, maximum layer `0.015825`, and maximum row `0.021885`.

The capacity fits the initial embodiment envelope. Rank-32 factors cost
134,217,728 bytes per full 256-expert layer; combined INT4, affine, and repair
bytes are `0.552599` of the source bank. Repair work is 2,097,152 MACs per
eight-expert mixture or `0.010417` of source expert MACs. Rank 56 reaches
`0.000485` aggregate error, which is evidence of memorization capacity rather
than deployable quality.

The old broad inference that INT4's remaining error might be irreducible is
superseded. It is irreducible by static diagonal output calibration, but not by
a small input-conditioned low-rank program. Generalization is wholly unproven:
the next test must fit rank 32 only on positions `0..111`, score `112..167`,
and keep `168..223` sealed. No bank or accumulated model is authorized.

Gate 8 passes at 78% minimum free memory, 221,907,904-byte maximum physical
footprint, zero swap growth or new throttled pages, and stable services. Raw
evidence hashes to
`e0cf60d13b3e55fd805b480bf834baa55e87f7cf5de6b49623f722c094c0d876`;
analysis hashes to
`754285ca807cde425f5742dfb3ffc1014d2a99be9cf7f188eb16307fb3f90042`.
No endpoint TPS or measured throughput constant changes in PW-0131.

PW-0132 supersedes PW-0131's open hope that its compact rank-32 program might
learn a transferable activation-repair rule. Fitted only on positions
`0..111`, it reaches train relative L2 of `0.000519`, `0.003862`, and
`0.023368` at layers 4/24/46, but frozen validation reaches `0.173981`,
`0.097538`, and `0.092928`. Aggregate validation error is `0.150331`, the
worst row is `0.574205`, and the strict and near-miss gates both fail.

This is direct overfit evidence, not merely insufficient route coverage.
Layer 4 has complete validation coverage yet worsens from uncorrected INT4's
`0.041919` to `0.173981`; layer 46 is also fully covered and remains far over
the near-miss threshold. Layer 24 separately records 15 identity-fallback
placements for an expert absent from training. PW-0131 therefore remains a
same-slice capacity oracle only. Do not read holdout, acquire a broader corpus
to rescue this exact mechanism, build its bank, or compose it with the
endpoint. Move to weight-domain calibration, outlier-aware mixed precision, or
a structurally different executable representation.

The prospective physical ledger is unchanged at `0.552599` of source bytes
and `0.010417` of source expert MACs, but embodiment fitness cannot rescue a
failed fidelity mechanism. Gate 8 passes across 216 snapshots at 78% minimum
free memory, 731,004,928-byte maximum peak RSS, 224,037,568-byte maximum
physical footprint, zero swap growth or new throttled pages, and stable
services. Raw evidence hashes to
`0499a40645452eab646276e1619fb2e94b74439ef4263a71f036fae61fd8a9fe`;
analysis hashes to
`c098eb01547d211de5f3bf7fa545b599701616b8142c5689559bcda73e808557`.
No endpoint TPS or measured throughput constant changes in PW-0132.

PW-0133 rejects the cheapest mixed-precision weight-domain continuation. It
keeps PW-0129's affine group-128 INT4 core and uses only training activation
second moments to rank exact source-FP8 row-group exceptions. Every baseline
reproduces exactly and the validation curve improves monotonically, but 1%,
2%, 4%, and 6% exceptions reach aggregate relative L2 of `0.093098`,
`0.089733`, `0.085955`, and `0.083871` from the `0.097661` baseline.

The maximum admissible 6% candidate restores 11,799 groups per expert yet
reduces aggregate error only 14.12%. Layer errors remain `0.025346`,
`0.096881`, and `0.137370` at layers 4/24/46, and the worst row is `0.171606`.
Its conservative sparse artifact is already 14,974,008 bytes per expert or
`0.594868` of source and adds `0.060013` correction MACs; 7% would occupy
`0.605485` of source. Two layer-24 validation experts use the declared
train-absent fallback for 15 placements, but fully covered layers 4 and 46
independently fail by large margins.

The superseded premise is specific: a small set of high diagonal
activation-weighted weight-error groups does not dominate affine INT4's routed
expert error. Do not build this sparse kernel or bank, read holdout, or spend
the remaining byte budget on more exceptions. This does not reject
weight-domain calibration generally. AWQ channel scaling changes the grid,
GPTQ propagates second-order error, and rotations change outlier geometry;
recovery training is deeper still. Each needs a separately frozen mechanism.

Gate 8 passes across 47 snapshots at 78% minimum free memory,
847,396,864-byte maximum peak RSS, 253,119,552-byte maximum physical footprint,
zero swap growth or new throttled pages, and stable services. Raw evidence
hashes to
`a0226e42058a04ea1009a6c00a6b44fdc85728bf36e383166a589b1d3e28b0d8`;
analysis hashes to
`02715ba47566a1269a34ce470e4e04bf6acfd0ebb55c2174b9d329d00300b350`.
No endpoint TPS or measured throughput constant changes in PW-0133.

PW-0134 rejects the official AWQ activation-mean exponent scale family as
adapted to MiMo's independently routed SwiGLU experts. Train-only searches pick
nontrivial input/hidden exponents and improve every validation layer:
`0.041919 -> 0.025631` at layer 4, `0.119174 -> 0.083810` at layer 24, and
`0.154606 -> 0.126141` at layer 46. Aggregate error falls 20.69%, from
`0.097661` to `0.077451`.

The improvement is physically cheap but numerically insufficient. The worst
layer remains `0.126141` and the worst row `0.175005`, far beyond the
`0.02/0.04/0.08` near-miss thresholds. Two layer-24 experts use pooled
activation and median-exponent fallback for 15 validation placements, while
fully calibrated layers 4 and 46 independently fail. Exact pre-quantization
transform reconstruction stays below `2.77e-8`, ruling out the non-homogeneous
SiLU placement as an implementation defect.

The packed representation plus conservative F16 input/hidden scales occupies
13,381,632 bytes per expert or `0.531608` of source. Its 4,096 input divides
are `0.000163` of source expert MACs. This supersedes the idea that a scalar
activation-mean exponent can redirect enough four-bit resolution on its own.
Do not read holdout, compose PW-0133 exceptions, or build a kernel/bank. GPTQ-
style correlated error propagation, function-preserving rotation, and recovery
training remain distinct mechanisms.

Gate 8 passes across 86 snapshots at 78% minimum free memory,
923,418,624-byte maximum peak RSS, 230,214,656-byte maximum physical footprint,
zero swap growth or new throttled pages, and every protected service name
remaining resident. One auxiliary `nxnode` PID exited while another remained;
the normative service-health rule passed and the PID-set change is preserved.
Raw evidence hashes to
`7d470bd5fa5541424c2b619afb49a2ebf493ce7a11b2498cf281b3d1c6f34490`;
analysis hashes to
`8f0da2e109befe20928a1134a178d23343d27afe6c3d60a3e8682d1b5925745c`.
No endpoint TPS or measured throughput constant changes in PW-0134.

PW-0135 finds that correlated second-order assignments have dramatically more
capacity than PW-0129's round-to-nearest INT4, but rejects the frozen
group-local fixed-grid GPTQ form by one narrowly missed criterion. The three
highest-validation-coverage experts reduce validation relative L2 by 69.82%,
63.07%, and 50.60%. Layers 4 and 24 pass outright at `0.033047` and
`0.066439`; layer 46 reaches `0.080659`, only `0.000659` above the frozen
`0.080000` ceiling. Its `0.107637` worst row, train improvement, 50% reduction,
byte ratio, and runtime-MAC conditions all pass.

This supersedes the broad premise that untrained affine INT4 lacks enough
assignment capacity on real routed activations. The evidence instead isolates
the remaining weakness to the frozen group-local grid/curvature scope (or the
margin needed for robust generalization). Do not weaken the gate or run the
contracted full-validation-expert expansion. Preserve the positive signal for
a separately frozen global-Hessian, function-preserving rotation, or recovery-
training experiment. Holdout remains sealed.

The unpacked RTN control stays within `0.000913` relative L2 of the packed MLX
control. Physical accounting remains 13,369,344 bytes per expert,
`0.531120` of source, with no added runtime MACs. Gate 8 passes across 60
snapshots at 78% minimum free memory, 1,063,256,064-byte maximum peak RSS,
353,652,480-byte maximum physical footprint, zero swap growth or new throttled
pages, and stable protected-service PID sets. Raw evidence hashes to
`56b9d38c3c630359b8d5b1a911627882df06a2e2fc374751fde2fddaeb3888db`;
analysis hashes to
`63565129c4f47cff5ab274b687a27bf9c64131ab83e86fab6b3ee4cb98a24bf6`.
No endpoint TPS or measured throughput constant changes in PW-0135.

PW-0136 rejects the missing explicit-`pread` embodiment for the unchanged
internal-SSD source-FP8 representation. Eight fixed-stride 25,214,976-byte
expert blobs are read into eight reusable 2-MiB-aligned allocations already
wrapped by Metal. All 24 trials perform exactly eight complete reads, reproduce
the authenticated 201,719,808-byte artifact hash, and retain exact Metal buffer
pointer identity and length. Every cold trial records the full artifact as
physical reads; every warm trial records zero.

Cold medians at 1/2/4/8 workers are `59.094`, `58.125`, `58.205`, and
`58.515` ms. The selected two-worker result misses PW-0108's unchanged 47.7 ms
continuation bound by `10.425` ms and all its trials exceed the 57.723 ms
ceiling. The result is essentially PW-0108's 58.034 ms Metal-I/O median, so the
shared floor is the internal SSD moving 201.7 MB—not demand paging, Metal-I/O
submission, `pread` submission, or insufficient read concurrency.

Warm parallel copies do improve monotonically to a 13.632 ms eight-worker
median, but that cannot rescue a continually cold source-FP8 decode. Do not
build the source-FP8 protected-slot/pending-MoE scheduler. The architectural
pattern remains useful only after the numerical branch qualifies a materially
smaller executable artifact; PW-0135's 0.531120-byte-ratio INT4 form is now the
specific convergence target, not yet a fidelity-qualified default.

Gate 8 passes across 29 snapshots at 79% minimum free memory, 417,251,328-byte
maximum peak RSS, 206,637,248-byte maximum physical footprint, zero swap growth
or new throttled pages, and stable services. Raw evidence hashes to
`e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56`;
analysis hashes to
`7ebf2cde5c4a3f4931d2d705993f822e38af13ea66bc3efc91410296b14e2aab`.
No endpoint TPS or measured throughput-model constant changes in PW-0136.

PW-0137 establishes that PW-0135's narrow failure was caused by its discarded
cross-group curvature, not by an intrinsically insufficient four-bit grid.
Holding the affine group-128 grids, 0.1% damping, activation order, bytes, and
runtime arithmetic fixed, full-Hessian error propagation moves layer 46/expert
28 from `0.080659` validation relative L2 to `0.059227`. The identical
round-to-nearest control is `0.163279`, so the candidate reduces validation
error by 63.73%; its worst row is `0.077608` and train error is `0.033130`.

The old group-local inference is therefore superseded: second-order assignment
capacity is not confined to independent storage groups, and 128-column storage
boundaries must not be treated as curvature boundaries. All projection-level
train errors improve, all cross-block update norms are nonzero, and the exact
13,369,344-byte (`0.531120` of source), zero-extra-MAC ledger remains intact.
This promotes only a three-expert confirmation. Holdout, a full layer, packed
runtime, model accumulation, and endpoint fidelity remain unauthorized.

Gate 8 passes across nine snapshots at 78% minimum free memory,
1,576,271,872-byte maximum peak RSS, 384,945,408-byte maximum physical
footprint, 365,005,952-byte maximum release-boundary footprint, zero swap
growth or new throttled pages, and resident protected services. Raw evidence
hashes to
`95fee340bb676ac7c9486ea713da9c461ca6fb62441b41b32ff988e97ed1502e`;
analysis hashes to
`7a741514aad2f4ec783cd95b1283ae5b98afbcdad17cd64e8a7759c12f3b5d67`.
No endpoint TPS or measured throughput-model constant changes in PW-0137.

PW-0138 confirms that full-Hessian fixed-grid GPTQ is not a one-expert rescue.
Without layer- or projection-specific tuning, the original representative
experts at layers 4/24/46 reach validation relative L2 of `0.022155`,
`0.059604`, and `0.059227`, reducing their identical affine-INT4 controls by
79.77%, 66.87%, and 63.73%. Maximum-row errors remain below 8%, train improves
for every expert, and every candidate also improves on PW-0135's group-local
assignment. Layer 46 exactly reproduces PW-0137's metrics and assignment
hashes.

This supersedes the concern that global coupling merely overfit the narrow
PW-0135 miss. It now earns an all-validation-expert audit, still on the sealed
train/validation split and still before holdout. The physical case remains
13,369,344 bytes per expert (`0.531120` of source) with zero extra runtime
MACs, but no packed artifact or endpoint conclusion follows until coverage,
accumulation, and execution are separately proven.

Gate 8 passes across 24 snapshots at 79% minimum free memory,
1,648,082,944-byte maximum peak RSS, 417,795,456-byte maximum physical
footprint, 395,726,080-byte maximum release-boundary footprint, zero swap
growth or new throttled pages, and resident protected services. Raw evidence
hashes to
`37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49`;
analysis hashes to
`7ed32546bfb042d5b863c23d812eeada89cafb7d65b9c1d86c30c7483022e14b`.
No endpoint TPS or measured throughput-model constant changes in PW-0138.

PW-0139 reverses the provisional bank direction from PW-0138. Across every
expert selected by validation at layers 4/24/46, the frozen global-Hessian
assignment reaches aggregate routed-output relative L2 of `0.035040`, not the
required `0.010000`. Layer errors are `0.010130`, `0.040686`, and `0.057541`;
the worst row is `0.082790`. Only the early layer clears the original 2% layer
and 5% row gates.

This is a generalization failure, not an authority or execution ambiguity. All
41 experts and 1,344 placements are accounted, source prefix reconstruction
and replays pass, every projection improves on its calibration input, the two
declared layer-24 fallbacks execute, and the three PW-0138 controls reproduce
exactly. The old inference that three high-coverage experts qualified this
representation as a bank candidate is superseded. Do not read holdout, build
the bank, or implement the packed/streaming runtime yet.

Sparse routed calibration is one causal lead: at layer 24, experts with 6 and
19 routed train placements reach `0.124820` and `0.108000` validation error;
at layer 46, a 10-placement expert reaches `0.100973`. The train-absent pooled
fallback experts reach `0.053410` and `0.071468`, suggesting pooled-Hessian
shrinkage deserves a cheap falsification. It is not sufficient evidence for a
rescue: several better-covered deep experts remain near 6--7%, and the routed
layer misses by multiples rather than a narrow margin.

The physical ledger remains 13,369,344 bytes per expert (`0.531120` of source)
with zero extra runtime MACs. Gate 8 passes across 296 snapshots at 78% minimum
free memory, 1,765,031,936-byte maximum peak RSS, 394,153,664-byte maximum
physical footprint, 348,393,152-byte maximum release-boundary footprint, zero
swap growth or new throttled pages, and resident protected services. Raw
evidence hashes to
`83bd204c9d5c35a684cab15a4ddacf48cf9b661563fb26223eb3655d0ef4a7b5`;
analysis hashes to
`9aecfdcd32e535b4b9d27fcac075dfd1c9014080d624aa3f4af2c678be3f3b6c`.
No endpoint TPS or measured throughput-model constant changes in PW-0139.

PW-0140 confirms that sparse routed calibration contributes materially to the
deep-layer GPTQ failure, but rejects pooled-only calibration as a complete
remedy. Using all 112 layer train inputs moves layer 24/expert 39 from
`0.124820` to `0.063960` validation relative L2 and expert 128 from `0.108000`
to `0.068350`. Layer 46/expert 140 improves from `0.100973` to `0.083363`, but
misses the frozen `0.080000` ceiling by `0.003363`.

Every pooled projection improves its round-to-nearest calibration control, so
the gain is real. The branch still cannot be promoted: its policy was inspected
on validation-visible failures, one expert fails, and PW-0139 also contains
better-covered deep experts near 6--7% while the complete routed layers miss by
larger margins. Do not derive a hybrid threshold from these validation labels,
read holdout, or build the runtime. The next mechanism must change outlier
geometry through a function-preserving rotation or learn transferable recovery.

Gate 8 passes across 24 snapshots at 79% minimum free memory,
1,643,970,560-byte maximum peak RSS, 370,298,304-byte maximum physical
footprint, 333,581,696-byte maximum release-boundary footprint, zero swap
growth or new throttled pages, and resident protected services. Raw evidence
hashes to
`824d66549da7833d855f430a60b761f145a98757fa191bb41db6cf6e56f78b9f`;
analysis hashes to
`1efbd70bba8c5a3a1a7ade6668ff76d90b2bcbde7a931988fae47db0a1a7ebe9`.
No endpoint TPS or measured throughput-model constant changes in PW-0140.

PW-0141 rejects a fixed model-wide randomized-Hadamard residual basis as the
missing weight-only INT4 geometry. The function-preserving algebra is sound:
unquantized expert parity is `1.93e-15` to `3.52e-15` relative L2 and the
orthogonal round trip is exact at reported precision. Every rotated GPTQ
projection also improves its rotated round-to-nearest calibration control.

The validation result is nevertheless neutral. Layer 4/expert 96 changes from
`0.022155` to `0.022241`, layer 24/expert 200 from `0.065851` to `0.065040`,
and layer 46/expert 249 from `0.067440` to `0.066235`. The deep gains are only
1.23% and 1.79%, versus the frozen 25% requirement. Rotated round-to-nearest
errors worsen to `0.095517`, `0.150095`, and `0.185599`, demonstrating that
full-Hessian assignment—not the fixed rotation—provides nearly all quality.

Do not spend another validation-visible seed, rotate the checkpoint, read
holdout, or build the runtime. A learned rotation is a different mechanism,
but after the decisive full-bank miss the next bounded branch should test
recovery training directly. The physical ledger remains 13,369,344 bytes per
expert (`0.531120` of source) with no prospective per-layer residual transform.

Gate 8 passes across 24 snapshots at 79% minimum free memory,
1,789,853,696-byte maximum peak RSS, 373,395,072-byte maximum physical
footprint, 346,541,696-byte maximum release-boundary footprint, zero swap
growth or new throttled pages, and resident protected services. Raw evidence
hashes to
`4dae2abe2a59457a77e09bd4d1328b7b6dce8f0e41e3ac115fd27645c93e56a9`;
analysis hashes to
`0cd0f7d5cd9d8fd563a1c35888a16e8452f8f9128d26c36ce7e02d646cc3bf26`.
No endpoint TPS or measured throughput-model constant changes in PW-0141.

PW-0142 rejects the first direct recovery-training schedule before it can make
a generalization claim. Holding PW-0139's four-bit codes fixed and training
only the existing group-128 scales and biases end to end leaves layer 4/expert
96 unchanged after F16 staging. It makes layer 24/expert 200 worse from
`0.030712` to `0.092812` on train and `0.065851` to `0.342938` on validation;
layer 46/expert 249 worsens from `0.039279` to `0.131436` on train and
`0.067440` to `0.375717` on validation.

This supersedes the idea that the already-assigned codes can be rescued by one
frozen, cheap group-parameter schedule. It does not reject recovery training
generally: the chosen optimizer fails its allowed training objective, so no
claim about validation transfer follows. Do not tune it against visible
validation, read holdout, expand to all experts, or build its bank. A successor
must change the codes, training authority, or executable representation and
freeze a new cheap gate.

All PW-0139 initial metrics and grids reproduce, codes remain fixed, and the
13,369,344-byte (`0.531120`) zero-extra-MAC ledger remains exact. Gate 8 passes
across 51 snapshots at 62% minimum free memory, 1,629,175,808-byte maximum peak
RSS, 937,020,992-byte maximum physical footprint, 286,133,184-byte maximum
release-boundary footprint, zero swap growth or new throttled pages, and stable
services. Raw evidence hashes to
`0c2095a2068ccf347ab86beccb41e8d303444ce371b52d8475a37b26c29e9cc7`;
analysis hashes to
`9e9faf22442899ad98b60dbcda6eb59bad5bf864d4daba0c62b82e94300d7460`.
No endpoint TPS or measured throughput-model constant changes in PW-0142.

PW-0143 repairs a separate installation-authority defect exposed by PW-0142.
The verified checkpoint did not change: all 39 files retained path, byte count,
inode, nanosecond mtime, and receipt SHA-256, while macOS changed only the APFS
mount-session device number from `16777233` to `16777231`. Treating `st_dev` as
a durable identity incorrectly invalidated a fully verified installation after
an ordinary remount or reboot.

Runtime file identity now has one shared authority. It continues to require a
verified hash-bearing receipt record and exact size, inode, and mtime, while
recording device drift diagnostically. Twenty-three affected tests and the full
63-Rust/205-Python plus Metal/MLX gate pass, all 39 real files pass with only
the known device drift, and PW-0142 opens real shards and completes through the
unchanged receipt. No receipt, model hash, accepted token, endpoint TPS, or
throughput-model constant changes in PW-0143.

PW-0144 rejects the first code-changing recovery schedule before it can test
generalization. Its straight-through optimizer gives layer 4/expert 96 no
effective update. At layer 24/expert 200 it changes 5,398,042 of 25,165,824
codes yet worsens train relative L2 from `0.030712` to `0.158823` and
validation from `0.065851` to `0.684947`. At layer 46/expert 249 it changes
5,870,177 codes and worsens train from `0.039279` to `0.170402` and validation
from `0.067440` to `0.444887`.

The changed-code capacity is real, but the frozen learning dynamics are not a
recovery mechanism: they make the allowed train objective substantially worse.
This supersedes only the specific fixed-grid, dimensionless-offset, 0.05-Adam
schedule. It does not reject code-changing or grid-changing QAT generally. Do
not use visible validation to tune this branch, read holdout, expand it, or
build its runtime. A successor must freeze materially different training
authority or move to a different executable representation.

All initial PW-0139 metrics reproduce, final codes remain in `[0,15]`, F16
grid metadata is bit-identical, and the 13,369,344-byte (`0.531120`)
zero-extra-MAC runtime ledger remains exact. Gate 8 passes across 51 snapshots
at 58% minimum free memory, 1,632,075,776-byte maximum peak RSS,
1,430,818,176-byte maximum physical footprint, 285,494,144-byte maximum
release-boundary footprint, zero swap growth or new throttled pages, and stable
services. Raw evidence hashes to
`8828db18f3d9471aa9abd2110b78994b86803ff393e4ec0d9fe81e10cef5d00c`;
analysis hashes to
`93871a88c85a883ced215a233e25791b1f39861a86332e60fd1e2a9cfc64db28`.
No endpoint TPS or measured throughput-model constant changes in PW-0144.

PW-0145 isolates why conservative code-QAT schedules cannot repair PW-0139.
On layer 46/expert 249 train data only, learning rates `0.0001`, `0.0005`,
`0.001`, and `0.005` for 32 steps produce maximum latent offsets of `0.003196`,
`0.015981`, `0.031962`, and `0.159808`. Every value remains below the 0.5
rounding boundary, all 25,165,824 integer codes stay unchanged, and train
relative L2 remains exactly `0.039279`.

This rejects the tested low-rate family without making a validation claim:
validation values were never loaded and holdout remains sealed. Together,
PW-0144 and PW-0145 reveal a discontinuous optimizer regime—small schedules
cannot cross a code bin, while the much larger schedule changes 23% of codes
and diverges. One explicitly threshold-crossing train-only schedule is the
remaining cheap test of this parameterization; repeated validation-visible
schedule tuning is not authorized.

Code domain, F16 grid metadata, and the 13,369,344-byte (`0.531120`)
zero-extra-MAC ledger remain exact. Gate 8 passes across 33 snapshots at 59%
minimum free memory, 1,502,953,472-byte maximum peak RSS, 1,423,019,456-byte
maximum physical footprint, 449,891,968-byte maximum release-boundary
footprint, zero swap growth or new throttled pages, and stable services. Raw
evidence hashes to
`1d5f4f4bf9dacc39114d483f90e3e61590f847aa24a31a1c6d48dbb077deafa4`;
analysis hashes to
`a562ec97d8a9e49566b562a4cee2d88df102f1d9afff2987ed16988de6bfa687`.
No endpoint TPS or measured throughput-model constant changes in PW-0145.

PW-0146 closes the remaining train-only schedule interval for fixed-grid
straight-through code offsets. The predicted threshold-crossing schedule
reaches `0.628309` maximum displacement and changes 590,345 of 25,165,824
codes (`2.3458%`), satisfying the bounded code-change condition. It is not a
recovery: train relative L2 explodes from `0.039279` to `1.062956` and loss
from `0.001542818` to `1.129876494`.

Together, PW-0144 through PW-0146 reject further schedule search for this
specific parameterization. Low rates remain inside the rounding dead zone;
the first bounded crossing changes many coupled weights discontinuously and
destroys train; the high schedule also diverges. Validation was never loaded
in PW-0145 or PW-0146, so this is optimizer/representation evidence rather
than a new fidelity result. Grid-changing training, a coordinated discrete
optimizer, or a different executable representation remains logically open.

Code domain, F16 metadata, and the 13,369,344-byte (`0.531120`)
zero-extra-MAC ledger remain exact. Gate 8 passes across 15 snapshots at 59%
minimum free memory, 1,533,788,160-byte maximum peak RSS, 1,424,920,128-byte
maximum physical footprint, 302,648,448-byte maximum release-boundary
footprint, zero swap growth or new throttled pages, and stable services. Raw
evidence hashes to
`7bb795455927295c673bfe65d06ae6311dbdd97b9d3517caa357307d189bdcf3`;
analysis hashes to
`9ea3ba39fc7381389e168b09478cdf3789cbcce846ac82b096e392b5d42bc3f3`.
No endpoint TPS or measured throughput-model constant changes in PW-0146.

PW-0147 tests the next representation-capacity point rather than another
optimizer schedule. Five-bit group-128 affine grids with unchanged
global-Hessian assignment materially improve all three exact PW-0138 four-bit
validation controls: relative L2 becomes `0.019898`, `0.047222`, and `0.043164`
at layers 4, 24, and 46. The early expert clears the frozen `2%` validation and
`5%` worst-row gates; both deep experts miss the validation gate, and layer 24
also misses worst-row at `0.054097`.

This rejects the tested five-bit form on the representative experts without
collapsing the broader low-bit branch. All candidates improve five-bit
round-to-nearest on train and four-bit global-Hessian on validation, so bit
capacity—not a failed optimizer—is causally implicated. A separately frozen
six-bit control is the next cheap falsification; an all-validation-expert audit
remains unauthorized.

The prospective five-bit representation remains physically exact at
16,515,072 bytes per expert (`0.656090` of source), 198,709,346,304 bytes for
the routed bank, and zero additional runtime MACs. Gate 8 passes across 24
snapshots at 66% minimum free memory, 1,902,084,096-byte maximum peak RSS,
397,365,184-byte maximum physical and release-boundary footprint, zero swap or
new throttled pages, and stable services. Raw evidence hashes to
`a7706fce33dc716930d080988e197089bcf1ebb6fb5729adcdb3203a8cccd62e`;
analysis hashes to
`4c7a5ef1a4a91816fd66021d4068e2224057d1c3967069b915bae3655e4cff4e`.
No endpoint TPS or measured throughput-model constant changes in PW-0147.

PW-0148 closes the affine-group-128/global-Hessian bit-width ladder for a
256 GiB companion embodiment. Six-bit validation relative L2 is `0.019209`,
`0.044075`, and `0.038193` across the early, middle, and late controls. All
maximum rows clear 5%, all candidates improve six-bit round-to-nearest on train
and immutable five-bit controls on validation, and all authorities reproduce;
nevertheless both deep experts miss the unchanged 2% gate.

The sixth bit provides only small validation gains over five-bit and leaves
optimized train errors essentially unchanged (`0.005448`, `0.041244`, and
`0.032966`). Treat this as a generalization/topology floor for the tested
fixed affine-grid assignment, not a reason to tune the acceptance threshold.
Seven-bit storage would require 22,806,528 bytes/expert and 274,408,144,896
bytes (255.5625 GiB) for routed experts alone. It is therefore ineligible for
a 256 GiB companion after spine, KV, runtime, OS, and safety headroom.

The exact six-bit prospective ledger is 19,660,800 bytes/expert (`0.781059` of
source), 236,558,745,600 bytes for the routed bank, and zero added MACs. Gate 8
passes across 24 snapshots at 66% minimum free memory, 1,899,380,736-byte peak
RSS, 395,677,568-byte maximum physical and release-boundary footprint, zero
swap or new throttled pages, and stable services. Raw evidence hashes to
`48d1c28cc589e55002ce5a4b836d62ef172d3ed77106c100b2ad49d708fd1257`;
analysis hashes to
`66cf40287ba8c60bd8ee52e143311fac21e0dc07414c9d1f3230edb5a71ecf64`.
No endpoint TPS or measured throughput-model constant changes in PW-0148.

PW-0149 rejects deterministic per-row-group nonuniform scalar INT4 as the
missing capacity mechanism. Sixteen F16 Lloyd centroids per group produce
validation relative L2 of `0.033762`, `0.060601`, and `0.060640`, with
maximum-row errors of `0.068806`, `0.081569`, and `0.077481`. Every expert is
worse than its immutable six-bit affine control and fails both frozen gates.

The assignment mechanism remains powerful on visible train positions:
global-Hessian propagation reduces nonuniform round-to-nearest train error from
`0.094429/0.179407/0.157468` to `0.005443/0.041036/0.032874`. Convergence near
the same four-to-six-bit train floor, coupled with poor validation, strengthens
the causal inference that fixed scalar grids are compensating the calibration
slice rather than learning a transferable expert representation. Do not spend
more experiments on scalar level counts, spacing, or another fixed seed.

The exact prospective ledger is 18,874,368 bytes/expert (`0.749817` of source),
227,096,395,776 bytes for the routed bank, and zero added matrix MACs. Gate 8
passes across 24 snapshots at 65% minimum free memory, 1,856,372,736-byte peak
RSS, 362,843,968-byte maximum physical and release-boundary footprint, zero
swap or new throttled pages, and stable services. Raw evidence hashes to
`f8860f648cc6596d5c6a35eca7b2236270676aa421d0890c1d2b02236dffd54a`;
analysis hashes to
`eeb5576f1d20f81cfa0c6326622fa649262ac5fcf13ca09625e59b7512044f18`.
No endpoint TPS or measured throughput-model constant changes in PW-0149.

PW-0150 supersedes PW-0102's implicit assumption that pinned-base embedding
row 151675 is a representative DFlash mask input. That base row is effectively
zero (`0.0000207325` F32 norm); the separately shipped and authenticated BF16
mask has norm `1.4794452`, relative L2 distance `71358.67`, and cosine
similarity `-0.0310`. Using the exported draft-only value changes the frozen
proposal causally from `[264,1773,102092,102092,102092,1773,1773,1773]` to
`[264,11,11,11,11,11,11,11]`. Future DFlash audits must preserve the shipped
mask embodiment rather than importing the base placeholder row.

That correction does not rescue the supplied proposer. The pinned base's first
required suffix remains token 13, while the draft chooses token 11. Token 13 is
only draft rank four and trails by `2.21875` logits, so matching suffix length
is zero and formal `A=1` counts only the anchor. Since a width-eight top-eight
route union necessarily has `U>=1`, this block has `A/U<=1` and cannot meet the
strict routed-byte leverage gate. No second target walk is justified. Reject
the supplied DFlash-8/mask/base combination on this trace while leaving a
base-trained proposer or wider route-coherent lattice logically open.

The passed raw manifest hashes to
`0582f905d8d6531e0c7d4e9a50def819a6d337a62c5e9b0cac351caa9435f882`;
analysis hashes to
`72051c021ae1d93989508b0423ab1b0811072c24799b8e986d4543b4a513f04e`.
Gate 8 passes at 62% minimum free memory, 4,110,647,296-byte peak RSS,
299,475,008-byte maximum physical footprint, zero swap growth or throttling,
and stable services. This is proposal evidence, not accepted endpoint TPS, and
no throughput-model constant changes.

PW-0151 binds the already-owned H11SSL-i/EPYC 7351P host and photographed EVGA
NEX750B to the complete mandatory matrix ledger. The CPU cannot explain either
useful horizon: even an impossible all-core 2.9-GHz, 16-FP32-op/cycle grant is
only `25.0463 TPS`. One P40, one P100, and one V100 also fail the 15-second 8K
matrix-only prefill floor at advertised FP32 peak. Two P100s are the only
tested source-preserving compute configuration that survives, at an impossible
`12.2596` seconds before all non-matrix work.

Compute survival is not system survival. The real `q=137` route moves
`22,730,287,104` source expert bytes. With two advertised-peak P100s and four
ideal 2.5-GB/s independent storage lanes, serial storage plus matrix compute is
`2.48297` seconds: `34.3 TPS` needs `A>=86/137`, and Prismwing 50 needs
`A>=125/137`. The latter is a `91.24%` exact accepted prefix before dense
weights, attention, KV, protocol, filesystem, cooling, and contention. This
supersedes any assumption that inexpensive card prices alone make the owned
host procurement-ready.

The photographed PSU is EVGA SuperNOVA NEX750B: 750 W continuous at 50 C,
61 A/732 W combined +12 V, four 20-A rails, with VGA1/VGA2 on +12V2 and
VGA3/VGA4 on +12V4. This does not reject two 250-W cards by itself, but it
makes original-compatible cabling, connector pinout, forced airflow, clearance,
rail loading, and measured wall power mandatory evidence. Retain two P100s,
roughly 10-GB/s-or-better expert storage, and a base-aligned wide proposer only
as a conditional envelope; no purchase or endpoint claim follows.

The authoritative report hashes to
`d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1`.
Gate 8 passes at 65% minimum free memory, 30,932,992-byte peak RSS, zero swap
growth or throttling, and stable services. The first report with a mistyped
full commit is preserved and rejected; the final analyzer now authenticates
HEAD and a clean tree rather than merely validating 40-hex syntax. No measured
throughput-model constant or endpoint TPS changes in PW-0151.

PW-0152 supersedes the remaining assumption that conventional DFlash block
chaining could supply PW-0151's extreme acceptance inside one target
transaction. The supplied width-eight shape needs 18 target transactions to
span 137 positions; the published width-16 shape needs nine. Each next block
depends on the clean target bonus anchor from the preceding verification, so
their accepted lengths cannot be summed while charging only one routed-expert
union.

Tree width does not repair the fixed `q=137` budget. The 34.3-TPS prerequisite
requires depth at least 86 and leaves 51 off-path nodes. Prismwing 50 requires
depth at least 125, placing 91.24% of candidate nodes on one path and leaving
only 12 for branching. The constant-independent-match diagnostic requires
`p=0.9925414` and `p=0.9986313`, respectively. The strongest published DFlash
Table 6 result, `tau=6.33/16`, implies only `p=0.8548765` under that same
diagnostic; the 50-TPS branch needs about 106.03x less mismatch. Cross-model
paper values are not a bound on a newly trained MiMo proposer, so retain only
a distinct `q>=137` or depth-at-least-125 base-aligned architecture as
unproven. Kill training and runtime work that preserves conventional
width-eight/16 DFlash boundaries or ordinary chaining.

The authoritative manifest hashes to
`68783813c30d08aabb6c23971d65b2579655314819ea8d6e1aef8b19328bc686`.
Gate 8 passes with 66% minimum free memory, 30,867,456-byte peak RSS, zero swap
growth or new throttled pages, and stable services. PW-0152 reports zero
accepted tokens and no endpoint TPS; no throughput-model constant changes.

PW-0153 supersedes the assumption that source-resident DRAM would leave
PW-0151's extreme storage-driven acceptance prerequisite unchanged. The pinned
checkpoint census contains `315,683,674,448` tensor bytes (`294.0033` GiB), so
five 64-GiB modules are the byte minimum with only `25.9967` GiB raw headroom.
The official EPYC-7001 population table does not explicitly enumerate five;
six is enumerated as unbalanced and not recommended, while eight is balanced.
The existing four 4-GiB modules cannot be mixed into the replacement bank
because the manual requires the same DIMM type, size, and speed.

At the explicitly impossible ceiling of five DDR4-2400 channels and two
encoding-adjusted PCIe-3-x16 links, PCIe limits PW-0151's real `q=137` expert
payload to a `0.72142`-second transfer. Adding the two-P100 matrix floor gives
`0.93137` seconds, reducing the 34.3-TPS requirement to `A>=32/137` and the
50-TPS requirement to `A>=47/137`. Preserve resident expert memory as a
physically meaningful architecture; do not report these nameplates as measured
bandwidth or endpoint throughput.

The dated procurement embodiment is nevertheless rejected under the `$500`
incremental cap. The authenticated active listing asks `$247.19` per compatible
64-GiB module, so the five-module byte minimum is `$1,235.95`, already `$735.95`
over the complete cap. With PW-0151's two P100 card-only subtotal, named parts
reach `$1,384.69` before cables, forced-air cooling, tax, shipping, or physical
validation. This falsifies the captured 2026-08-09 procurement branch, not all
future used-market opportunities. Reopen only from a new dated compatible
complete BOM; do not buy or train a resident-only proposer from PW-0153.

The authoritative manifest hashes to
`b11989c53cb93da52140e61c5d16b0152ffc80322184451c08a63c66712444c4`.
Gate 8 passes with 66% minimum free memory, 139,083,776-byte peak RSS,
85,919,360-byte maximum physical footprint, zero swap growth or new throttled
pages, and stable services. Two failed invocations published no manifest and
are preserved externally: one corrected the assumed census evidence class and
one corrected an operator commit typo. PW-0153 reports zero accepted tokens,
no endpoint TPS, and no throughput-model constant changes.

PW-0154 supersedes the assumption that only a large host-DRAM bank can provide
meaningful exact source-expert residency. Two P100s' 32 decimal GB aggregate
HBM can arithmetically reserve all `12,814,555,472` non-routed source tensor
bytes, `2,340,993,024` bytes for three maximum routed-layer arenas, and
209,879,040 bytes for exact BF16 8K KV, leaving 660 complete expert slots.
This is an aggregate necessary bound; per-card sharding and communication are
not proven.

A static frequency set learned only from the 87-position prompt is causally
valid for PW-0112's following 137-position suffix. It hits 71.434% of accesses
and avoids 53.045% of the suffix union records, leaving 424 exact source experts
or `10,672,914,432` bytes. This promotes prompt-calibrated HBM residency only
as a component of a changed physical envelope, not as a cache runtime or
measured I/O result.

One 3.5-GB/s lane plus the PW-0151 compute floor cannot reach Prismwing 50 even
with perfect `A=137`; its impossible ceiling is 42.033 TPS. Two, three, and
four such lanes require `A=87`, `A=62`, and `A=49` for 50 TPS; four require
`A=34` for the separately valuable 34.3-TPS horizon. Conventional width-eight
and width-16 DFlash remain structurally insufficient. Retain only the combined
two-to-four-lane cache envelope pending complete BOM, sustained-read, CUDA,
prefill, 1M-KV, communication, electrical, and thermal evidence.

The authoritative manifest hashes to
`1b57250d45f1b24e32f43e93a653fc3d00fa061e37cd0df1c6f0fdff551535f2`.
Gate 8 passes with 67% minimum free memory, 146,636,800-byte peak RSS,
90,555,968-byte maximum physical footprint, zero swap growth or new throttled
pages, and stable services. One failed invocation published no manifest and
corrected an eight-versus-nine full-attention-layer count by deriving it from
the pinned config. PW-0154 reports zero accepted tokens, no endpoint TPS, and
no throughput-model constant changes.

PW-0155 supersedes the assumption that PW-0154's four storage lanes require
four separate PCIe adapter cards. The owned H11SSL-i exposes x16 slots 2, 4,
and 6, each with `x4x4x4x4` bifurcation, so two double-width P100s can
logically coexist with one passive four-drive M.2 carrier. This proves a lane
topology only; chassis, cooler, connector, and obstruction clearance remain
physical gates.

The authenticated PSU/P100 ledger tightens the branch. Two 250-W P100 board
limits plus the EPYC's 170-W TDP total 670 W, only 62 W below the NEX750B's
732-W combined +12-V label before the board, memory, drives, fans, and
transients. NVIDIA permits up to 240 W/20 A at each card's CPU-style 8-pin
input, equal to one entire labeled PSU rail, and specifies the
`030-0571-000` dual-PCIe dongle. Full-power execution therefore remains
unproven until original cable inventory, pinout, rail assignment, staged wall
power, temperature, ECC, and throttling evidence exists.

The dated named component subtotal is `$403.38`, but its `$96.62` arithmetic
margin excludes unknown tax/shipping and any missing original cables. The SSD
listing contradicts itself on model identity, the cheap dongles are unbranded,
and sustained reads and cooling fit are unmeasured. Reject this captured list
as purchase authority while retaining the two-P100/quad-NVMe architecture as
conditional. The authoritative manifest hashes to
`226603fb2b44e1162a038f51bae47520238150f3b26e39e1cf33c7420b88b064`.
Gate 8 passes with 67% minimum free memory, 45,236,224-byte peak RSS, zero swap
growth or new throttled pages, and stable services. PW-0155 reports zero
accepted tokens, no endpoint TPS, and no throughput-model constant changes.

PW-0156 establishes that exact source-FP8 router boundary ties are not rare
enough to ignore in long causal route traces. The primary and alternate 8K
fixtures stopped at causal positions 466 and 261. A closed, pre-frozen panel
then stopped at positions 509, 309, 20, 146, and 423. None produced the
contracted 512-position manifest, so there is no observed distinct-expert
count and no evidence for either side of the 9,003-record four-lane prefill
gate.

Supersede the assumption that custom `nth_element` tie choice can serve as an
exact PyTorch routing authority merely because untied fixtures agree. A tied
expert changes all downstream hidden states; prompt selection cannot repair
that semantic gap. Preserve the two-P100/four-lane branch as conditional and
require actual source-framework route indices, or a separately proven exact
equivalent, before repeating the coverage walk. No throughput-model constant
changes, and no Gate-8 or endpoint claim follows from these no-manifest safe
stops.

PW-0157 supersedes only PW-0156's tie-authority blocker. A hash-pinned fixture
from the actual PyTorch 2.13.0 CPU build proves Prismwing's libc++ bridge exact
on adversarial tied rows, including unsorted output order. The 512-position
original control and bounded one-shot K/V-release runtime preserve every
route-semantic field. Boundary ties occur 3, 9, 12, 23, and 63 times across
the exact 512, 1,024, 2,048, 4,096, and 8,000 position walks; they are now
authorized and explicitly counted rather than ignored.

Distinct `(layer, expert)` coverage is 2,980, 3,572, 4,456, 4,585, and 4,903.
At 8K, the impossible 660-record offline-residency grant still leaves 4,243
records or `106,804,660,224` source bytes, but the observation remains 4,100
records below the predeclared four-lane rejection point. Retain four-lane 8K
storage capacity only as a conditional arithmetic envelope. This does not
measure storage, CUDA, prefill time, or endpoint TPS and does not reverse
PW-0158's complete-system rejection. The authoritative analysis hashes to
`e7df87bb326e543b5b500c698eae1700d2fd204d6b2d2a833736706456955cfc`.
Gate 8 passes at 69% minimum free memory, `3,967,156,224`-byte peak RSS, zero
swap growth or throttling, and stable services. No accepted-token or
throughput-model performance constant changes.

PW-0158 supersedes the assumption that PW-0151/PW-0154's surviving two-P100
8K decode envelope could remain a complete target-faithful hardware candidate
without first closing the one-million-token capability slice. At exactly one
million positions, the pinned nine global-attention layers require
`184,320,184,320,000,000` FLOPs for QK and weighted-V arithmetic alone; the 39
sliding layers bring mandatory attention work to
`184,524,643,656,007,680` FLOPs. Even granting both P100s their combined
advertised 37.4-TFLOPS FP16 peak continuously, perfect scaling, and zero cost
for every other operation yields an `82.2302`-minute attention-only floor.
The 30-minute gate would require `102.5137` TFLOPS, or `2.7410x` that favorable
peak. Reject ordinary dense attention on two P100s for the 1M slice; kernel,
fusion, storage, and scheduling work cannot repair an arithmetic lower bound
that already makes them free.

Exact BF16 KV at one million positions is `23,065,559,040` bytes. Together
with PW-0154's non-routed source tensors and three arenas it exceeds aggregate
two-P100 HBM by `6,221,107,536` bytes. Even free-streaming every non-routed
tensor leaves only 261 complete expert slots. The PW-0151/PW-0154 results
remain useful 8K component evidence but no longer retain the complete
two-P100 target-faithful embodiment. Changed-attention L3/L4 mechanisms and
other complete hardware candidates remain open and must preserve the full
long-context gates. The authoritative report hashes to
`3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315`.
Gate 8 passes at 62% minimum free memory, 32,997,376-byte peak RSS, zero swap
growth or throttling, and stable services. PW-0158 reports zero accepted
tokens, no endpoint TPS, and no measured throughput-model constant changes.

PW-0159 corrects and narrows the remaining cheap Ampere counterexample. The
12-GB RTX 3080 has 70 SMs, but the official GA102 rates must not be conflated:
dense BF16 with FP32 accumulation is `61.2864` TFLOPS, while `122.5728`
TFLOPS is dense FP16 with FP16 accumulation. The first analyzer made that
mistake; its preserved manifest is rejected. At one million positions,
mandatory matrices plus ordinary attention require
`214,165,790,024,007,680` FLOPs. The source-oriented BF16 control therefore
needs `58.2418` minutes before every omitted cost and is rejected. The L3 FP16
diagnostic narrowly survives at `29.1209` minutes with only `52.7462` seconds
left, but has no fidelity promotion.

The first 4,096 exact PW-0157 positions already establish the storage floor.
After an impossible free stream of all common tensors and perfect-foresight
preload of all 375 available expert slots, 4,210 distinct records or
`105,973,985,280` bytes remain. Two ideal 3.5-GB/s lanes plus the favorable L3
arithmetic need `17.1832` seconds for 8K; three lanes first survive at
`12.1369` seconds. The active captured card plus three drives and adapters is
`$575.00` before unknown tax and missing parts, so that dated procurement
branch is rejected. Reopen only below a `$371.72` delivered-card ceiling before
tax and with a complete valid BOM; an already-sold bargain is not purchase
authority.

Exact BF16 1M KV exceeds 12-GB HBM by `11,065,559,040` bytes. Turbo4 can fit
arithmetically but remains an unqualified L3 mode with material prior component
error. The authoritative corrected report hashes to
`945079702501f990e2cdd40a326b09fad0f2bb71b3f9615c8114c0bbd71590c2`.
Gate 8 passes at 63% minimum free memory, 294,305,792-byte peak RSS, zero swap
growth or throttling, and stable services. PW-0159 reports zero accepted
tokens, no endpoint TPS, and no measured throughput-model constant changes.

PW-0160 proves the local construction side of a million-token hosted canary but
does not prove the external answer-key path. The pinned source template renders
exactly 1,000,000 tokenizer IDs; its request hashes to
`a21c154c87bb2ce0f3c3305b52655cac04538fdfa4224b36f73e96503167048b`,
with the early SHA-derived needle at token 32 and question at token 999,973.
Frozen OpenRouter metadata advertised Parasail FP8, 1,048,576 context, and both
required logprob parameters.

Three bounded attempts produced no model response: one HTTP-200 JSON 502 and
two HTTP 429 bodies that explicitly name Parasail's shared upstream provider
pool and request a later retry. The sequence is transient availability
evidence, not a prompt-length, logprob, truncation, or retrieval failure. The
attempt budget is exhausted, so keep the million-token hosted reference
unproven without killing it, switching provider, or weakening TARGET. The
authoritative report hashes to
`635748d36a1fc6d690d0261c3526519f5b1bc558745cb4dc574432369f133048`.
Reported usage/cost is absent; the frozen-price worst case for all attempts is
`$0.42001344`. Gate 8 passes at 47% minimum free memory, 1,230,258,176-byte
peak RSS, zero swap growth or throttling, release boundaries, and stable
services. PW-0160 reports zero accepted tokens, no endpoint TPS, and no
throughput-model constant changes.

PW-0161 closes the actual 32-GB PCIe Volta forms rather than inferring from
memory capacity or the newly confirmed single-card PSU margin. Mandatory 1M
matrices plus ordinary attention require `214,165,790,024,007,680` FLOPs. Even
granting the standard V100's advertised 112-TFLOPS deep-learning peak and the
owned EPYC's impossible `0.7424`-TFLOPS peak concurrently yields a
`1,899.6029`-second floor, missing the entire 30-minute gate by `99.6029`
seconds before all non-arithmetic work. Reject ordinary dense 1M execution on
the standard V100 regardless of kernels, storage, or future price.

V100S's favorable 130-TFLOPS L3 ceiling survives arithmetically at
`1,638.0745` seconds but leaves only `161.9255` seconds for every omitted
operation and has no source-BF16 fidelity promotion. Exact 1M KV, three arenas,
and non-routed source tensors exceed its 32-decimal-GB HBM by
`6,221,107,536` bytes; free-streaming all common tensors leaves 261 complete
expert slots. The active captured standard V100 costs `$679.00` and V100S
costs `$1,054.99` delivered before tax, so both card-only ledgers already
exceed the complete `$500` cap before cable, cooling, or storage.

Reject both captured procurement branches. Retain only V100S as a future
price-triggered L3 hypothesis; arithmetic survival does not authorize runtime
work or purchase. The authoritative report hashes to
`fc438d593d8ac99be3cc426496feb830256ffc48c75d58fc8bb9d6b09a2c6c8f`.
Gate 8 passes at 52% minimum free memory, 32,800,768-byte peak RSS, zero swap
growth or throttling, and stable services. PW-0161 reports zero accepted
tokens, no endpoint TPS, and no measured throughput-model constant changes.

PW-0163 closes AMD MI100 rather than treating the NVIDIA-only accelerator
survey as complete. The strongest source-oriented MI100 rate is 92.3-TFLOPS
BF16 Matrix, not its 184.6-TFLOPS FP16 figure. Even granting that BF16 peak and
the owned EPYC's impossible peak concurrently, mandatory 1M matrices plus
ordinary attention need `2,301.8085` seconds, missing the complete gate by
`501.8085` seconds before every omitted cost. Reject source-oriented ordinary
dense MI100 execution permanently.

The 184.6-TFLOPS FP16 ceiling survives arithmetically at `1,155.5143` seconds
but remains an unqualified L3 mode. Exact 1M KV, three arenas, and common
weights exceed 32 decimal GB by `6,221,107,536`; free-streaming common tensors
leaves 261 complete expert slots. A single 300-W card plus the 170-W CPU leaves
262 W below the PSU's combined +12-V label, which is margin rather than cable,
airflow, fit, or measured-load proof.

The active captured used MI100 costs `$999.00` before unknown tax, independently
rejecting the current procurement branch by `$499.00` before any mandatory
cooling, cabling, storage, or OS work. ROCm 7.1 excludes the owned Debian 13
installation for MI100. Retain FP16 only as a future price-triggered L3
hypothesis; authorize no purchase or HIP runtime. The authoritative report
hashes to
`dcc6a60955e8dfd67a3f1da582b33b332bfa31ece45a70d4606df8e367bcb145`.
Gate 8 passes with 51% minimum free memory, 32,210,944-byte peak RSS, zero swap
growth or throttling, an explicit release boundary, and stable services.
PW-0163 reports zero accepted tokens, no endpoint TPS, and no measured
throughput-model constant changes.

PW-0164 closes the strongest NVIDIA Blackwell tier officially launched below
the complete `$500` hardware cap. RTX 5060 Ti's 759 advertised AI TOPS is not
a dense BF16 rate and cannot be substituted into Prismwing's unchanged source
arithmetic. Scaling NVIDIA's official same-generation RTX 5070 dense Tensor
rates by 36/48 SMs and 2570/2512 boost gives `47.343451433121`-TFLOPS
BF16/FP32-accumulate and `94.763634554140`-TFLOPS FP16/FP16-accumulate
ceilings.

Neither survives the complete one-million-position arithmetic bound. Granting
the owned EPYC's impossible peak concurrently, source-oriented BF16 needs
`4,453.8213` seconds and the favorable L3 FP16 mode needs `2,242.4320`
seconds. The latter is still `442.4320` seconds beyond the entire TTFT gate
before softmax, routing, decode, memory traffic, storage, dispatch, or any
other work. Permanently reject RTX 5060 Ti for ordinary dense 1M execution
regardless of price; do not generalize to RTX 5070+, changed attention, or
modified FP8/FP4 weights.

Exact BF16 1M KV alone is `23,065,559,040` bytes, exceeding 16 decimal GB by
`7,065,559,040`; KV, three arenas, and common weights exceed it by
`22,221,107,536`. The 180-W GPU plus 170-W CPU leaves 382 W under the
authenticated 732-W +12-V PSU label, but that is not installation proof.
Official 16-GB MSRP was `$429`; a dated NVIDIA Marketplace row observed
`$479.99` out of stock, so neither is a complete delivered BOM.

The authoritative report hashes to
`6e34c7496694db3aca10c105bbc642b440c6e97922100fb083a2f1be1acea856`.
Gate 8 passes at 44% minimum free memory, 41,091,072-byte peak RSS,
20,694,336-byte maximum physical footprint, zero swap growth or throttling,
and stable services. PW-0164 reports zero accepted tokens and no endpoint TPS;
no measured throughput-model constant changes.

PW-0165 closes the strongest currently in-stock AMD consumer card below the
complete hardware cap. Granting RX 9060 XT's full official 103-TFLOPS dense
half-precision Matrix rate to BF16/F32 accumulation, plus the EPYC's impossible
concurrent peak, gives a `2,064.3998`-second lower bound for mandatory 1M
matrices and ordinary attention. That is `264.3998` seconds beyond the entire
gate before softmax, routing, source-FP8 decode, memory, storage, or dispatch.
The same rate rejects the favorable dense FP16 L3 diagnostic. Permanently
reject ordinary-dense 1M RX 9060 XT regardless of future price.

AMD's 205-TFLOPS structured-sparse rate is a real but separate physical clue.
It would yield an ideal `1,040.9414` seconds, but the authenticated RDNA4 ISA
requires two zero elements per four for the sparse form. Unchanged Prismwing
weights do not admit that premise. Retain explicit 2:4 weight modification as
a named modified-representation branch; do not report the sparse nameplate as
source performance.

Exact BF16 1M KV exceeds 16 decimal GB by `7,065,559,040` bytes. The 160-W
GPU plus 170-W CPU leaves 402 W under the authenticated 732-W +12-V PSU label,
and the card uses one 8-pin input, but those are not installation proof. AMD's
16-GB SEP was `$349`; a dated new in-stock retailer row is `$449.99` with free
shipping before unknown tax, leaving only `$50.01` for the rest of the complete
BOM.

The authoritative report hashes to
`7ce474e66fca10bb87b3a5c016f689792119539b87c54438245473e153999d58`.
Gate 8 passes at 49% minimum free memory, 40,108,032-byte peak RSS,
20,252,288-byte maximum physical footprint, zero swap growth or throttling,
and stable services. PW-0165 reports zero accepted tokens and no endpoint TPS;
no measured throughput-model constant changes.

PW-0166 supersedes the remaining uncertainty that Intel Arc B580's unpublished
BF16 peak might leave an affordable ordinary-dense Xe2 path open. Intel's
pinned compiler semantics assign two BF16 operations per DPAS channel versus
four for INT8, and its Xe2 scheduler models equal same-size DPAS latency and
occupancy independent of precision. Combined with Intel's official
233-INT8-TOPS row, the strongest source-oriented BF16/F32-accumulate ceiling is
therefore 116.5 TFLOPS, not 233.

Mandatory one-million-position matrices plus ordinary attention total
`214,165,790,024,007,680` operations. Even granting that ceiling and the owned
EPYC's impossible peak concurrently yields a `1,826.6923`-second floor,
already `26.6923` seconds beyond the complete gate before every omitted cost.
Permanently reject B580 for ordinary-dense 1M execution regardless of future
price. This does not reject changed attention, modified weights, faster Xe2
products, or multi-card systems.

Exact BF16 1M KV exceeds 12 decimal GB by `11,065,559,040` bytes. The 190-W
board plus 170-W CPU leaves 372 W under the authenticated PSU's 732-W combined
+12-V label, but this is not installation proof. The official `$249` launch
price is not a delivered BOM, and no purchase or oneAPI implementation is
authorized. The authoritative report hashes to
`30908aee4e494aa12c31223ba6b2072684f3c1a954e300d9b55566b978591bce`.
Gate 8 passes at 73% minimum free memory, 36,192,256-byte peak RSS,
19,416,576-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable services. PW-0166 reports zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes; 116.5 TFLOPS is a derived ceiling rather than achieved performance.

PW-0167 reverses the broader inference that affordable Intel consumer cards
have no ordinary-dense one-million arithmetic survivor. Intel's official A770
row gives 262 dense INT8 XMX TOPS, and its Xe-HPG architecture specifies 4,096
INT8 versus 2,048 FP16/BF16 operations per Xe-core-cycle. The resulting
source-oriented ceiling is 131 TFLOPS. Mandatory 1M matrices plus ordinary
attention total `214,165,790,024,007,680` operations; granting that ceiling and
the EPYC's impossible peak concurrently yields a `1,625.6406`-second floor,
leaving `174.3594` seconds inside the gate before every omitted cost. Retain
A770 as an arithmetic survivor only, not as an achieved endpoint.

Exact BF16 1M KV still exceeds 16 decimal GB by `7,065,559,040` bytes, and KV
plus three arenas and common source tensors exceeds it by `22,221,107,536`, so
layer-major or host/storage streaming is mandatory. The 225-W card plus 170-W
CPU leaves 337 W under the photographed NEX750B's authenticated 732-W combined
+12-V label, but this is not installation proof. The owned H11SSL-i lacks
supported native Resizable BAR, Intel requires it for optimal Arc performance,
and the listed oneAPI client-GPU Linux systems do not include the owned Debian
13 environment. Two dated sub-`$500` used sales are not an active complete BOM.

The authoritative report hashes to
`0ff6f2cb1017cb6589b8c5705e7adda349fc2637721e3ddc8c695f051dff2c01`.
Gate 8 passes at 70% minimum free memory, 34,635,776-byte peak RSS,
21,087,488-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable services. PW-0167 reports zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes; 131 TFLOPS remains a derived ceiling. Require an active complete BOM
and reversible installed oneAPI BF16/PCIe/ReBAR-off/on component evidence
before purchase or runtime work.

PW-0168 closes PW-0167's active-inventory uncertainty for one exact board, but
not its complete-BOM or installation gates. GUNNIR's authenticated Photon 16G
OC panel specifies two 8-pin inputs, 285-W TBP, and 300x118.5x50-mm dimensions;
this corrects marketplace metadata claiming three inputs and replaces the
225-W Intel reference-card premise only for this candidate. The card plus the
170-W EPYC leaves 277 W under the authenticated 732-W combined +12-V label.
VGA1/+12V2 and VGA3/+12V4 are a candidate separate-rail plan, not cable or
pinout proof.

The active new listing shows `$411` plus `$20` shipping, four available, and
import fees included. It leaves `$69` before unauthenticated destination tax
and installation parts, so it supersedes only the inference that no active
sub-cap A770 observation exists. Chassis clearance, original EVGA cables,
checkout total, cooling, absent native ReBAR, unsupported Debian 13 oneAPI
placement, and installed performance all remain open. The authoritative report
hashes to
`dfd12ca7bb331003e28241e1c5eac49c579eecfa90cb5216fb41edb8a297f6bd`.
Gate 8 passes at 70% minimum free memory, 31,162,368-byte peak RSS,
20,252,224-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable services. PW-0168 reports zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes. Retain the exact card only pending physical and checkout evidence; do
not purchase from this record.

PW-0169 finds a materially better exact active A770 candidate and supersedes
PW-0168's Photon as the preferred listing, not as purchase authority. The used
domestic Intel Limited Edition card identifies MPN `21P01J00BA`; Intel binds
that form to 225-W TBP, required 8-pin plus 6-pin inputs, 279.9-mm maximum
bracket-inclusive length, 126.36-mm maximum width, and 42-mm maximum height.
GPU plus CPU leaves 337 W under the PSU's combined +12-V label.

The active listing shows `$300` plus `$11.71` shipping to the renderer's
`27709` destination, leaving `$188.29` before actual-destination differences,
tax, and installation parts. That is credible complete-BOM room, unlike the
Photon's `$69`, but the seller's working-order statement and original box are
not a component test; the listing has no seller returns. Clearance, original
EVGA cables and pinout, actual checkout, cooling, native ReBAR, supported
oneAPI placement, and installed performance remain open. The authoritative
report hashes to
`127a898e54f51044bf68bf58f80d071e98b2e10130f2b008a6fe0d313d2d9db3`.
It binds all four original listing images: the box label independently shows
the 16-GB `21P01J00BA` identity, and the card photos match the Limited Edition
form and connectors. Images do not prove function. The earlier image-unbound
report
`b6c125f1a8cb937b0bb847936e5b251a9d65cb13f7254ca1ae215d60aa450baa`
and image-bound report
`d08060c9fa494245069bb61169c48b0b8484c2c2796fa68b34b5cc89c892bfb9`
are preserved and superseded.

The installed continuation gate requires at least `118.238594` sustained
BF16/F32-accumulate TFLOPS even with the EPYC at its impossible peak and zero
time for every omitted operation. That is `90.2585%` of A770's derived ceiling;
30, 60, and 120 seconds reserved for everything else raise it to `91.7979%`,
`93.3904%`, and `96.7460%`. This makes a source-shape-weighted component
benchmark a cheap kill gate, not performance proof. Gate 8 passes at 71%
minimum free memory, 31,195,136-byte peak RSS, 20,219,264-byte maximum physical
footprint, zero swap growth or throttling, an explicit release boundary, and
stable services.
PW-0169 reports zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes. Prefer this candidate for physical and checkout evidence; do not
purchase from the report.

PW-0170 supersedes the residual implication that a 16-GB A770 could turn its
HBM into a meaningful exact expert cache after satisfying common state. Every
non-routed source tensor, three maximum arenas, and exact 8K BF16 KV consume
`15,365,427,536` of 16 decimal GB. The remaining `634,572,464` bytes hold only
25 complete experts. Prompt-frequency calibration avoids 25 of the real
`q=137` suffix's 903 union records, or `2.76855%`, leaving 878 records and
`22,100,987,904` source bytes. Reject the HBM cache as a primary mechanism.

The stronger A770 arithmetic does reduce but does not remove the speculation
prerequisite. Four ideal 2.5-GB/s lanes plus full derived A770 and impossible
EPYC compute need `A=77/137` for 34.3 TPS and `A=113/137` for 50. Four ideal
3.5-GB/s lanes need `A=56/137` and `A=81/137`; the latter's diagnostic
independent conditional match probability is `0.9914746`. Width-eight/16
blocks cannot provide these values in one target transaction. Retain only a
new base-aligned `q>=137` proposer combined with four measured storage lanes,
not the supplied or published proposer shapes.

The active card's `$311.71` item-plus-rendered-shipping observation leaves
`$188.29` before actual tax, four drives/carrier, compatible cables, and
cooling. The owned host has no NVMe. This remains credible room, not a complete
BOM. Installed A770 BF16/ReBAR-off/oneAPI, sustained storage, proposer,
physical, electrical, and checkout evidence all remain required. The
authoritative report hashes to
`c8eba5c4348378177d0d297b8eb4713fd9be71aa2f5a7c2790895c35859af5af`.
Gate 8 passes at 65% minimum free memory, 141,000,704-byte peak RSS, zero swap
growth or throttling, an explicit release boundary, and stable services.
PW-0170 reports zero accepted tokens, no endpoint TPS, and no measured
throughput-model constant changes.

PW-0171 supersedes PW-0169/PW-0170's inference that the A770's remaining
`$188.29` was credible room for a complete four-lane storage BOM. An active
quantity-four listing for exact Samsung PM981a 256-GB drives costs `$159.96`,
and even the favorable lower bound charges only one `$8.15` order shipping
fee. The active passive-bifurcation quad-M.2 carrier costs `$39.99` delivered.
Together storage is at least `$208.10`; adding the authenticated `$311.71`
card observation yields `$519.81`, already `$19.81` over the complete cap
before tax, GPU cables, or cooling. Reject this exact active BOM.

The physical storage premise itself is not disproved. Four drives provide
`1,024,000,000,000` decimal bytes, and Samsung specifies 3,500 MB/s sequential
read for the exact 256-GB model, matching PW-0170's favorable 14-GB/s aggregate
nameplate. Concurrent sustained reads and platform bifurcation remain
unmeasured, and a cheaper future or already-owned four-lane set could reopen
the mechanism. The authoritative report hashes to
`14549b38ee1daee523fd5a76ca9654cdcf7aa6284c651fb36eccac68908b28d3`.
Gate 8 passes at 72% minimum free memory, 29,048,832-byte peak RSS,
17,958,400-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable services. PW-0171 records zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes; authorize no purchase.

PW-0172 prevents PW-0171's 3.5-GB/s BOM rejection from being generalized to
the slower storage branch. An active exact-part listing has six Samsung PM981
256-GB drives at `$28.99` with free observed shipping. Four drives plus the
active `$39.99` carrier and PW-0169's `$311.71` card observation total
`$467.66` before tax, leaving `$32.34` for tax, cables, and cooling. On the
`$455.95` taxable item subtotal, only a `7.092883%` tax rate consumes all
remaining room even if installation parts are free. Retain this only as a
pre-tax BOM; complete delivered cost is not proved.

The retail product page advertises 2,800 MB/s for the matching
`MZVLB256HAHQ` base part, 12% above PW-0170's 2.5-GB/s-per-lane grant, but this
is not manufacturer authority or a sustained installed measurement. The
conditional branch still requires four concurrent measured lanes, working
bifurcation, the A770 compute gate, and a base-aligned `q=137` proposer with
`A=113` for 50 TPS (`A=77` for 34.3 TPS). The authoritative report hashes to
`2b38a618c0364ce2c11a7d93b2bf57e357c38d8cc5f3edfc2da954a6795da564`.
Gate 8 passes at 72% minimum free memory, 29,163,520-byte peak RSS,
17,680,000-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable services. PW-0172 records zero
accepted tokens, no endpoint TPS, and no measured throughput-model constant
changes; authorize no purchase.

PW-0173 closes the implication that a newer released speculative-decoding
configuration can directly supply PW-0170's missing horizon. After favorably
granting one target bonus token, EAGLE-3, P-EAGLE, AngelSpec DFly, and BASTION
have maximum published paths of 9, 6, 8, and 17 tokens. The strongest is still
39 tokens short of PW-0170's least demanding `A=56` requirement. Reject all
four audited configurations as direct Prismwing proposers.

BASTION's largest reported slice mean accepted length is 10.60. The retained
`A={56,77,81,113}` requirements are 5.2830x, 7.2642x, 7.6415x, and 10.6604x
that value. This is a cross-model research prior, not a MiMo bound: preserve a
newly trained or scaled MiMo-specific `q>=137` proposer as an explicitly
unproven residual branch. No audited project publishes compatible MiMo draft
weights, and the neural configurations require target-specific training and
hidden-state or distribution access.

The authoritative report hashes to
`15ec2cfa3ea80a3914ce500f3cb8288a2149cc1948469aeecde04922f6f7a16d`.
Gate 8 passes at 72% minimum free memory, 35,520,512-byte peak RSS,
17,728,896-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable services. PW-0173 records zero
accepted Prismwing tokens, no endpoint TPS, no measured throughput-model
constant changes, and no purchase authority.

PW-0174 closes the released L3 mismatch-acceptance loophole left outside
PW-0173. Approximate Speculative Decoding changes target trajectories by
accepting bounded low-regret mismatches, but its primary `DSpark-14B-block7`
configuration can commit at most eight tokens with a favorable target bonus.
That remains 48 tokens below PW-0170's minimum `A=56`; request regret budget
`B=8` is not proposal depth. Reject the released configuration as the missing
proposer.

The mechanism is promising within its actual scope: mean accepted length rises
from 3.85 to 4.20, with 7.78% mean and 15.26% maximum throughput improvement
over matched strict verification. But the paper also reports over 95% hash
divergence on named tasks and a worst task-score point change of -1.52
percentage points. It does not report Prismwing's hosted top-20 distributional
gate, native modalities, one-million context, paired capability confidence
intervals, or MiMo execution. Reject it as sufficient L3 fidelity evidence.

Preserve only a separately scaled MiMo-specific `q>=137` ASD branch with the
complete validation protocol as unproven. The authoritative report hashes to
`2a8bbcc3d70740501fea245e33b28313d23447cfdde205c139f86981e4f4dd6e`.
Gate 8 passes at 72% minimum free memory, 29,818,880-byte peak RSS,
19,170,816-byte maximum physical footprint, zero swap growth or throttling,
an explicit release boundary, and stable services. PW-0174 records zero
accepted Prismwing tokens, no endpoint TPS, no measured throughput-model
constant changes, and no purchase authority.

PW-0175 changes the changed-attention prior: PW-0162's probability-ranked
history oracle does not exhaust structured, causally selectable sparsity.
MInference is training-free at the weight level, acts on prefill, and performs
a model-specific offline layer/head pattern search followed by a last-64-query
online vertical/slash selector. Its released GLM-4-9B-1M configuration has
1,280 head records and a favorable `1.230279%` selected-causal-pair upper bound
at one million positions. Charging the online index QK work raises this to
`1.237959%` of ordinary global-attention work.

The independently reproduced complete two-P100/EPYC allowance leaves
`21.056139%` of ordinary global work after matrices and sliding attention.
MInference's released configuration therefore passes a structural continuation
screen with substantial arithmetic margin. This is not a MiMo or hardware
result: the GLM head map is not reusable, perfect sparse-kernel efficiency is
granted, and the sources provide no MiMo pattern, Metal/P100 implementation,
hosted top-20 gate, or native-modality fidelity. Promote only a MiMo-specific
source-state oracle that derives its own head patterns and executes the online
selector before any kernel work.

Quest's released query-aware sparse path is decode-only; its code explicitly
leaves prefill dense. Reject it as the PW-0158 prefill repair without rejecting
its decode mechanism. The authoritative PW-0175 manifest hashes to
`e5ac56b7f710285cdeb0088f9fa750748ad74cbc68cd6d4dcb627061209a37ab`.
Gate 8 passes at 72% minimum free memory, 50,085,888-byte peak RSS,
18,613,632-byte maximum physical footprint, zero swap growth or throttling, an
explicit release boundary, and stable services. PW-0175 records zero accepted
tokens, no endpoint TPS, no measured throughput-model constant changes, and no
purchase authority.

PW-0162 found that a content hash is only as semantic as its payload. The
first global-attention oracle walk failed its route non-interference guard, but
the observer-disabled same-shape control also produced a different
`layer_routes_sha256` from PW-0157. Direct comparison then showed that every
one of the `47 * 512 = 24,064` ordered expert rows and route-weight rows was
bit-exact. The hash had included the entire `LayerRouteTrace`, particularly
per-layer `wall_ms`, so it was guaranteed to drift across executions without
any model-state change.

The repaired authority hashes only layer number, ordered selected experts, and
ordered route weights. Both authenticated traces produce
`c0e5c8fd8c72f148895d39fdf38b95e84e93228206563ea49b242f48b0c69872`;
a deterministic fixture proves timing changes preserve that identity while a
route change does not. The observer-disabled control manifest hashes to
`480b02816b293ed8a2275e3c2810ee940fa0916db31fd1d730d6331e9f00a025`
and passes Gate 8. This is a correctness repair, not evidence for or against
20%-history pruning, and it changes no throughput-model constant or purchase
decision.

The corrected full run then established a second, genuinely causal instrument
failure: its semantic route hash changed. A dataflow can be shadow-only and
still fail to be an observationally passive execution boundary when it
allocates, sorts, renormalizes, and performs large reductions between source
head computations. Do not infer a pruning result from that failed run. Move
all candidate work after the authoritative pass: preallocate bounded sampled
Q/K/V and reference-output storage, perform copies only during source
execution, require the exact route hash, then require bit-exact offline replay
before computing oracle errors. This repair changes no throughput-model
constant, architecture disposition, or purchase authority.

The preallocated copy-only attempt still failed exact route identity after
`1,679` seconds. Moving observer arithmetic offline is therefore necessary but
not sufficient for observational passivity. The failure manifest hashes to
`026f116129543b02285d81e20bb0a3a7746c91623f4403a0aa10f262b9d87189`.
Do not guess whether the remaining cause is capture allocation/copy behavior,
binary code generation, or Accelerate variability: run a same-commit
no-capture control and make the next fail-closed message quantify actual hash,
expert, route-weight, absolute, and ULP drift. This remains a correctness
repair with no pruning, throughput, or purchase decision. The next bounded
repair removes heap-backed capture vectors in favor of one fixed-offset
anonymous mmap and must pass a same-commit 64-position control/capture smoke
before another full walk.

The unchanged-binary no-capture discriminator completed in `1,677.943614`
seconds and reproduced all `24,064` semantic route rows bit-exactly to
PW-0157. Its manifest hashes to
`9e95643ae0cba8ee9eda2f0447f477d05e839a02e13ff457e80499cbba86bcce`.
The rebuilt source runtime is not the cause of the prior drift; the capture
path is. Gate 8 passes with 71% minimum free memory, 562,126,144-byte maximum
footprint, 839,761,920-byte peak RSS, zero swap growth or throttling, and
383,928,960 bytes after release. Continue only through the mmap-backed short
smoke; this still changes no pruning or throughput conclusion.

The fixed-offset anonymous-mmap capture passes that required same-commit,
same-shape 64-position smoke. The frozen-original no-capture authority hashes
to `d6b1483b0d6161611f58b2746edcd5f356f503c337bf590fcee2989f3d436f66`;
the capture manifest hashes to
`07adc240519642719c49c822aa25a1e7b38581d7ac629ddde5e0a5690e8013aa`.
All 3,008 source route rows remain exact under the observer, and offline 100%
replay is bit-exact for all 73,728 captured output values. Gate 8 passes for
both runs with at least 70% free memory, no swap growth or throttling, and
resident protected services. This supersedes the belief that any bounded
capture necessarily perturbs this source runtime; capture embodiment and
allocation topology matter. It authorizes the final 512-position oracle walk,
but the 64-position candidate errors remain diagnostic and change no pruning,
throughput, or purchase decision.

PW-0162's production preflight exposed one more authority distinction before
opening weights: hashing parsed JSON route numbers is not byte-identical to
hashing the runtime's typed F32 route payload. The authenticated PW-0157 route
values hash to `c0e5c8fd...c69872` under analyzer JSON canonicalization and
`9cf63371...b7a0dc` under runtime F32 canonicalization. Pin both with explicit
names. Use the latter for the runtime raw-report guard and the former for
cross-manifest analyzer comparison; do not call their expected difference
route drift.

PW-0162 decisively rejects simple probability-ranked global-history pruning
at the arithmetic fraction required by the cheap two-P100 system. The valid
non-causal oracle preserves exact source routes and replays 1,105,920 control
values bit-exactly, yet retaining the best 20% of visible rows yields
`0.172375` aggregate relative L2, `3.025940` in the worst layer, and
`4.888554` head-query p99, versus limits of `0.010000`, `0.020000`, and
`0.050000`. The exact `21.056139%` arithmetic boundary remains far outside at
`0.167131` aggregate error and `4.740153` p99. Because an implementable causal
selector has less information than this oracle, kill the same fixed-subset
premise rather than training a selector for it.

Raw and analysis manifests hash to
`15c0cb8ab6e5058e6413efeb2a60effd200a8c5e9bc915f708fe030c4f6f4cbe`
and
`afc32798c5a474286e3eea65ccd6d32ab05f04921df1bacf5622585cad09d422`.
Gate 8 passes with 69% minimum free memory, 864,507,968-byte maximum
footprint, 1,104,789,504-byte peak RSS, zero swap growth or throttling, and
54,970,304 bytes after release. PW-0162 records zero accepted tokens, no
endpoint TPS, and no measured throughput-model constant change. It rejects
this numerical mechanism, not learned linear/recurrent attention, changed
weights, retrieval with repair, or faster future hardware.

PW-0176 closes the released MInference vertical/slash pair family on a real
MiMo 64K layer-0 slice. The strongest uniform pair, `(1000,6096)`, fits the
two-P100 complete-system allowance at `20.599935%` effective work but yields
`0.055171` aggregate relative L2, `0.884388` maximum position error, and
`0.723112` head-query p99, versus `0.010000`, `0.020000`, and `0.050000`
limits. Its final-question band passes aggregate error at `0.008556`, while
early and interval bands fail at `0.258773` and `0.030050`; late-context
success cannot substitute for the mandatory complete slice.

The strictly more favorable noncausal best-pair-per-head-query oracle also
fails at `0.047658` aggregate error, `0.721474` maximum position error, and
`0.435570` p99. Every fixed layer/head map over these five released pairs is a
restriction of that oracle, so kill all such combinations rather than fitting
a map. This supersedes PW-0175's numerical-promotion premise for the released
pair family, not for trained/repaired selectors with different widths or
mechanisms.

The two final source walks are numerically byte-identical. Fixture, raw, and
analysis manifests hash to
`d7c45847e2106a0ce5161a6e35fb87160888ea0eeebadf73b7040130ecd12526`,
`1d6c4b4fd607fee439b170da0e26e4a9f1c380231a6baa47b009a7fd0061c9a9`,
and
`3176fed9199aba3d30ac1916d96ce1b8d5b55fbb005561b16b769873097da0da`.
Gate 8 passes at 70% minimum free memory, 790,664,192-byte maximum footprint,
815,284,224-byte peak RSS, zero swap growth or throttling, and 23,102,976
bytes after release. PW-0176 records zero accepted tokens, no endpoint TPS,
and no throughput-model constant change.

PW-0177 separates compressed expert arithmetic from acquisition topology on
the onboard M1. Core ML executes a real row-scaled vector-code expert at
1.4222 ms warm median and 2.2138 ms p95 from a 13,140,830-byte package, versus
1.3651 ms and 2.3983 ms for the 50,364,779-byte FP16 control. Thus compressed
resident arithmetic is viable and does not inherently require dense expansion
or a warm latency penalty on this substrate.

The exact untrained four-effective-bit rule is not viable: validation relative
L2 is `0.159577` and maximum-row relative L2 is `0.180525`, while the FP16
control passes at `0.036504` and `0.046242`. The earlier implication that a
directly executable vector code might supply enough fidelity merely by
changing the lookup geometry is superseded for this fitting rule. Training,
activation-aware assignments, or a different program family remains distinct.

Acquisition is an independent rejection. The vector package takes 503.257 ms
to load and 7.109 ms for its first prediction. Kill route-time per-expert Core
ML package switching even if a future codebook repairs fidelity; 376 expert
executions per accepted second allow only about 2.66 ms each for the entire
expert transaction. Any continuing Core ML branch must be a resident shared
multi-expert transaction and must not preload the full routed bank. The
content-addressed report hashes to
`911f1db4b0c7d3f0af068a1f55acc78c8a7b3993ae3cea228bee91adc1ad756c`.
PW-0177 records zero accepted tokens and changes no throughput-model constant.

PW-0178 shows that changing vector orientation does not by itself solve the
low-rate fidelity problem. Input-subvector codebooks have a compelling exact
ledger—6,291,456 code bytes/expert, 2,365,587,456 code bytes for 376 expert
executions, and 246,415,360 resident codebook bytes across 47 layers—but the
favorable private layer-46/expert-28 oracle reaches `0.207785` validation
relative L2 and `0.240740` maximum-row error. Gate and up already fail at
`0.119019` and `0.084431`; training error remains `0.191608`.

Private per-expert 256-centroid input-group codebooks are a capacity upper
bound on layer-shared codebooks at the same two-index-bit rate. Kill that
single-codebook family rather than building shared fitting or a packed kernel.
The physical dataflow remains useful only if a separately charged low-rank or
multi-codebook residual, or recovery training, closes the large error without
erasing the traffic advantage. The report hashes to
`1311a8ced8ea4d376229efc9e1508542e5023d41b8f9cec546fcaab3548ac559`.
PW-0178 leaves the holdout sealed, records zero accepted tokens, and changes no
measured throughput-model constant.

PW-0179 rejects the hypothesis that PW-0178's two-bit error is a compact
low-rank residual. Rank 96—the last point inside the frozen 75%-of-INT4 and
8%-extra-MAC envelope—still yields `0.198209` complete-expert validation L2.
Diagnostic rank 128 reaches only `0.196296`, captures just
`0.232786/0.231847/0.401065` of gate/up/down residual energy, and already grows
to `0.823529` of affine-INT4 bytes with `0.09375` of source MACs.

The residual is broadly distributed rather than a few missing directions.
Kill low-rank weight repair on this two-bit core; more rank gives back the
traffic and compute needed for 1 TPS. The report hashes to
`afbf05fde482f234f2bf6f19176cdf363d25835ae282407f3d39436a9fe9d4df`.
Only a non-low-rank trained representation remains distinct. PW-0179 keeps the
holdout sealed, accepts zero tokens, and changes no throughput-model constant.

PW-0180 rejects the last cheap trained repair on the two-bit subvector core.
Continuous centroid optimization is not blocked by scalar rounding: its F32
train objective falls `55.875%`. Yet F16-staged frozen validation worsens to
`0.340665` complete-expert L2 and `0.445062` maximum-row error; gate/up reach
`1.234581/0.742788`. The optimization memorizes the narrow routed trace and
does not learn a transferable codebook rule.

Kill fixed-index centroid-only recovery and do not scale it to shared experts
or tune its schedule on visible validation. The report hashes to
`cbc780d45eda7be74c50955019b62514e0a5b885cb1c16d0796f85dbfa3a80c3`.
PW-0180 keeps the holdout sealed, accepts zero tokens, and changes no measured
throughput-model constant.

PW-0181 closes the existing-M1 one-TPS frontier without a success claim. The
15.206-ms warm one-barrier routed layer would total 0.714682 seconds across 47
layers, but sustained residency is impossible. PW-0104's impossible 8 GiB
offline-Belady oracle still misses 39.962569% of exact expert accesses. Applied
to PW-0108's measured 2.727590-second acquisition, misses alone take 1.090015
seconds. Perfectly overlap all warm MoE compute and add only the promoted
0.131220-second attention subtotal: the lower bound is 1.221235 seconds, or at
most 0.818843 TPS, before every remaining endpoint operation. A causal cache
and physically admissible footprint are worse.

The new lossy escape paths fail independently: PW-0177 at `0.159577`, PW-0178
at `0.207785`, PW-0179 rank 96 at `0.198209`, and PW-0180 after training at
`0.340665` validation relative L2. More bits/rank/private state gives back the
traffic advantage; per-expert Core ML loading is 510.365 ms; wide speculation
cannot overcome route union. With sidecar probes excluded and every cheap
training prerequisite failed, no evidence-backed onboard hypothesis remains.
PW-0181 records zero accepted tokens, no endpoint TPS, and no throughput-model
constant change.

PW-0182 tests the changed premise supplied by MLX 0.31.2's directly executable
microscaling formats. MXFP4 exactly matches the 13,369,344-byte INT4 envelope
and executes a real deep expert in `0.587479` ms warm median, but complete
validation relative L2 is `0.193978`. NVFP4 reaches `0.167407` at 1.0588 times
INT4 bytes; affine group-32 reaches `0.134476` at 1.1765 times. Fast compressed
arithmetic is real, but none of these four-bit number systems supplies MiMo's
missing fidelity. The report hashes to
`db62501ba622bb09a18db327c06cc883ab51ec978836f6d0dc703ab72ebbf485`.

PW-0183 rejects the hypothesis that spending extra bits only on the down
projection repairs that failure. `3/3/6` gate/up/down exactly matches INT4
bytes and runs in `0.597708` ms, but complete validation error is `0.255800`;
gate/up already fail at `0.163618/0.113376`. Even `4/4/8` remains at
`0.123779` while growing to 1.313725 times INT4 bytes. The report hashes to
`38a6ac68ce858bd0e7e06ffa8a31974ea625e6cf1648d554801d2dc289506fb0`.
Keep mixed precision available only with a different learned representation,
not as projection-only bit allocation. Zero tokens are accepted and no
throughput constant changes.

PW-0184 attacks the assumption that every selected expert must read every
source-weight column. At the minimum useful 25% per-token column sparsity,
weight-aware scoring yields `0.108212` complete-expert error and `0.131193`
maximum-row error; magnitude-only scoring is slightly worse at `0.110157`.
The best gate/up errors are still `0.064557/0.046195`, and 40--50% sparsity
degrades monotonically. The current TEAL/WiSparse activation-sparsity premise
therefore does not transfer to this deep routed MiMo control at the required
fidelity. Kill direct channel deletion before storage or kernel work. The
report hashes to
`6bd4a396d9c4139bf6a60c1c920ae8ff7169040857b093399b97b815350798d7`.
Zero tokens are accepted and no
throughput constant changes.

PW-0185 rejects exact prompt lookup as the one-TPS escape on PW-0112's
authenticated text suffix. The strongest deliberately permissive minimum-one-
token, `q=4` rule commits 137 target tokens in 126 passes (`A=1.087302`), with
maximum `A=4`. Even at impossible `U=1`, PW-0181's miss term remains
`1.002496` seconds per accepted token before attention, verification overhead,
or any remaining endpoint work. Minimum n-gram two falls to `A=1.037879`; four
or higher accepts no draft token. The report hashes to
`be1709777b2b2402c0419c917f010389fcffa0dc1fa9169ac6a72ec26a20a2d6`.
This closes history repetition on the trace, not target-generated Jacobi or
lookahead candidates. Zero executed tokens and no throughput constant change.

PW-0186 supplies the first positive changed premise after PW-0181. PW-0102's
authenticated posterior is shifted into a second Jacobi block without a draft
model. Exact source verification accepts `A=3` at mean normalized expert union
`U=2.268617`, giving `A/U=1.322392`. The frozen favorable physical expression
is `0.868016` seconds per accepted token (`1.152053` TPS) before dense work, LM
head, correction, rollback, or a real wide Metal transaction. This is a
continuation gate, not endpoint throughput.

The posterior is `[13,15,13,15,15,15,15,264]`; its manifest hashes to
`f773fa2859f08b57f851944aa8ba0ef9b502040058580a9344be4ce3ee1e1d1c`.
The CPU source oracle takes 274.337 seconds post-prefill. Gate 8 passes with
59% minimum free memory, 4,022,747,136-byte peak RSS, 208,292,928-byte maximum
physical footprint, zero swap growth or throttling, and stable services.
Promote one third/convergence iteration; change no measured throughput constant.

PW-0187 strengthens that changed premise enough to promote physical
integration. A third authenticated Jacobi iteration proposes
`[264,13,15,13,15,15,15,15]`, produces posterior
`[13,15,13,15,481,13,15,15]`, and accepts `A=5` at mean normalized expert
union `U=2.050532`. Thus `A/U=2.438392`; PW-0181's frozen favorable expression
falls to `0.473266` seconds per accepted token, or `2.112976` TPS before all
omitted work. This is not endpoint throughput, but it rejects the previous
belief that route union necessarily destroys target-generated wide decoding
on the authenticated suffix. The manifest hashes to
`a1066fafa979b923f9c2f5d259ff85b2f3d5aa2e77400e8b7075a48f3fa67950`.
Promote a production-shaped wide Metal verifier using original-checkpoint
page-rounded no-copy bindings; do not build the approximately 303 GB repacked
bank and do not change a measured throughput constant yet.

PW-0188 validates that physical binding premise on the real checkpoint. The
layer-46 expert-28 gate tensor begins 14,712 bytes into a 16,384-byte page; a
page-rounded read-only mapping binds through Metal with that explicit buffer
offset, zero source-copy bytes, and exact GPU/CPU samples at the first, middle,
and final tensor bytes. Mapping overhead is only 16,384 bytes over the
8,388,608-byte tensor. The report hashes to
`9e9bfd44287ab2c74df915d6c242320145387366f34755e7dcdd918f30ae4a7a`.
This supersedes the assumption that a page-aligned approximately 303 GB
repacked expert bank is required for Metal no-copy execution. Promote a real
FP8 projection parity test through the original shard mapping; the byte-sample
timing is diagnostic and changes no throughput constant.

PW-0189 prevents physical success from silently changing numerical semantics.
The first direct-checkpoint real FP8 projection fails PW-0101's source-BF16
authority at `0.00617527` relative L2 and `0.0678575` maximum error because the
current Metal kernel omits source dynamic activation FP8 quantization and BF16
output staging. This is consistent with PW-0114's explicitly modified
`metal-native-l3` result, not a new target-faithful path. Reject the projection
under the source contract; separately test no-copy versus copied/readable
equivalence only under the already named L3 semantic.

PW-0190 passes that isolated L3 physical gate. Real layer-4 expert-64 gate
weights and scales bind directly from the original shard at offsets 3,264 and
7,360 with zero copied source bytes. The unchanged Metal projection reaches
`9.13839e-7` relative L2 and `1.19209e-5` maximum error against the readable
mapped-FP8/F32 authority, with a 0.579-ms warm median. The report hashes to
`f3b2aaa099cd0c47b29efa4bfe41279ec608b6ea974528d0945112f113a70f31`.
Promote a complete direct-checkpoint expert under the named L3 semantic only;
PW-0189 remains the target-faithful rejection and no endpoint constant changes.

PW-0191 completes the no-copy expert rung without an external sidecar or
repacked artifact. Six page-rounded original-shard bindings execute the real
layer-43 expert-32 gate/up/SwiGLU/down path with zero copied source bytes and
reproduce PW-0034's output SHA exactly. Complete error is `4.69754e-7` relative
L2 and `4.45652e-11` maximum absolute; cold and warm-median command times are
2.997 and 1.078 ms. The runtime report hashes to
`f45ed1c4becbb3640948bf26d7455c8a8b8f1a3bb29e9ecc6b3ed5d1cf3f61d4`.
Promote heterogeneous original-shard scheduling in named L3 mode. Target-BF16
fidelity and endpoint accepted-token execution remain separate gates.

PW-0192 proves that direct checkpoint binding preserves the width-reuse
primitive. One real expert executes eight deterministic rows through the
shared-weight GEMM8 transaction with zero copied source bytes, `1.62608e-6`
complete relative L2, and a 2.104-ms warm median (0.263 ms/position, 3.881x
over PW-0034 batch one). The corrected report hashes to
`0471f6932abecd830da7cc42ac3da05345d67b249f6937d3cabae85d52a8eb24`
and reports zero executed/accepted tokens and `A=0`; the eight rows are not
misreported as acceptance. Promote a heterogeneous PW-0187-route scheduling
probe in named L3 mode, not an endpoint constant.

PW-0193 crosses from shared-expert width to the real PW-0187 layer-43 union.
Seventeen original-shard experts and all 64 routed placements reproduce an
independent weighted-mixture authority at `1.49824e-6` relative L2 with zero
copied source bytes. Warm layer wall is 32.906 ms, but the cause is now
specific: fixed batch eight executes 136 rows for 64 placements, or 112.5%
padding overhead. The report hashes to
`cf86d431140848bb090eac05e0ad2309c0fb61bed3b0ca071f3cdd1cc3818e6a`.
Promote a count-aware 1--8-row shared-weight kernel. The APFS device authority
changed uniformly while size, inode, nanosecond mtime, and pinned hashes stayed
fixed; record the `16777233 -> 16777231` transition and fail closed on a
nonuniform transition.

PW-0194 rejects runtime count-awareness despite perfect causal isolation. It
executes exactly 64 rows, preserves PW-0193's output SHA, and passes its active-
count fixture at `1.67638e-8`, but warm wall regresses from 32.906 to 62.600 ms.
The report hashes to
`6c7a22ce209fc6ac429d21daab3691cf584d8ea3b5cf26932aef83bc47a5493e`.
The changed premise is compiler specialization: test eight compile-time widths
so Metal can unroll position loops; do not retain the dynamic-count kernel.

PW-0195 validates compiler specialization as the missing condition. Widths
one through eight all pass their correctness fixtures; the real route union is
byte-identical to PW-0193 while warm wall falls to 19.745 ms, 1.667x faster than
fixed batch eight and 3.170x faster than runtime counts. The report hashes to
`bfbfed78d13ad80b47a6dc1cedefea3fdb9ce7ef2ff8add015f071283a0a0450`.
Promote compile-time width selection by authenticated placement count across
the direct-checkpoint L3 verifier. A layer-43-only warm extrapolation is about
0.1866 MoE seconds per PW-0187 accepted token; it is diagnostic, not endpoint
TPS or source-fidelity evidence.

PW-0196 reverses PW-0189's broad numerical rejection. Exact readable dynamic
group-128 activation quantize/dequantize before the unchanged direct Metal
projection and BF16 staging afterward reproduce PW-0101 byte for byte. The
report hashes to
`f7cc290c12293de07f3061e15747054e7cdec75f96313b8465e5fcacf4352b6a`.
The source-weight reduction topology is therefore not independently blocking
on that real projection; promote GPU-resident semantic adapters, not the
adapter-excluding timing.

PW-0197 then rejects the first complete wide source-semantic composition. Its
BF16 staging fixture passes, all 64 placements execute, and maximum output
error is only `2.38419e-7`, but relative L2 is `0.00272864` against the frozen
`2e-5` gate because the routed authority has a very small norm. Do not hide this
behind absolute tolerance or promote its timing. The next falsifier should use
the finite BF16 domain to replace backend-dependent SwiGLU transcendental
evaluation with an authenticated lookup, then remeasure the unchanged output
gate before blaming batched projection reduction.

PW-0198 performs that finite-domain isolation and rejects the hypothesis. A
256-KiB BF16 SiLU table eliminates Metal transcendental evaluation and passes
an exact CPU fixture, yet produces the exact same rejected output SHA as
PW-0197. The mismatch is therefore upstream in projection arithmetic on this
fixture. Test a bounded family of parallel reduction widths before designing a
new accumulator; do not retain the lookup as a correctness repair.

PW-0199 rejects the remaining ordinary tree-width family. Sixteen lanes is the
best of 16/32/64/128/256 but still reaches only `0.00263641` relative L2; the
others cluster near `0.00271--0.00273`, all with the same `2.38419e-7` maximum
error. Stop tuning power-of-two lane counts. A future exact local experiment
must change the accumulation representation (for example a bounded exact or
source-calibrated accumulator), while the endpoint branch must separately test
whether this cancellation-sensitive synthetic fixture predicts whole-model
token identity under the already frozen endpoint gates.

PW-0200 narrows that cancellation mismatch to nine BF16 values among 65,536
real projection outputs (`0.0137%`); 15 of 24 projections are byte-exact and
19 pass the unchanged gate. PW-0201 then rejects the intuitive repair: a
correctly rounded float64 dot matches Metal at eight of those nine sites, not
the source BLAS result. Source fidelity is an association property, not a
mathematical-accuracy property. PW-0202 closes the conservative sparse repair
escape because its fail-closed forward-error certificate flags `65.7455%` of
all rows despite capturing all nine misses. Preserve the source-BLAS sparse
shape observation, but do not promote it without a new non-calibrated
certificate.

PW-0203 reverses the belief that PW-0197's cancellation-sensitive layer gate
predicts endpoint failure. A complete eight-position verifier using the same
named L3 projection reductions reproduces the frozen posterior
`[13,15,13,15,481,13,15,15]` in every cold and warm run, with exact `A=5`.
Endpoint behavioral gates, not an ill-conditioned synthetic mixture norm, are
the authority for this acknowledged arithmetic reordering.

PW-0203 also replaces component extrapolation with a measured complete-path
constant. Direct-checkpoint Metal MoE alone leaves warm throughput at
`0.06422` accepted TPS. Extending direct Metal execution through attention,
dense layer zero, and the eight-row LM head reaches `0.09302`. Removing
duplicate per-request FP8 scans and moving heavyweight OS monitoring outside
model time reaches `0.21985` warm accepted TPS, 22.743 seconds per target
verification, with 27,508,178,944 physical bytes read. The new best verifier
is a real `3.42x` gain over the first PW-0203 variant, but still misses one TPS
by `4.5486x`.

The final bottleneck is physical embodiment rather than ordinary kernel
tuning. Correctly labeled run 004's 47 MoE transactions take 10.435 seconds while reading
19.812 GB of page-rounded expert regions, and the complete transaction reads
27.508 GB. The 16 GiB host cannot retain that exact working set; PW-0108 can
improve acquisition topology only partially, PW-0181's impossible 8 GiB
Belady premise is already below one TPS before the full spine, and the tested
four-bit executable representations fail held-out fidelity. Therefore reject
the current `q=8` source-FP8/internal-SSD branch for one TPS. A future reopening
requires a genuinely changed premise—held-out-passing executable-byte
reduction or a MiMo-specific long-horizon proposer—not another lane count,
content scan, safety-placement tweak, or hardware sidecar probe.

PW-0204 supersedes the assumption that PW-0203's frozen-block token identity is
enough to promote its modified-L3 arithmetic into arbitrary-prompt generation.
The first real run completed all six prefill chunks and six repeated
proposal/verification/rollback transactions, but its 32 committed tokens
decoded as repetitive punctuation and fragments rather than a coherent answer.
The report and progress-log SHA-256 values are respectively
`7a6674f5946a195cc58732c4b9acae322a3b6e4dacc802833dab58c86d85b266`
and `78732c76be24c76e4dcf8d3cc0c7789a7ebf10b599f8bf7aae2f061e40691119`.
Reject the arbitrary-text modified-L3 branch as a usable endpoint while
retaining its successful causal-path and accounting evidence. This also
corrects its semantic name: source weights and routes do not make reordered
Metal reductions target-faithful. PW-0092's slow source path is not an assumed
repair because its fixed Hello continuation already diverges sharply from the
hosted behavior. Localize the unresolved behavioral semantic before another
full generation run. No throughput-model constants change because the run
failed output quality and establishes no accepted TPS.

PW-0205 finds that the old whole-model “source parity” claim shared a QKV layout
mistake with its Python oracle. MiMo's raw QKV tensors concatenate four
tensor-parallel `[Q,K,V]` shards; they are not global `[all Q,all K,all V]`.
The old 108-scale-row audit consumed every scale but assigned raw shard rows to
the wrong semantic heads, and sliding-window tensors hid the same error behind
an ordinary-looking aligned grid. This supersedes PW-0089 through PW-0095 as
behavioral authorities until their captures are regenerated with deinterleaved
QKV; their evidence remains valid only for the explicitly named old layout.

Block-scaled FP8 association by itself did not repair behavior: partial and
complete block-scaled controls still chose `.`. After deinterleaving both
global and sliding-window QKV according to the pinned SGLang loader, the same
arbitrary-prompt first-token probe chooses token 30092 (`Sun`). The corrected
report and progress SHA-256 values are
`01bedf3b1028b7b66ad92ab9c0662f62507c4734c8d8f8a06b147ea30785b63b`
and `336301452aea0e2e301b2a31f208384e5c8ae4bbd9e0015f0103c0a370d27879`.
This is the first plausible local language-bearing token, but not yet a
coherence result. Require a bounded phrase and then 32--64 committed tokens.
No throughput constant changes because these are correctness probes.

PW-0205 run 007 proves that corrected QKV produces fluent 32-token text, but a
post-run invariant audit rejects its repeated cache transition. On convergence,
seven new suffix tokens are observable and the last suffix remains the next
unevaluated anchor; retaining all eight proposal input rows duplicates that
anchor in hidden history. Proposer/verifier agreement cannot expose a shared
cache error. This supersedes PW-0204's converged-retention rule and the cache
portion of run 007 while preserving its behavioral evidence. Schema 2 must
separate verifier-authorized tokens/rows from the final output-limit slice and
retain only seven rows for a fully converged width-eight proposal. Repeat the
full milestone after this correction; do not report run 007 as accepted TPS.

PW-0205 run 008 validates the corrected schema-2 cache transition across five
transactions: verifier-authorized, observable, and retained rows remain
distinct, and the final cache length agrees with the observable output. Its
report and progress SHA-256 values are
`ccafb4374e98626cae5027f95b517d0a5b6e59f2747dba0ce7bdd81fd9dc3ff9`
and `b848a930d0678d75c96383de5950102b503cf165f9dd9b9699c4b585afe3654a`.
The coherent output is nevertheless capped mid-sentence at exactly 32 tokens.
This supersedes the assumption that the lower bound is also a suitable fixed
quality boundary. Endpoint runs now declare 32 tokens as a minimum, accept a
caller maximum through 64, and stop after a second completed sentence. This
changes no target-faithful throughput constant: PW-0205 remains explicitly
SGLang-directed modified arithmetic, and the next full run must establish its
actual output count and complete-path rate.

PW-0205 run 009 passes that full modified-mode endpoint gate. A clean commit
produces 47 coherent verifier-committed tokens and stops at its predeclared
second completed sentence. The report and progress SHA-256 values are
`c87f2a12809c1accc52fc5d5092765ad4cb90cb9d1fa0a2f916a2ccb6d23e1b9` and
`9a51a914eff401050f24310c743af6443d32bea4916a3a958b4b016cb1f8dadb`.
Complete wall is 1,790,267.803 ms and the measured complete-path rate is only
0.026253 tokens/s; this is provenance for the accepted endpoint run, not a new
target throughput constant. Per-pass `A` is `[7,3,7,7,7,7,7,1]`; the final
short retention proves that the sentence boundary does not leave hidden cache
rows. Promote the arbitrary-text causal path and corrected QKV/cache semantics
within the explicitly named SGLang-directed mode. Preserve target-faithful
arithmetic, hosted parity, multimodal coverage, and 50 TPS as unpassed gates.

PW-0206's first corrected source-faithful audit proves the old-layout proposer
authorities are non-portable rather than merely suspect. On the frozen Hello
prompt, corrected QKV changes the greedy token from 264 to 9707, changes all
1,269 routed positions across all 47 MoE layers, and then produces local tokens
`[9707,0]` (`Hello!`), exactly matching the frozen hosted fixture. The corrected
prefix and two-token decode hash to
`0002c617c5459d7531de99e779ecad7335afc1e6f86cbbb6071afa23da107807`
and `f405225ea063bf3bfaf38a450fe752dc32c5afe54f69f5803c3ae61308caab2d`.

This also supersedes PW-0103's rejection of native MTP on that trace. Pairing
the corrected layer-47 states with target transition `9707 -> 0` makes MTP
layer zero propose token 0 at rank one instead of placing the stale target at
rank 175. The corrected MTP manifest hashes to
`07233ee71f194c887d96aac2cb341239df1728a7e05fc36f692a1188c65b3379`.
Reopen native MTP through PW-0208's cost-aware chained test; do not infer
multi-token acceptance or TPS from one correct proposal.

The corrected exported-mask DFlash control independently reverses PW-0150's
first-suffix rejection. Its exact block is `[9707,0,0,0,0,0,0,0]`, so the first
speculative suffix token matches the corrected target transition `9707 -> 0`.
The clean manifest hashes to
`e5084e606349fb9fe0b01f8e5505f43fa58969cae5398330b343f40dab7228c9`.
Reopen DFlash only through exact width-eight target verification: draft logits
alone establish neither accepted length nor route-union leverage. PW-0206 still
owes the corrected width-eight `A/U` authority, so PW-0203's physical bound is
not yet transferable. No throughput-model constant changes at this stage.

The exact corrected DFlash target walk preserves a smaller but genuine gain:
formal acceptance doubles from stale PW-0102's `A=1` to `A=2`. Its posterior
is `[0,2585,2585,2585,2585,2585,2585,2585]`, but eight-position routed union
rises to `U=2.686170`, leaving `A/U=0.744554`. The manifest hashes to
`edf677be8406bd663e0d99b67c8cfb12fdad3914a100dbfd31f9d92b4787693e`.
Reject the raw block as expert-byte leverage because it remains below one;
preserve the doubled acceptance and feed the exact posterior into the corrected
Jacobi successor test rather than erasing the sub-threshold improvement.

Corrected Jacobi iteration two improves that preserved signal without yet
crossing the physical gate. Posterior feedback raises `A` from 2 to 3 and
`A/U` from `0.744554` to `0.947103` even as `U` rises to `3.167553`. The clean
manifest hashes to
`dd53f80c02418d4d0321b400a47c1a88bcc70cf72626570fb5302266e6cf39cf`.
Retain the 27.204% leverage improvement and authorize iteration three because
the predeclared convergence falsifier improved; do not promote iteration two
while `A/U` remains below one.

Corrected Jacobi iteration three crosses the minimum expert-byte leverage gate,
but only narrowly. It commits `A=4` across `U=3.702128`, so `A/U=1.080460`;
the clean manifest hashes to
`cf9403b441b9453557d9c6fb2481d0dd361e319efbd6dad2c4e21d5c424ed3d1`.
Promote this 45.115% improvement over raw corrected DFlash as a real
lower-milestone mechanism. Do not promote it as a 50-TPS architecture: it is
only 14.313% of the otherwise-free INT4 requirement and its measured
post-prefill diagnostic is 0.009733 accepted token/s. PW-0206 is complete;
carry native MTP to PW-0208 and the small Jacobi leverage into later combined
economics without erasing either the gain or its limit.

PW-0207's corrected first-transaction route trace closes the missing identity
gap in the high-residency hypothesis. Its report hashes to
`e5c0b93d039ec8d8c6b1f7a0087ec3991ba55df2a1cee7d388f08d6e668d830b`
and exactly reproduces PW-0205 run 009 transaction zero while retaining every
proposal and verifier expert identity.

The byte-accurate offline falsifier hashes to
`ee9f71b83ca427bd1a98d166ada77c778c53d81157dfa6ccb071afada54e73eb`.
A legal 12,878,375,808-byte static set predicts only 1.791485× fewer physical
reads, so the 4× byte hypothesis fails. The same set predicts 3.262043× less
attributed acquisition wall from a positive two-phase measured solve, passing
the alternative 2× continuation gate. Authorize pressure-safe implementation;
do not activate the 13 GiB ceiling or claim TPS until warning eviction,
critical stop, exact transaction parity, and real interleaved speedup pass.

The original 12,878,375,808-byte PW-0207 total is exact source data but is
superseded as a physical allocation constant. Page-aligned ownership on this
16 KiB host requires 12,882,755,584 bytes for the same 592 objects, leaving
2,146,304 bytes in the 12 GiB declaration. The corrected clean manifest hashes
to `1dedbef7c79aa23835d194f52760a1f2c65dcca1481bd6df2d5602615c3fdad6`.
Because the selected objects are unchanged, the 1.791485× byte and 3.262043×
wall predictions remain unchanged. Use the corrected allocation total for the
resident implementation and retain the older number only for source-byte
provenance.

PW-0207 now has an executable pressure-safety substrate, but not yet a
high-residency runtime. Clean commit
`c10cc1e0df23efc69e3e66521e7b57a445bf13d4` validates the 592-object authority,
owns payload lifetime, drains a dedicated Darwin pressure queue, evicts warning
payloads in the declared total order, and permanently rejects growth after
critical pressure. The corrected synthetic report hashes to
`199107b541670a915fba5a17b5ef9cc2c139309e1e81e476692161161867e6a2`.
It changes the old belief that no pressure callback exists, but it does not
change the 8 GiB default: injected events over an 8-byte fixture do not prove
real OS pressure response, 12 GiB safety, transaction acceleration, or TPS.

The preceding synthetic artifact hashes to
`758d895c28b253fcc1b0567de53d9cdb4812eef183eac9d675c72a3bcdbf6e52`
and is rejected because its supplied implementation commit did not match live
HEAD. Preserve it as the reason the evidence command now authenticates exact
HEAD and requires a clean worktree.

PW-0207's page-aligned authority now reaches real checkpoint bytes for one
complete expert bundle. Clean report
`84ed0150b8f20868b4ae4fa143ddf6e7536f08a00e96b2c1f4215a9629f7d942`
copies all six `expert:14:162` tensors exactly from 25,171,968 source bytes into
25,182,208 declared resident bytes, then an injected warning evicts the owned
mapping to zero. The pilot stays far below the old ceiling with 140,266,368
bytes maximum physical footprint, 70% minimum free memory, and no swap or
throttling growth. Promote the backing/loader mechanism to decoder integration;
do not infer full-set safety, real pressure response, transaction speedup, or
TPS from a one-object copy pilot.

PW-0207's first decoder-integrated resident object preserves exact execution
and a small positive signal without satisfying promotion. Clean commit
`53483cd1171cb4f4076d83d6310764bfdf4b813b` substitutes all six tensors of
`expert:14:162` for eight transaction accesses, accounting 201,375,744
resident source bytes, and evicts the 25,182,208-byte mapping to zero on
warning. In the A–B–A sequence, resident transaction times are 189,434.333 and
186,885.288 ms around a 189,201.042 ms mapped control; their 188,159.811 ms
median is 0.550331% faster. Preserve the gain and the first candidate's 0.123%
slowdown together: this is a working exact substitution mechanism worth
bounded scaling, not repeatable 2x evidence or permission to raise the 8 GiB
default.

PW-0207 bounded scaling preserves another small gain and closes the naive 3 GiB
growth path under current loading. A 3 GiB ranked request stopped at the eighth
object on the unchanged 4 GiB post-phase footprint gate; do not weaken it. The
seven-object 1,652,555,776-byte prefix passes exact execution and pressure-safe
eviction. Its A–B–A candidate median is 186,182.679 ms versus 188,266.418 ms
mapped, improving committed transaction TPS 1.106803% from 0.037181 to
0.037597, but the second candidate is individually 0.480% slower. Preserve the
median gain without promotion. The resident ledger also exposes a concrete
remaining gap: proposal-side one-row LM-head execution bypasses residency, so
only 4,471,128,064 of 13,220,446,208 predicted bytes are substituted. Repair
that causal path before judging the ranked set, and use a non-double-resident
loader before attempting more than seven objects.

Closing the proposal LM-head gap produces a much larger gain than residency.
Clean commit `677217ac17bc23a3de2975c9e98b6dea3c491b86` moves the seven
one-row proposal LM-head calls onto the existing exact wide BF16 Metal path.
Mapped transaction median falls from 188,266.418 to 170,647.894 ms, a repeatable
9.358294% wall reduction, and committed TPS rises 10.324490% from 0.037181 to
0.041020. Preserve and promote this lower milestone even though it is far from
50 TPS.

The repaired seven-object residency test supersedes its partial-path speed
signal for promotion. It substitutes the full exact 13,220,446,208 bytes, but
two resident transactions have a 171,186.580 ms median versus 170,647.894 ms
mapped: 0.315672% slower and 0.314678% lower TPS. Reject the seven-object cache
as a performance default, retain its correctness/pressure machinery, and keep
only direct-to-owned loading open as the route to test a meaningfully larger
prefix without weakening the 8 GiB/4 GiB limits.

Direct-to-owned loading reopens bounded residency and preserves a repeatable
gain. Clean commit `54ab1f0c1db68b3b925b076b9e4c54bf88fd1150` admits 30
ranked objects and 3,196,059,648 bytes under the unchanged 4 GiB release gate,
then substitutes exactly 25,568,477,184 transaction bytes. Candidate median
163,955.518 ms versus 170,329.600 ms mapped improves wall 3.742205% and
committed TPS 3.887690%, from 0.041097 to 0.042695. Promote the loader and keep
the lower milestone; do not promote endpoint residency because the 2x gate
still fails. Peak RSS stays at 3,529,293,824 bytes with 60% free memory and no
swap/throttling, leaving only a narrow final larger-prefix falsifier—not a path
to weaken limits or allocate the 12 GiB offline set.

PW-0207 closes with a larger repeatable lower milestone, not its 2x target.
The final 42-object, 4,001,366,016-byte prefix substitutes exactly
32,010,928,128 bytes per transaction. Candidate median 159,978.184 ms versus
169,277.593 ms mapped improves wall 5.493585% and committed TPS 5.812923%, from
0.041352 to 0.043756. Maximum install footprint 4,146,854,912 bytes stays under
the unchanged 4 GiB release gate; maximum peak 4,330,487,808 stays under 8 GiB,
with 57% free memory and no swap/throttling. Preserve and carry this conditional
gain forward, but close the high-residency branch: it misses the predeclared 2x
gate, cannot authorize endpoint promotion, and has no meaningful remaining
prefix within the required 256 MiB reserve.

PW-0208's corrected 32-window corpus kills the native-MTP cost-aware
expert-byte hypothesis more strongly than an observed draft failure. The
complete-history manifest hashes to
`a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b`.
Corrected Jacobi commits `A=213` across 53,251 exact layer/expert units. Even an
omniscient q=4 proposer reaches only 0.721493x its accepted-token/unique-byte
rate; perfect q=8 and a per-window q oracle reach at most 1.051643x. The clean
upper-bound report hashes to
`3aaca59be0e000cac77d5a36b8e3b9d2e2fc5bbb02792c8846dae3da16747f8c`.
Reject PW-0208's 2x cost gate without spending compute on real proposals.

Do not supersede PW-0206's corrected native-MTP first-token recovery or infer
that native MTP has no value. PW-0208 bounds verifier expert bytes, not the
roughly 140-second same-model proposal wall. Preserve native MTP as PW-0211's
separately named proposal-latency lower milestone; any repeatable positive
complete TPS gain remains valuable even though the 2x and 50-TPS gates fail.

AN-0001 reviews three independent project analyses and adopts their strongest
research rule: optimize only a crossing on the measured cold critical cut,
price its impossible-best enclosing-path effect before implementation, and
require a fixture on which the candidate causal explanations disagree. This
makes PW-0210 conditional: applying its full 1.5x hypothesis to PW-0111's
8.383 ms GPU interval predicts only a 1.026x cold-layer gain. PW-0211 remains
first because it can delete seven measured same-model proposal transactions;
PW-0212 then tests bounded corrected-route predictive prefetch before runtime
work. Missing 50 TPS does not erase any smaller repeatable complete-path gain.

The same audit propagates PW-0205/PW-0206 farther. PW-0112's route trace and
PW-0116's activation corpus explicitly use the old QKV implementation, so
their `U`, cache, route-persistence, activation-weighted fidelity, and
PW-0186/PW-0187 proposal-economics values remain authentic only for that named
old layout. Layout-independent byte, device, hardware, and kernel results
survive. PW-0208 provides corrected text route/byte authority for 32 balanced
windows, not the multimodal, long-context, or final representation holdout.
Do not close corrected-layout representation or prefetch branches by silently
importing PW-0112/PW-0116 values.

PW-0211's clean last-row native-MTP reference reproduces PW-0206 more strongly
than token agreement: all 152,576 logits are bit-identical to the full-row
oracle. On the first ordinary, code, multilingual, and rare-route PW-0208
windows, newly committed q4 endpoint lengths are `[3,3,1,3]`. The verification
helper's full-convergence prefix includes the already-known anchor and must not
be used directly as endpoint `A`. The first, second, and fourth native blocks
match every available three-token draft position; multilingual rejects
immediately and commits one verifier correction token. Complete CPU-reference walls are only
8.70--10.44 seconds versus 138.83--157.26 seconds for the seven same-model q8
proposal steps, but they include authority work and are not endpoint TPS.
Promote the branch to one real q4 verifier timing before spending all 32 CPU
references. Preserve the three positive slices even if the combined schedule
later loses.

PW-0211's real q4 replay tightens the ordinary model without extrapolating
verifier width. At the exact transaction-one cache state, the same native block
and target posterior converge over a four-token prefix including the known
anchor, so endpoint accounting is `A=3`, with `U=5.377660` and 26.286 seconds of
real verifier wall. Composed with the 10.444-second CPU native reference, the
diagnostic candidate is 0.081676 accepted TPS versus 0.039288 for the matched
q8 control, or about 2.079x. Reject the earlier 0.108902-TPS/2.771859x
expression because it counted the anchor as newly accepted output. This
authorizes live-cache integration; it is not yet accepted TPS because the two
walls came from separate processes.

PW-0211's first live-cache native-MTP integration preserves the corrected
positive result end to end. It emits exactly seven verifier-authorized tokens;
transaction one commits three tokens in 35.571 seconds, or 0.084338 TPS,
versus 0.039288 TPS for the matched q8 control, a measured 2.146654x gain.
Complete request wall is 332.965 seconds versus 447.480 seconds for the
same-output q4 control, a 1.343927x gain including prefill. This is one
candidate/control observation, not a repeatable default promotion. Preserve it
as a positive lower milestone and run an interleaved matched control plus a
second candidate before deciding promotion.

PW-0211's cold candidate-control-candidate sequence promotes that ordinary-text
result to a conditional lower milestone. The two native q4 complete walls are
332.965 and 328.675 seconds (1.297% apart), for median 0.021160 accepted TPS;
the interleaved q8 same-output control is 616.664 seconds and 0.011351 TPS. The
repeatable complete-request gain is 1.864046x. Post-prefill wall improves
4.847533x. Transaction-one accepted TPS improves 2.250573x after correctly
accounting for native `A=3` versus control `A=7`; do not report the 5.251x wall
ratio as acceptance-normalized TPS. All runs emit identical seven tokens, and
swap growth, throttling, and protected-service loss remain zero. This changes
the ordinary Apple-M1 performance belief and throughput constants, but not the
general default: other categories, contexts, and holdouts still require live
full-path evidence.

AN-0002 audits the owner-supplied DS4 review against pinned commit
`84cc882352757baf628a1776badf7cc54d584e28`. DS4 genuinely strengthens the
shape prior for PW-0210: it fuses routed gate/up reduction through SwiGLU and
uses an F16 materialized intermediate when the down consumer's MMA path loads
half. It does not change PW-0210's 1.026x impossible-best Prismwing cold-layer
bound, so execution remains conditional. DS4 issue 437's page-aligned
`F_NOCACHE`, disabled automatic read-ahead, owned double-buffer, and trailing-
drop result is a measured prototype rather than mainline authority; preserve it
as PW-0213's exact transport falsifier. Confidence-gated speculation motivates
PW-0214's corrected-route cost-adaptive `q` oracle without importing DS4
thresholds. A local audit also closes the suspected accepted-token replay bug:
Prismwing already retains verifier-produced K/V and final-hidden rows and only
truncates rejected suffix state. The review changes research branches, not a
throughput constant or runtime default.

PW-0212 closes corrected-route runtime prefetch under its frozen physical gate
without erasing the route signal. Across 16 held-out PW-0208 text windows, all
6,016 events discriminate among controls. Last-route recall is 38.370595% and
calibration-frozen category-frequency recall is 32.415642%, but neither is a
causal hidden-latency result in the current batched verifier. The only
one-layer-ahead control recalls 2.944232%. More decisively, an impossible
future oracle constrained to a 25% traffic tax can hide only 1.616833% of
complete transaction wall, versus the predeclared 10% gate (7.988956% of
verifier wall). Report
`2365033116e194b6bac34d2017f644c3499c5fb92a3727f7db9162dce318587f`
therefore rejects runtime implementation for this endpoint. Retain the logical
signals for residency and changed-cut research, and keep PW-0213 open because
uncached transport changes how demanded bytes are acquired rather than
depending on prediction.

PW-0213's isolated raw-checkpoint transport passes its file-backed gate while
failing its acquisition-speed gate. Cacheable `pread` leaves every probed
source page resident; page-aligned `F_NOCACHE` with `F_RDAHEAD=0` leaves none,
a repeatable 100% reduction at 1.003905x full-layer read amplification. The
two-buffer uncached layer median is 72.191416 ms, 4.523759% faster than the
75.611917 ms sequential uncached path but 2.808891% slower than the 70.219040
ms cacheable control. The one-expert two-buffer result is only 0.241140%
slower than control. Preserve the recovered overlap and control regression
together. This authorizes one bounded verifier pilot based on page-cache
topology, not a speed claim; trailing drop is causally redundant because
`mincore` already sees zero source pages after uncached reads. Raw report
`51b2898314ff42ecca0eb7e29802f23346a329244126cd566dea80da9171f17f`
and analysis
`764fba9b12d8bacc5d4d2cd7f1fc57a42323a94bc90717bc41fa948842803fe3`
carry no endpoint TPS.
