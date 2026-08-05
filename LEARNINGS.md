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
