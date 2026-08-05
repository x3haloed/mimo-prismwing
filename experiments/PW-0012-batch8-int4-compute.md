# PW-0012 — Batch-eight INT4 compute

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `5d7d19c`; dirty candidates
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; source MTP
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`;
  MLX fixture `1f76479353ec7579405ae131b76b226b2dc05a056a8a4a5c658b6dd82aceaa34`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift 6.3.3; MLX 0.31.2;
  warm application buffers; MLX setup peak 169,869,312 bytes
- Related records: PW-0008, PW-0010, PW-0011

## Hypothesis and mechanism

The byte-only DFlash-8 bound assumes an expert weight loaded once can serve all
eight verifier positions cheaply. A production-shaped batch-eight projection
should reveal the arithmetic cost, and an optimized native low-bit primitive
may outperform Prismwing's transparent scalar-accumulator Metal kernel.

## Contract

L3 diagnostic on the real 16,384×4,096 MTP projection, which is exactly eight
expert-projection weight equivalents. Batch eight, concurrency one, ten MLX or
five custom-kernel warm-ups, and 30 measured runs. Source load and installation
quantization are excluded and named. Correctness requires a committed packed
fixture that an independent Rust scalar decoder can evaluate.

The custom signed-INT4 and MLX affine-INT4 modes are distinct representations
and fidelity candidates. Compare execution mechanisms, representation bytes,
and source error separately; do not infer endpoint TPS from either projection.

## Baseline and candidate

The baseline is PW-0011's direct signed-INT4 batch-one kernel. The first custom
batch-eight kernel keeps eight scalar accumulators while loading each quantized
weight once. A column-major `float4` variant attempts vector accumulation.

MLX 0.31.2 is an independently optimized native candidate. Its affine group-
128 representation packs eight unsigned nibbles per u32 with one f16 scale and
bias per group. Its 35,651,584 executable bytes equal the signed candidate's
size. MLX exposes the same primitive through a C++ API; the measured Python
entry point is orchestration, not a proposed final Python runtime.

## Isolated attribution

| Candidate | Batch | GPU/wall median ms | GFLOP/s | Decision |
| --- | ---: | ---: | ---: | --- |
| Custom signed INT4, 32 lanes | 1 | 1.0708 GPU, interleaved mean | 125.3 | prior baseline |
| Custom signed INT4, 32 lanes | 8 | 4.4230 GPU | 242.8 | selected custom batch path |
| Custom signed INT4, 64 lanes | 8 | 4.5709 GPU | 234.9 | rejected width |
| Custom signed INT4 `float4`, 32 lanes | 8 | 5.8280 GPU | 184.2 | rejected layout |
| MLX affine INT4, repeat 1 | 8 | 2.6683 wall | 402.4 | candidate |
| MLX affine INT4, repeat 2 | 8 | 2.6938 wall | 398.6 | candidate |

MLX's mean batch-eight median is 2.6810 ms, 1.65× faster than the selected
custom batch kernel. Batch-eight work is 2.70× more efficient per position than
MLX batch one, but it is not free.

A MiMo token's top-eight routed experts contain three such projection-weight
equivalents per layer. Under the optimistic assumptions that every verifier
position chooses the same experts and all three shapes perform identically,
the MLX measurement implies:

```text
target-pass routed time = 2.681031 ms × 3 × 47 = 378.025 ms
routed-only accepted TPS at A=8, U=1         = 21.163
```

This is a production-size compute diagnostic, not a strict architecture
ceiling: fused gate/up/SwiGLU/down kernels and actual down-projection shapes
could change it. Dense work and draft execution can only reduce endpoint rate.

## End-to-end result

No endpoint TPS is claimed. This experiment replaces the byte model's implicit
free-batch assumption with measured arithmetic. A complete endpoint near this
range would still be a valuable Prismwing 10 result and useful progress toward
Prismwing 25, regardless of the 50-TPS primary gate.

## Correctness result

The committed MLX affine-INT4 fixture stores real packed u32 words, f16 scales
and biases, and source-derived inputs. The dependency-free Rust oracle decodes
the affine representation and reproduces its f32 projection. Full-matrix MLX
output differs from the four-row MLX fixture by at most `3.052e-5`, attributable
to shape-dependent f16 accumulation order.

Affine INT4's four-row error versus source FP8 is 4.09% relative L2, better
than PW-0011 signed INT4's 9.84% on its closely related input, but still only a
small component slice.

Raw evidence SHA-256:

- MLX repeat 1: `4e0e20f6b3fb668be1046ee6b707834e04d603b235e9bdfee115ee46e2b49c25`
- MLX repeat 2: `457de22413cb3e087f3c49167edee311a04f258bbcd9d1e5b17f9c6832fa964f`
- custom scalar 32: `8c45b51c03a461e493b85d28d17c22b8b229dc27baf6ceeb04b459ca42670304`
- custom scalar 64: `3b7423eaa9267d8410ba27486963f9adb60a7d46ab78a2b3cce57f37fd64fb01`
- rejected vector 32: `a604f831c2b82409543ff2c15c746ac179b7d5d37e8d4e76cb6b107abaf724c5`

External evidence root:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0012`.

## Decision

Retain MLX affine INT4 as the optimized native low-bit comparison and potential
C++ substrate. Retain the readable signed kernels as correctness/performance
oracles. Reject the `float4` input layout and 64-lane custom width. Do not
promote any INT4 mode until real expert layer-local route/logit drift passes.
Measure a complete real gate/up/SwiGLU/down expert next; then determine whether
fusion can close enough of the measured compute gap to justify a full L3
runtime.
