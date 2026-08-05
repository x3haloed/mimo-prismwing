# PW-0018 — Complete-expert affine precision sweep

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `1c2d995`; dirty exploratory sweep and
  parameterized MoE benchmark
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; source tensors and routes from
  PW-0015/PW-0016
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2; all candidate buffers warm
- Related records: PW-0015, PW-0016

## Hypothesis and mechanism

The INT4 substrate is fast but has unacceptable component drift. Explore MLX
affine 5-, 6-, and 8-bit modes on the same complete real expert to find a
precision that materially reduces error without discarding the optimized
quantized-matmul substrate.

## Contract

Exploratory L3 sweep, not a predeclared promotion experiment. Compare 4, 5, 6,
and 8 bits with group size 128 on actual layer-43/expert-32 gate, up, and down
weights. Use batch eight, 10 warm-ups, 30 measurements, and rotating
interleaved order. Report source-FP8 output error and complete expert wall time.

Because numerical success thresholds were not frozen before this exploratory
measurement, this record may select a candidate for a later gate but may not
claim production promotion.

## Baseline and candidate

The source and input match PW-0015. Every candidate uses MLX affine group-128
quantization and the complete gate/up/SwiGLU/down path. Source loading and
installation quantization are excluded.

## Isolated attribution

| Bits | Executable bytes/expert | Median ms run 1 | Median ms run 2 | Relative L2 | Cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 13,369,344 | 1.4758 | 1.4794 | 15.475% | 0.987969 |
| 5 | 16,515,072 | 1.6801 | 1.8254 | 7.618% | 0.997149 |
| 6 | 19,660,800 | 1.9030 | 2.0556 | 3.781% | 0.999286 |
| 8 | 25,952,256 | 1.4963 | 1.5943 | 0.912% | 0.999958 |

Affine 8-bit is Pareto-superior to 5- and 6-bit here: lower error and lower
wall time. Its mean median is only 4.6% slower than affine 4-bit, despite nearly
doubling executable bytes.

The same 8-bit mode on PW-0016's real heterogeneous MoE block repeats at
11.1463 and 11.1127 ms versus INT4's 9.9059 and 9.7564 ms. Block relative L2
falls from 17.02% to 1.026%, while the routed-only 47-layer diagnostic changes
from 17.31 to 15.29 TPS.

## End-to-end result

No endpoint result is claimed. Affine 8-bit is not a compression of source FP8:
its per-expert executable representation is 3.1% larger because scale and bias
metadata accompany eight-bit codes. Its value is the much faster MLX kernel
substrate and substantially lower numerical drift, not reduced storage traffic.

## Correctness result

The source route sets remain exact at 8 bits. A dedicated real block fixture is
committed. The two block runs produce identical output metrics: 1.0261%
relative L2 and 0.999947 cosine versus source FP8.

Evidence SHA-256:

- sweep repeat 1: `9f3792858c4223c5884f83abacb2706d7738e3ab223d1707ab796dd292a2d2ea`
- sweep repeat 2: `51cc698089623cd6f8b295b74d57afa807c6e869d655856400eb39d6c7311066`
- INT8 block repeat 1: `aa247541aa0e99ac1e8e7c0d3d4607cfea690478e581e10474619f1b1f3fd557`
- INT8 block repeat 2: `107794d9301f478ba93956b4518d2219adfe33dd1dab26aaa4f141600e129e14`
- INT8 fixture-verified block: `ada816495a084dd747008c428a2ec94cfac2505ebe5bf4f2edb656feccb3c3a7`
- committed INT8 fixture: `740945a15b000be2ae9292809cbe411d25930bc0b0dc2b89fa9549cbc237b234`

External evidence root: `/Volumes/Elements/mimo-prismwing/evidence/PW-0018`.

## Decision

Select affine 8-bit, not 5- or 6-bit, for the next predeclared quality/performance
gate. Keep INT4 as a bandwidth-oriented research branch. Do not promote INT8
to a target embodiment until real-activation layer/logit evidence passes; a
1% component error can still accumulate materially through 47 layers.
