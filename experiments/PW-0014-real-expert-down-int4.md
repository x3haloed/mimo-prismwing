# PW-0014 — Real expert-down INT4

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `5be7007`; dirty generalized benchmark
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; source shard
  `fd89388271eac237e06ace68a832156357b42f85820856afee24da7bb36d9dcc`;
  fixture `40e74ac15c976ae92469320c233e26422db94c0b8b003aa30e4f8f57a17720b1`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2; warm application
  buffers; no material memory pressure
- Related records: PW-0012, PW-0013

## Hypothesis and mechanism

PW-0012 assumed that eight real expert down projections, whose matrix shape is
32,768×2,048 when concatenated, perform like the equal-byte 16,384×4,096 MTP
gate/up proxy. Measure the actual shape and bytes to validate or replace that
third of the routed compute estimate.

## Contract

Explicit L3 affine INT4. Use the actual layer-43/expert-32 FP8 down tensor and
scale grid. Concatenate eight copies only to reproduce the top-eight production
shape; this is a performance fixture, not a claim that eight experts share
weights. Run batch one and eight with ten warm-ups and 30 measurements, twice.
Source loading and installation quantization are excluded and named.

Correctness requires a committed actual-tensor packed fixture and independent
Rust affine-INT4 decode. Report quantization error separately from kernel error.

## Baseline and candidate

Baseline is PW-0012's 16,384×4,096 real MTP projection at a 2.6810 ms mean
batch-eight median. Candidate is eight concatenated copies of the actual
4,096×2,048 down projection, producing shape 32,768×2,048 with the same
35,651,584 executable bytes and the same arithmetic count.

## Isolated attribution

| Repeat | Batch 1 wall median ms | Batch 8 wall median ms |
| --- | ---: | ---: |
| 1 | 0.8649 | 2.6340 |
| 2 | 0.9510 | 2.5984 |

Mean batch-eight median is 2.6162 ms and approximately 408 GFLOP/s. The actual
down shape is 2.4% faster than the gate/up-shaped proxy, so the proxy was
conservative but materially accurate.

Using two PW-0012 gate/up proxy times and one actual down time per layer:

```text
47-layer routed time = (2 × 2.681031 + 2.616219) ms × 47 = 374.979 ms
routed-only accepted TPS at perfect A=8, U=1          = 21.335
```

## End-to-end result

No fused expert or endpoint TPS is claimed. Eight identical copies measure the
production down shape and arithmetic but do not exercise per-position expert
gather, gate weighting, SwiGLU, storage misses, or actual route union.

## Correctness result

The independent Rust scalar decoder matches the committed actual down packed
fixture. Full-shape MLX output stays within `7.63e-6` of its four-row MLX
fixture.

Affine INT4 changed the four real down outputs by 15.51% relative L2 on the
deterministic synthetic input. Cancellation makes this a harsh and narrow
measure, not a layer-quality prediction, but it strengthens the requirement
for real-activation layer/logit gates before any L3 promotion.

Raw evidence SHA-256:

- repeat 1: `f80e8c8ef4543fde2e4c3c7bf837f8779bd1a2274bf3501812ab3a7e63774fa9`
- repeat 2: `14c9d220ea31bc01b655aaf24ef1f03bfaaa2caaeb05b957044fc35149e26c00`

External evidence root:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0014`.

## Decision

Retain the actual down shape in the routed compute model and retain MLX affine
INT4 only as a conditional L3 candidate. The MTP proxy did not hide a major
down-projection slowdown. Wait for the paired real gate/up tensors, then run a
complete real expert with SwiGLU and actual intermediate/output error.
