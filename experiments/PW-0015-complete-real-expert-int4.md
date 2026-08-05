# PW-0015 — Complete real expert INT4 path

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `5388b6c`; dirty remote extraction and fixture
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; locked shard-0 LFS SHA-256
  `879caa9e27753caa056bf53aad9f773554d6ff128c118a830de7ebc5cc5295b4`;
  locally verified shard-1 SHA-256
  `fd89388271eac237e06ace68a832156357b42f85820856afee24da7bb36d9dcc`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2; warm application buffers;
  peak MLX allocation 30,146,560 bytes
- Related records: PW-0002, PW-0012, PW-0013, PW-0014

## Hypothesis and mechanism

Execute gate, up, SwiGLU, and down together for one actual routed expert. This
tests whether separate projection timings hid a large nonlinear/composition
cost and replaces the equal-shape projection proxy with a causal expert path.

## Contract

Explicit L3 affine INT4 candidate compared with source FP8. Use all six actual
layer-43/expert-32 tensors. The gate/up payloads may be fetched as exact pinned
remote byte ranges while the 34.37 GB source shard continues downloading, but
their range offsets and payload hashes must be recorded and the remote
whole-file hash must not be described as locally verified.

Correctness requires deterministic tiny quantized-composition parity, a
committed real-expert fixture, source-FP8 output comparison, and two timed
runs. Performance success is a repeatable full-expert measurement; it is not
an endpoint TPS claim. A relative L2 error above 10% keeps the candidate behind
the real-activation layer/logit gates.

## Baseline and candidate

`remote_tensor_extract.py` losslessly materialized gate/up weights and scale
grids from `model_pp0_ep1_shard0.safetensors`. The already locally verified
`model_pp0_ep1_shard1.safetensors` supplied down weight and scales. Exact
payload hashes are committed in the fixture; the extracted safetensors artifact
is 16,781,816 bytes with SHA-256
`ca02748075edd889014c1e5beb4a2ce2abd96c1a2adebe5bd3faf278aa724276`.

The benchmark installs MLX affine INT4 group-128 weights, then times this warm
path with one process and concurrency one:

```text
gate = qmm(input, gate_weight)
up = qmm(input, up_weight)
hidden = sigmoid(gate) * gate * up
output = qmm(hidden, down_weight)
```

Input row `r`, column `c` is
`f16(sin((c + 19*r) / 17) * 0.01)`. Source loading and installation
quantization are outside the timer. Each batch has 10 warm-ups and 30 measured
runs.

## Isolated attribution

| Repeat | Batch 1 median ms | Batch 8 median ms | Batch 8 p10–p90 ms |
| --- | ---: | ---: | ---: |
| 1 | 0.5251 | 1.3445 | 1.2360–1.9520 |
| 2 | 0.5933 | 1.3124 | 1.2326–1.7615 |

Mean batch-eight median is 1.3285 ms per expert for eight positions. The three
installed projections occupy 13,369,344 bytes. SwiGLU and dispatch overhead
are included; routing, expert weighting/sum, storage installation, attention,
and logits are not.

For a deliberately conservative sequential diagnostic, perfect `A=8, U=1`
means the same eight experts serve all eight positions. Eight complete expert
calls across 47 routed layers imply 16.02 routed-only accepted TPS. This is
more causal than PW-0014's 21.33 TPS concatenated-projection proxy, but remains
neither an optimized schedule nor endpoint throughput.

## End-to-end result

No whole-layer, token, or endpoint result is claimed. This is the first
executable complete routed expert. It proves that the nonlinear composition
works and is cheap relative to its three qmatmuls, while exposing the cost of
eight separate expert calls.

## Correctness result

The deterministic tiny test compares the quantized expert with the same
composition over MLX-dequantized weights. The committed real fixture was then
verified by a fresh complete run.

The complete real expert's affine-INT4 output differs from source FP8 by 15.96%
relative L2 at batch one and 15.48% at batch eight; cosine similarities are
0.98720 and 0.98797. This is a stronger component-fidelity warning, not a
whole-model quality result.

Evidence SHA-256:

- remote extraction manifest: `df4ae1de5b32ea00f8583a97c0a21a91efb2688c89c81871ea1a48221d7225ba`
- timed repeat 1: `8a4bb4e44de58463a68eb96633df84ea693207dbbcd7c01005d2e373439ba23b`
- timed repeat 2: `3930797ebd30ff6e84a42d35f015a404622088c628ff9efd0e6ee86dc0f0d520`
- fixture-verified run: `d70d8eaebc8569c18d957e8ee40abdb2c46e7f3d34cbdd44279a91bd570ae6a8`
- committed fixture: `61e0c92c635e846567d51b785fd43830e9a7f53cc005cd42aa1d1a8281ee2676`

External roots: `/Volumes/Elements/mimo-prismwing/artifacts/PW-0015` and
`/Volumes/Elements/mimo-prismwing/evidence/PW-0015`.

## Decision

Promote the complete-expert executable and pinned range extractor as research
substrate. Retain affine INT4 only as conditional L3: its full-expert error
crosses the predeclared 10% caution threshold. Next measure real router-selected
experts/activations and implement expert weighting/sum in a complete MoE block;
do not extrapolate this synthetic expert input to target fidelity.
