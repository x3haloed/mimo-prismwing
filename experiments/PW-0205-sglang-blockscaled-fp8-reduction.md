# PW-0205 — SGLang block-scaled FP8 reduction

- Status: in progress
- Disposition: correctness-repair
- Date: 2026-08-10
- Execution mode: SGLang-directed modified arithmetic candidate
- Hardware/runtime: 16 GiB Apple M1, Metal, internal SSD checkpoint
- Related records: PW-0053, PW-0091, PW-0197 through PW-0204

## Hypothesis

PW-0204 proves a real repeated endpoint but rejects its output as incoherent.
The accelerated path quantizes activations to FP8, immediately dequantizes
them, applies the weight scale to every product, and reduces across all input
blocks with one Metal tree. The pinned SGLang fallback instead retains FP8
activation codes and per-token-group scales, computes a dot for each
128-column block, multiplies that block result by the activation and weight
scales, and accumulates scaled block results in FP32.

Although the real deployment may select DeepGEMM rather than Triton, this
block-scaled equation is explicit in the officially recommended runtime and is
a stronger behavioral candidate than the Hugging Face/Accelerate arithmetic
that is locally exact but produces unusable text.

## Contract and gates

Add a separate Metal quantization output containing exact E4M3FN codes and one
FP32 scale per token-group. Add a block-scaled projection that reduces each
128-value FP8 dot before applying its two scales, then accumulates blocks in
increasing order. Preserve the old L3 kernels and names as historical controls.

Before any full endpoint run:

1. compare quantized codes and scales with the existing PyTorch byte fixture;
2. compare the complete block-scaled Metal projection against an independent
   scalar implementation of the declared equation on multiple blocks and
   nonuniform scales;
3. run one arbitrary-prompt first-token probe with complete provenance; and
4. continue to 32--64 tokens only if the result clears a predeclared
   behavioral gate and does not violate cache, safety, or verifier authority.

This experiment may change the explicitly named modified arithmetic mode. It
may not be labeled target-faithful, tuned to a desired token, or used to weaken
the frozen source and hosted comparisons. A failed first-token probe kills the
candidate without another terabyte-scale generation walk. No isolated kernel
timing is accepted TPS.

## Partial result

The first implementation preserves exact activation codes and scales and adds
the declared 128-column block-scaled projection. Its deterministic Metal probe
matches the existing CPU/PyTorch quantization authority byte for byte and
matches an independent two-block scalar projection within `0.01` maximum
absolute error. The complete Rust library suite passes 87 tests.

Run 001 then changed only ordinary FP8 spine projections while retaining the
historical QKV and routed-MoE reductions as explicit controls. It completed the
41-token arbitrary prompt in six chunks and chose token 13 (`.`), rather than
PW-0204's token 264 (` a`). The report hashes to
`db08d69f8e471128f8fdf5981fb8558235f40c6dc7a28b2d2fae291ab53cda66`
and its progress log to
`0b9c090e8342b5d3ab73d6d13389b1a90fef56f16bfc92de4cc574ddb224560c`.
Complete wall was 155,572.482 ms, including 154,832.475 ms prefill; logical
source bytes were 190,814,088,448, process disk reads 191,044,632,576, and peak
RSS 3,922,575,360 bytes.

That lone punctuation token does not clear a behavioral gate and cannot
justify generation. It does prove the arithmetic distinction is materially
causal at whole-model output. Continue the same source-directed equation
through routed experts, then repeat the one-token gate. Do not call run 001 a
coherence result or accepted TPS.
