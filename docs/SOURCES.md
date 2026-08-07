# Source ledger

Every source is pinned to the revision used for a decision. A link to a moving
branch is convenient navigation, not identity evidence.

## MiMo-V2.5

- Repository: `XiaomiMiMo/MiMo-V2.5`
- Hugging Face revision:
  `63651580ca774f8504f676040460aed3e1244ac1`
- Verified SSD source root:
  `/Users/chad/Models/mimo-prismwing/checkpoints/MiMo-V2.5-63651580`
- Machine-readable file identity: `spec/model.lock.json`
- Semantics: pinned `configuration_mimo_v2.py`, `modeling_mimo_v2.py`,
  tokenizer, processor configuration, and chat templates.
- Decision: authoritative source checkpoint and component semantics.

The upstream revision contains 39 files totaling 315,714,053,402 bytes. Its 18
safetensors files use upstream LFS SHA-256 identities. The complete local
verification manifest hashes to
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
and binds all 39 files to this revision.

## PyTorch CPU semantic reference

- Installed version: 2.13.0
- Build commit: `cf30153c4c131c8164ee7798e5022d810682e2cb`
- Authoritative files: `aten/src/ATen/native/cpu/SoftMaxKernel.cpp`,
  `aten/src/ATen/native/cpu/SumKernel.cpp`,
  `aten/src/ATen/cpu/vec/functional_base.h`, ARM
  `aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h`, and
  `aten/src/ATen/native/TopKImpl.h` at that commit. PW-0070 additionally uses
  the four-part reduced-precision dot reduction in
  `aten/src/ATen/native/cpu/BlasKernel.cpp` at the same commit. PW-0072 also
  uses the specialized vector-dot topology in
  `aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp`;
  PW-0075 and PW-0076 apply its vector-tail topology to attention-value dots.
- Build identity: CPU capability `DEFAULT`,
  `AT_BUILD_ARM_VEC256_WITH_SLEEF`, Apple aarch64.
- Decision: operation-order reference for PW-0061 through PW-0066, PW-0070,
  PW-0072, PW-0073, PW-0075, PW-0076, PW-0078, PW-0079, PW-0081, and
  PW-0082. PW-0085 corrects PW-0066's horizontal-sum interpretation using the
  same pinned `vaddvq_f32` specialization: low/high lane pairs reduce before
  the final addition. PW-0088 distinguishes generic four-part BF16 GEMM for
  probability-vector-by-value-matrix accumulation from specialized contiguous
  BF16 dots, superseding PW-0076's non-discriminating operation inference.
  PW-0091 confirms the LM-head oracle widens BF16 operands to F32 and uses the
  one-row matrix path before BF16 output rounding; standalone BF16 dot behavior
  is diagnostic only there.
  It is not a runtime dependency or an authority for MiMo model topology.

## OpenRouter MiMo-V2.5

- Model slug: `xiaomi/mimo-v2.5`
- Initial provider: Parasail, individually pinned with fallbacks disabled.
- Discovery date: 2026-08-04.
- Decision: only external whole-model reference; PW-0001 proves scoreable text,
  image, multi-image, audio, video, and mixed requests.
- Limitation: provider implementation and future availability are external and
  require frozen endpoint metadata plus drift canaries.

## MiMo-V2.5-DFlash

- Repository: `XiaomiMiMo/MiMo-V2.5-DFlash`
- Hugging Face revision:
  `1f58446181abcaa01030fdbde835fbd38ae9a2b1`
- Local immutable metadata/draft root:
  `/Volumes/Elements/mimo-prismwing/checkpoints/MiMo-V2.5-DFlash-1f584461`
- Machine-readable file identity: `spec/dflash-model.lock.json`
- Decision: candidate draft semantics and weights only; its bundled target is
  not the authoritative Prismwing checkpoint.

PW-0009 proves by 48 deterministic remote payload samples that every bundled
target shard differs from the pinned base target. The target tensor map and
payload sizes match, so this is not a topology change, but published DFlash
acceptance cannot be assumed for the base revision. The source verifier is
target-preserving for greedy decoding; its positive-temperature sampling path
lacks the correction needed for an L2 distribution-preservation claim.

## SGLang DFlash runtime

- Repository: `sgl-project/sglang`
- Revision: `2fc557254b3aaf539e80266e52a6d1e1f8da9980`
- Local source root:
  `/Volumes/Elements/mimo-prismwing/research-sources/sglang-dflash`
- Machine-readable source identity: `spec/sglang-dflash.lock.json`
- Decision: executable semantic authority for the DFlash draft path named by
  Xiaomi's later DFlash deployment instructions; not a Mac runtime dependency.

The pinned SGLang model defines full-head Qwen3 RoPE, unscaled values, no
attention sink in the draft attention softmax, target embedding/LM head reuse,
a masked width-eight block, and greedy logits from positions one through seven.
Its loader silently ignores checkpoint weights it does not register. PW-0102
found an important disagreement in the nominal Hugging Face path: Transformers
4.57.6 consumes the exported `partial_rotary_factor=0.5`, creates 64-wide
rotary factors, and the published wrapper then applies them to 128-wide heads,
failing dimensionally in its first layer. SGLang explicitly sets
`rotary_dim=head_dim` and is the named deployment runtime, so the bounded Mac
reference adapter must normalize only that factor to full-head RoPE and record
the modification. The five `attention_sink_bias` tensors and nested
`attention_value_scale` remain exported-but-unused; partial RoPE does not.

## SGLang MiMo-V2 MTP runtime

- Repository: `sgl-project/sglang`
- Revision: `2fc557254b3aaf539e80266e52a6d1e1f8da9980`
- Local source root:
  `/Volumes/Elements/mimo-prismwing/research-sources/sglang-dflash`
- Machine-readable source identity: `spec/sglang-mimo-mtp.lock.json`
- Decision: executable semantic authority for PW-0103's native MTP transition;
  not a Mac runtime dependency.

The pinned runtime maps one selected `model.mtp.layers.N` payload into a single
draft model. It normalizes a shifted base-token embedding and the target hidden
state separately, concatenates and projects them with `eh_proj`, runs one dense
MiMo SWA decoder block, applies the MTP final norm, and reuses the base
embedding and LM head. Multi-layer execution creates one runner per draft step;
PW-0103 first tests only layer zero's real causal proposal.

## Atomic llama.cpp TurboQuant fork

- Repository: `AtomicBot-ai/atomic-llama-cpp-turboquant`
- Revision: `074bf826e1b06005a51737d29387e36657f41bf7`
- License: MIT
- Machine-readable source identity: `spec/atomic-turboquant.lock.json`
- External source checkout:
  `/Volumes/Elements/mimo-prismwing/research-sources/atomic-llama-cpp-turboquant`

This source is research input, not a vendored runtime or validated dependency.
Its WHT-based KV formats and Metal kernels must pass PW-0020 and the normal
correctness ladder before any implementation is promoted.

## Predecessor runtimes

- TurboFieldfare: `7a99f2a635e3adf7ed0720b882d2edb600f2f0da`
- Swiftlet: `d0cf7021b0544bf1ba4f264c592386a54bc49a00`
- Decision: architecture and workflow references only. No code has been copied
  into the Prismwing runtime at this milestone.

## MLX

- Installed version: 0.31.2
- Upstream release commit: `68cf2fd`
- Decision: independently optimized Apple-silicon quantized-matmul comparison
  and possible C++ substrate; not an authoritative MiMo semantic reference.
- Relevant API: affine `quantize`/`quantized_matmul`, group size 128, four bits.

PW-0012 measures the installed native Metal path through Python orchestration.
Any final dependency must pin and build the C++ implementation or reproduce
its kernel behavior; a mutable Python environment is not a release substrate.

## Apple-silicon Metal execution model

- Apple Developer, *Metal Compute on MacBook Pro* (accessed 2026-08-06):
  <https://developer.apple.com/videos/play/tech-talks/10580/>
- Apple Developer, `newBufferWithBytesNoCopy:length:options:deallocator:`
  (accessed 2026-08-06):
  <https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29>
- Apple Developer, *Resource loading* and `MTLIOCommandQueue` (accessed
  2026-08-06): <https://developer.apple.com/documentation/metal/resource-loading>
  and <https://developer.apple.com/documentation/metal/mtliocommandqueue>

Decision: primary substrate authority for PW-0105 and its gated successors.
Apple documents that CPU and GPU access the same physical memory through shared
resources, recommends avoiding duplicate/shadow resources, recommends
multi-buffering and larger submissions to avoid CPU/GPU stalls, and describes
Metal I/O queues loading filesystem data directly into GPU resources alongside
compute work. The no-copy API wraps an existing single-VM-region allocation and
requires a page-aligned pointer and page-aligned region length. These are
candidate capabilities, not evidence that the M1 checkpoint layout or current
Rust bindings realize them; each path still requires a real device probe and
end-to-end measurement.

## Mixture-of-Basis-Experts

- Paper: Chen et al., *MoBE: Mixture-of-Basis-Experts for Compressing MoE-based
  LLMs*, arXiv `2508.05257v1`, submitted 2025-08-07:
  <https://arxiv.org/abs/2508.05257v1>.
- Accessed: 2026-08-06.
- Decision: research input for PW-0115 and a prospective PW-0045
  routed-mixture compiler; not a Prismwing artifact or validated implementation.

MoBE factorizes each gate/up expert matrix into an expert-specific transform
and a learned combination of layer-shared basis matrices. It reports 24--30%
parameter reduction on several MoE models with small average benchmark loss,
but explicitly leaves down projections unchanged. Prismwing therefore treats
the paper as evidence that learned shared bases are plausible, not that its
published embodiment meets MiMo's much deeper executable-byte requirement.
PW-0115 derives that applicability bound from MiMo's pinned shapes before any
training or activation-corpus walk.

- Implementation: inclusionAI/MoBE commit
  `7f3501da2a9f7b12d773cb52c454a0be0ceeb185`, `train.py`, accessed
  2026-08-06: <https://github.com/inclusionAI/MoBE/blob/7f3501da2a9f7b12d773cb52c454a0be0ceeb185/train.py>.
- Decision: primary implementation authority for PW-0117's algebra audit.

The released trainer applies SiLU or tanh to each expert's softmax-weighted
basis matrix before multiplying by its expert-specific factor. That nonlinear
step is relevant to reconstruction quality but prevents basis evaluations from
being linearly shared across selected experts. PW-0117 separates the published
activated representation from Prismwing's prospective identity-activation
all-projection transaction form instead of conflating their compute claims.
