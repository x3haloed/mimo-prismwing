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
  `aten/src/ATen/cpu/vec/functional_base.h`, ARM
  `aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h`, and
  `aten/src/ATen/native/TopKImpl.h` at that commit. PW-0070 additionally uses
  the four-part reduced-precision dot reduction in
  `aten/src/ATen/native/cpu/BlasKernel.cpp` at the same commit. PW-0072 also
  uses the specialized vector-dot topology in
  `aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp`.
- Build identity: CPU capability `DEFAULT`,
  `AT_BUILD_ARM_VEC256_WITH_SLEEF`, Apple aarch64.
- Decision: operation-order reference for PW-0061 through PW-0066 and PW-0070.
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
