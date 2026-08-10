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
  PW-0157 additionally hash-binds `TopKImpl.h` to
  `1ff24ba878ccb3816511ba34609d7247225342c6aa61740b51917c8ca79407ab`
  and uses actual PyTorch 2.13.0 outputs for adversarial tied
  `topk(sorted=False)` rows. This authority is build-specific; it is not a
  promise about tie indices from another PyTorch or standard-library build.
  It is not a runtime dependency or an authority for MiMo model topology.

## OpenRouter MiMo-V2.5

- Model slug: `xiaomi/mimo-v2.5`
- Initial provider: Parasail, individually pinned with fallbacks disabled.
- Discovery date: 2026-08-04.
- Decision: only external whole-model reference; PW-0001 proves scoreable text,
  image, multi-image, audio, video, and mixed requests.
- Limitation: provider implementation and future availability are external and
  require frozen endpoint metadata plus drift canaries.
- PW-0160 metadata refresh: OpenRouter public model and endpoint APIs, captured
  2026-08-10:
  <https://openrouter.ai/api/v1/models> and
  <https://openrouter.ai/api/v1/models/xiaomi/mimo-v2.5/endpoints>.
  The raw payloads hash to
  `ce8154b4ee4ae42f5e14c071847a5acc35c9e89e69a42e84f4b8676d2cd3133e`
  and `09cd75b1b2e4d053f99e1f13be64c680ec6de0db3ab2e36642af475d1e9e9033`.
  They advertise Parasail FP8 at context 1,048,576 with `logprobs` and
  `top_logprobs`; this is moving metadata, not execution proof.
- PW-0160 provider result: three identical million-token requests reached only
  error 502 and two explicit Parasail shared-pool 429s. These preserved bodies
  establish transient provider unavailability during the bounded epoch, not
  context or logprob capability failure. No provider substitution is inferred
  or authorized.

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

## Under-$500 companion candidate authorities

- Intel, Xeon E5-2680 v2 specifications (accessed 2026-08-06):
  <https://www.intel.com/content/www/us/en/products/sku/75277/intel-xeon-processor-e52680-v2-25m-cache-2-80-ghz/specifications.html>.
- Dell, PowerEdge R720/R720xd owner's manual, technical specifications
  (accessed 2026-08-06):
  <https://www.dell.com/support/manuals/en-us/poweredge-r720/720720xdom-v3/technical-specifications>.
- Intel, *Intel 64 and IA-32 Architectures Optimization Reference Manual*,
  including the earlier-generation throughput volume (accessed 2026-08-06):
  <https://www.intel.com/content/www/us/en/developer/articles/technical/intel64-and-ia32-architectures-optimization.html>.
- NASA Advanced Supercomputing, *How to Use Sandy Bridge Processors*, AVX peak
  of 16 single-precision operations/cycle/core (accessed 2026-08-06):
  <https://www.nas.nasa.gov/hecc/assets/pdf/training/How_to_Use_Sandy_Bridge_062712.pdf>.
- eBay sold listing `168220564406`, Dell R720, dual E5-2680 v2, 512 GB RAM,
  no disk, observed sold at `$303.75` on 2026-08-06. This is a market
  feasibility observation, not an active BOM or purchase authority:
  <https://www.ebay.com/itm/168220564406>.
- NVIDIA, Tesla P40 data sheet (accessed 2026-08-06):
  <https://www.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf>.
- eBay listing `116590232547`, used Tesla P40 card-only, observed at `$249.99`
  with three available on 2026-08-06; accessories are excluded:
  <https://www.ebay.com/itm/116590232547>.
- NVIDIA, Tesla M40 24-GB data sheet (accessed 2026-08-06):
  <https://images.nvidia.com/content/tesla/pdf/78071_Tesla_M40_24GB_Print_Datasheet_LR.PDF>.
- NVIDIA, legacy CUDA GPU compute capability table, identifying Tesla M40 as
  compute capability 5.2 (accessed 2026-08-06):
  <https://developer.nvidia.com/cuda/gpus/legacy>.
- NVIDIA, CUDA toolkit/driver/architecture matrix, listing CUDA 12.x and R580
  as the last toolkit and driver families for Maxwell (accessed 2026-08-06):
  <https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html>.
- NVIDIA, CUDA 13.0 release notes, removal of Maxwell offline compilation and
  library support (accessed 2026-08-06):
  <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html>.
- Dell, PowerEdge R720 GPU card installation guidelines (accessed 2026-08-06):
  <https://www.dell.com/support/manuals/en-us/poweredge-r720/720720xdom-v3/gpu-card-installation-guidelines>.
- GPUDojo, Tesla M40 24-GB market tracker, observed used-from price of `$150`
  at 12:39 UTC on 2026-08-06. This affiliate market observation is not a
  complete BOM or purchase authority:
  <https://gpudojo.com/tesla-m40>.

Decision: source authority for PW-0127's pre-purchase ceiling, not evidence of
measured MiMo performance or a complete purchasable BOM. Intel specifies ten
cores, 3.60-GHz maximum turbo, AVX, 115-W TDP, and 59.7 GB/s maximum memory
bandwidth per E5-2680 v2. Dell supports two E5-2600-v2 CPUs and up to 512 GB
RDIMM in the R720. The sold server example shows that capacity has appeared
below the project cap; it cannot satisfy PW-0048's dated active-BOM or measured
stage gates. NVIDIA specifies 24 GB, 12 FP32 TFLOP/s, 47 INT8 TOPS, 346 GB/s,
PCIe 3.0 x16, and 250 W for the P40. The representative server-plus-current-
P40 prices already total `$553.74` before storage, adapters, or cooling, so
that pair is not an eligible BOM.

Decision: additional source authority for PW-0128's legacy-accelerator
pre-purchase ceiling. NVIDIA specifies 24 GB, up to 7 FP32 TFLOP/s, 288 GB/s,
PCIe 3.0 x16, passive cooling, and 250 W for the M40. Dell's supported R720 GPU
configuration requires the GPU enablement kit, redundant 1100-W supplies, two
CPUs at no more than 115 W each, and a 30 C maximum inlet; these requirements
belong in the complete BOM and operational power/thermal checks. Maxwell is a
legacy compute-capability-5.2 target whose supported build environment must be
pinned to CUDA 12.x or earlier.

## Low-bit weight-calibration and exception-store authorities

- Dettmers et al., *SpQR: A Sparse-Quantized Representation for Near-Lossless
  LLM Weight Compression*, arXiv `2306.03078`, accessed 2026-08-06:
  <https://arxiv.org/abs/2306.03078>.
- Lin et al., *AWQ: Activation-aware Weight Quantization for On-Device LLM
  Compression and Acceleration*, MLSys 2024, accessed 2026-08-06:
  <https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf>.
- Official AWQ implementation, `mit-han-lab/llm-awq`, `auto_scale.py`, accessed
  2026-08-06:
  <https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py>.
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers*, arXiv `2210.17323`, accessed 2026-08-06:
  <https://arxiv.org/abs/2210.17323>.
- Official GPTQ implementation, `IST-DASLab/gptq`, commit
  `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`, `gptq.py`, accessed 2026-08-07:
  <https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py>.
- Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*,
  arXiv `2404.00456`, accessed 2026-08-06:
  <https://arxiv.org/abs/2404.00456>.
- Official QuaRot implementation, `spcl/QuaRot`, commit
  `5008669b08c1f11f9b64d52d16fddd47ca754c5a`, rotation utilities and
  randomized-Hadamard implementation, accessed 2026-08-07:
  <https://github.com/spcl/QuaRot/tree/5008669b08c1f11f9b64d52d16fddd47ca754c5a>.

Decision: design authorities for PW-0133 and successor weight-domain branches,
not MiMo fidelity or performance evidence. SpQR establishes the causal value of
isolating sensitivity outliers in a sparse high-precision representation. AWQ
uses calibration activations and a bounded scale search to protect salient
weight channels without backpropagation; its official code minimizes module
output error over candidate activation-derived scales. GPTQ uses approximate
second-order information to update remaining weights while quantizing, and
QuaRot applies function-preserving rotations to remove outliers before low-bit
execution. PW-0133 deliberately tests the smaller mechanism first: exact
source-FP8 exceptions selected by a train-only diagonal activation-weighted
error proxy. None of the published quality or speed results transfer to MiMo's
dynamic source-FP8 MoE without the repository's routed-activation and endpoint
gates.

The official implementation is the mechanism authority for PW-0137's
cross-group distinction. It forms one Hessian across the full input dimension,
uses an inverse-Cholesky factor, applies column updates inside a bounded block,
and then multiplies the block's accumulated error into every remaining column.
With activation order plus static quantization groups, it selects each grid by
the column's original pre-permutation group. This is design authority only;
PW-0137 must establish MiMo fidelity on the repository's routed activations.

The official QuaRot implementation is the mechanism authority for PW-0141's
fixed residual-basis test. It constructs a randomized orthogonal Hadamard
matrix, right-multiplies MLP gate/up weights, and left-multiplies the down
projection so a globally rotated residual stream preserves the unquantized
model. PW-0141 tests only that folded residual rotation; it does not transfer
QuaRot's Llama quality results or claim a complete MiMo rotation.

## Explicit routed-expert I/O and slot-ownership authorities

- TurboFieldfare commit `3249be40a33ea6560b35531c184609d7be67ac1a`,
  `VerifiedInstallReceipt.swift`, `PreadExpertStreamer.swift`, system design,
  expert-I/O summary, and optimization journey; accessed 2026-08-07:
  <https://github.com/drumih/turbo-fieldfare/tree/3249be40a33ea6560b35531c184609d7be67ac1a>.
- Swiftlet commit `02f8101e1a9671f93f1f3d3a31926344d751a6e2`,
  `Qpack.swift`, `ExpertCache.swift`, `QwenMetalModel.swift`, and
  `SwiftletSession.swift`; accessed 2026-08-07:
  <https://github.com/leonickson1/Swiftlet/tree/02f8101e1a9671f93f1f3d3a31926344d751a6e2>.

Decision: primary implementation and measurement authorities for PW-0136 and
any conditional successor, not transferred MiMo performance evidence.
TurboFieldfare binds a trusted installation receipt to manifest, source
revision, sizes, and hashes; maps common weights read-only; and fills bounded,
page-aligned, Metal-wrapped routed slots through explicit `pread`. Its current
planner reserves hit and avoided slots, marks a miss slot invalid before I/O,
and publishes its expert identity only after all concurrent reads succeed.
Swiftlet independently stores all projection material at fixed offsets within
one fixed-stride expert blob and performs one `pread` per miss.

TurboFieldfare's measured cold expert read is 9.88 ms through mmap versus 2.79
ms through `pread`; its simulator reports about 0.50 versus 3.97 tok/s. Its
coarse resident/hit/read-miss/join schedule outperformed per-completion expert
launching, and persistent independent row-claiming workgroups outperformed an
eight-task cooperative kernel. Swiftlet defers memory-pressure cache shrink
until between tokens and carries a completed layer's MoE as pending work into
the next command buffer. These mechanisms motivate explicit slot ownership and
pending-MoE orchestration only after PW-0136 clears the real MiMo acquisition
bound.

Both repositories also preserve important negative evidence: increased cache
hit rate need not improve decode, argument-buffer reuse can reduce allocations
while slowing the workload, and monolithic fusion can reduce throughput.
Prismwing therefore retains end-to-end promotion gates and does not infer that
fewer reads, buffers, or dispatches are sufficient.

## Owned EPYC companion-envelope authorities

- AMD, EPYC 7351P specifications, accessed 2026-08-09:
  <https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7001-series/amd-epyc-7351p.html>.
- Supermicro, H11SSL-i product specifications and motherboard manual, accessed
  2026-08-09:
  <https://www.supermicro.com/en/products/motherboard/H11SSL-i> and
  <https://www.supermicro.com/manuals/motherboard/EPYC7000/MNL-2085.pdf>.
- EVGA, SuperNOVA NEX750B manual, product 120-PB-0750, accessed 2026-08-09:
  <https://www.evga.com/support/manuals/files/120-PB-0750.pdf>.
- NVIDIA, Tesla P40, P100 PCIe, and V100 official specifications, accessed
  2026-08-09:
  <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf>,
  <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf>, and
  <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/v100-application-performance-guide.pdf>.
- eBay item 188207963486, used P100 PCIe 16-GB listing, observed `$74.37`
  each at quantity two on 2026-08-09; and item 336284264679, new single-M.2
  PCIe adapter, observed `$12.80` each on 2026-08-09. These moving listings
  are price observations, not hardware validation, a complete BOM, or purchase
  authority:
  <https://www.ebay.com/itm/188207963486> and
  <https://www.ebay.com/itm/336284264679>.

Decision: source authority for PW-0151's authenticated pre-purchase envelope.
AMD specifies 16 cores, 2.9-GHz maximum boost, 155/170-W TDP, PCIe 3.0 x128,
and eight memory channels. Supermicro specifies three x16, three x8, and one
native M.2 x4 slot. EVGA specifies 750 W continuous at 50 C and 61 A/732 W
combined across four 20-A +12-V rails, with the exact VGA rail map used by the
report. NVIDIA specifies the advertised compute peaks and 250-W passive PCIe
forms. None of these nameplates establishes achieved MiMo throughput, SSD
sustained reads, physical fit, cable compatibility, cooling, wall power, or
fidelity.

## Wide-proposer acceptance authority

- Chen, Liang, and Liu, "DFlash: Block Diffusion for Flash Speculative
  Decoding," arXiv:2602.06036v2, camera-ready ICML 2026 version, accessed
  2026-08-09: <https://arxiv.org/abs/2602.06036v2>.

Decision: semantic and empirical authority for PW-0152's DFlash-specific
analysis. The paper defines expected accepted tokens per cycle as
`tau in [1, gamma+1]`, including the target bonus token; trains each block from
a clean target-produced bonus anchor; uses block size 16 in its main
experiments; and reports a maximum Table 6 `tau=6.33`. These facts constrain
the published block embodiment, not every possible future long-depth proposer.

## Owned EPYC resident-bank authorities

- Supermicro, H11SSL-i motherboard manual, revision 1.2a, accessed and captured
  2026-08-09:
  <https://www.supermicro.com/manuals/motherboard/EPYC7000/MNL-2085.pdf>.
- Newegg product page for Hynix `HMAA8GL7MMR4N-UH`, 64-GB DDR4-2400 ECC
  LRDIMM, sold and shipped by A-Tech, observed at `$247.19` on 2026-08-09:
  <https://www.newegg.com/hynix-64gb/p/0RN-000W-003B5>.

Decision: source authority for PW-0153's physical population and one dated
procurement falsification, respectively. The manual supports eight memory
channels/slots, 64-GiB module populations up to 512 GiB, and fewer than eight
populated channels while recommending a balanced bank; it also requires the
same DIMM type, size, and speed. The listing is an immutable captured price
observation for one compatible LRDIMM candidate, not a market-wide lower
bound, compatibility warranty, complete BOM, or purchase authority.

## Owned EPYC installable-BOM authorities

- NVIDIA, Tesla P100 PCIe GPU Accelerator product brief
  `PB-08248-001_v01`, accessed and captured 2026-08-09:
  <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NV-tesla-p100-pcie-PB-08248-001-v01.pdf>.
- Supermicro H11SSL-i manual and PW-0151's authenticated EVGA NEX750B photo
  and manual, as listed above.
- Dated moving observations for eBay items 188207963486 (two used P100s),
  386086936012 (four 256-GB NVMe candidates), 336060640814 (quad-M.2 passive
  carrier), 405322665251 (claimed `030-0571-000` dongles), and 313527520178
  (P100 cooling kits), accessed 2026-08-09:
  <https://www.ebay.com/itm/188207963486>,
  <https://www.ebay.com/itm/386086936012>,
  <https://www.ebay.com/itm/336060640814>,
  <https://www.ebay.com/itm/405322665251>, and
  <https://www.ebay.com/itm/313527520178>.

Decision: primary physical authority for PW-0155's P100 connector and power
ledger. NVIDIA specifies 250-W total graphics power, a CPU-style 8-pin input,
up to 240 W/20 A on that auxiliary rail, and dongle NVPN `030-0571-000` fed by
two PCIe cables. The market rows are a structured transcription rather than
immutable seller evidence; one SSD row is internally inconsistent and the
dongles are unbranded. They support rejecting the captured list as purchase
authority, not selecting or buying those parts.

## Twelve-GB Ampere envelope authorities

- NVIDIA, *NVIDIA Ampere GA102 GPU Architecture*, captured 2026-08-10:
  <https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf>.
- NVIDIA, GeForce RTX 3080/3080 Ti specifications, including the 12-GB RTX
  3080 variant, accessed and captured 2026-08-10:
  <https://www.nvidia.com/zh-tw/geforce/graphics-cards/30-series/rtx-3080-3080ti/>.
- Dated moving observations for eBay items 298525951637 (active used RTX 3080
  12 GB), 206381378508 (sold RTX 3080 12 GB), 386086936012 (three ambiguous
  256-GB NVMe candidates), and 336284264679 (three single-drive adapters),
  accessed 2026-08-10:
  <https://www.ebay.com/itm/298525951637>,
  <https://www.ebay.com/itm/206381378508>,
  <https://www.ebay.com/itm/386086936012>, and
  <https://www.ebay.com/itm/336284264679>.

Decision: technical and moving-market authority for PW-0159. NVIDIA binds the
12-GB card to 8,960 CUDA cores, 1.71-GHz boost, Ampere third-generation Tensor
Cores, and 12 GB GDDR6X; the GA102 whitepaper distinguishes dense BF16 with
FP32 accumulation from dense FP16 with FP16 accumulation and structured-sparse
figures. The market rows are a dated structured transcription, not immutable
seller evidence or purchase authority. They support rejecting the captured
active BOM and defining a reopening price, not inferring future availability,
device health, delivered identity, sustained I/O, fit, or tax.

## Thirty-two-GB Volta envelope authorities

- NVIDIA, V100 product page with V100 PCIe and V100S PCIe specifications,
  accessed and captured 2026-08-10:
  <https://www.nvidia.com/en-sg/data-center/v100/>.
- NVIDIA, *Tesla V100 PCIe GPU Accelerator Product Brief*,
  `PB-08744-001_v03`, accessed and captured 2026-08-10:
  <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/Tesla-V100-PCIe-Product-Brief.pdf>.
- Dated semantic transcriptions for active eBay items 188322612403 (used V100
  PCIe 32 GB) and 198372771385 (used Dell V100S PCIe 32 GB), observed
  2026-08-10 after direct page capture returned HTTP 403:
  <https://www.ebay.com/itm/188322612403> and
  <https://www.ebay.com/itm/198372771385>.

The official HTML hashes to
`39557823ad6871fbfe5afd7d572d5192c754e27feef90c8cc092e562f59b4f4d`;
the product brief hashes to
`7e2a80764520d744ae146ec276655a6359ecd2bcd83feaba802cb29efcedadee`;
and the explicitly weaker market transcription hashes to
`c7f378c65bd2c24633ccce238f0dcaffc1731de8f670a34721d0fb50cc3c010c`.

Decision: technical and moving-market authority for PW-0161. NVIDIA binds the
standard V100 PCIe to 112 TFLOPS deep-learning arithmetic and 900 GB/s HBM,
and V100S PCIe to 130 TFLOPS, 32 GB HBM2, and 1,134 GB/s. The product brief
binds the 250-W passive dual-slot form, CPU 8-pin auxiliary input, and dongle
NVPN `030-0571-000`. The market rows support rejecting two dated card-only
ledgers, not inferring future prices, delivered identity, device health,
cooling, fit, tax, or purchase authority.

## Thirty-two-GB CDNA envelope authorities

- AMD, *AMD Instinct MI100 Accelerator* product page and product brochure,
  accessed and captured 2026-08-10:
  <https://www.amd.com/en/products/accelerators/instinct/mi100.html> and
  <https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/product-briefs/instinct-mi100-brochure.pdf>.
- AMD, ROCm 7.1 Linux system requirements, accessed and captured 2026-08-10:
  <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/reference/system-requirements.html>.
- Dated semantic transcription for active eBay item 285796378466, a used MI100
  32-GB card observed 2026-08-10 after direct archival fetch returned HTTP 403:
  <https://www.ebay.com/itm/285796378466>.

The official product page hashes to
`9d0b74dc18ac8afcced3a9efbca17f77f6fd4b148c6823c11a5f479d8f9cbcc6`;
the brochure hashes to
`ad383b0c0d2bcb8c719ddcf09ed5a4d7a0afeb901c3b51bb2490fa3e65e6dc2e`;
the ROCm requirements hash to
`dead3ad053cde897c83aa33d58f096b1a9b25878abbe82dbcd5206c3f86d3772`;
and the explicitly weaker market transcription hashes to
`bfdc3bcd99685518f810f4d7f5caaa7d6563511cc1f62dfed3f317f2d0bd9022`.

Decision: technical, software-support, and moving-market authority for
PW-0163. AMD binds MI100 to 92.3-TFLOPS BF16, 184.6-TFLOPS FP16, 32 GB HBM2,
1.2-TB/s nameplate bandwidth, PCIe 3/4 x16, a passive full-height dual-slot
form, and 300-W peak power. ROCm 7.1's supported MI100 distributions exclude
the owned host's current Debian 13 installation. The market row supports
rejecting one dated card-only ledger, not inferring future prices, delivered
identity, device health, cooling, fit, tax, or purchase authority.

## Affordable Blackwell envelope authorities

- NVIDIA, GeForce RTX 5060 family specifications, accessed and captured
  2026-08-10:
  <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/>.
- NVIDIA, *NVIDIA RTX Blackwell GPU Architecture*, accessed and captured
  2026-08-10:
  <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>.
- NVIDIA, *NVIDIA Blackwell GeForce RTX Arrives for Every Gamer, Starting at
  $299*, official launch release captured 2026-08-10:
  <https://nvidianews.nvidia.com/_gallery/download_pdf/67fe58ae3d63325f115ecd52/>.
- Dated semantic transcription of NVIDIA Marketplace's out-of-stock GIGABYTE
  WindForce RTX 5060 Ti 16-GB page, observed 2026-08-10 after direct archival
  retrieval timed out:
  <https://marketplace.nvidia.com/en-us/consumer/graphics-cards/gigabyte-windforce-geforce-rtx-5060-ti-gv-n506twf2-16gd/>.

The product page hashes to
`238d00c79c20939e5208e1a6507a6949e00e21ab3c3c3cc79d25b97eb0af20fd`;
the architecture whitepaper hashes to
`906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa`;
the launch release hashes to
`76ca4fce0315435079d72f3725174b704b9b8990b3be7d89591471333a418394`;
and the explicitly weaker market transcription hashes to
`98400749a4ca60351ff71b0450bf545ee542691051e783426c2c183219774cf6`.

Decision: technical and moving-market authority for PW-0164. NVIDIA binds RTX
5060 Ti to 4,608 CUDA cores, 2.57-GHz boost, 16/8-GB GDDR7 forms, 180-W total
graphics power, and one 8-pin or qualifying Gen-5 power cable. The same-
generation architecture table distinguishes dense BF16/FP32-accumulate and
FP16/FP16-accumulate rates from sparsity-enhanced values. The launch release
binds the 16-GB MSRP to `$429`; the market transcription preserves a `$479.99`
out-of-stock observation and is not delivered-cost or purchase authority.

## Affordable RDNA4 envelope authorities

- AMD, Radeon RX 9060 XT 16-GB product specifications, accessed and captured
  2026-08-10:
  <https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html>.
- AMD, *RDNA4 Instruction Set Architecture*, document 70651, accessed and
  captured 2026-08-10:
  <https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture>.
- AMD, RX 9060 XT launch release, accessed and captured 2026-08-10:
  <https://www.amd.com/en/newsroom/press-releases/2025-5-20-amd-introduces-new-radeon-graphics-cards-and-ryzen.html>.
- Dated semantic transcription of Newegg item `N82E16814150910`, observed
  2026-08-10:
  <https://www.newegg.com/xfx-swift-rx-96tsw16bq-radeon-rx-9060-xt-16gb-graphics-card-double-fans/p/N82E16814150910?Item=N82E16814150910&SoldByNewegg=1>.

The product page hashes to
`9013b9e7dfd1e4ecc805e2756df2838b60f3ee0d69a44d25ff8059671194f4ba`;
the ISA hashes to
`96dc97df3468a4e63a13095e2540ba13aaa75cf4635a29516b59760695e25e0c`;
the launch release hashes to
`03df4b873908c7e15ef80644888bfa4f1a49999628eda9c4260e34c6c2cdb977`;
and the explicitly weaker market transcription hashes to
`79065195a1e523514aa377a91dad8f514db72a504e533b595303699d3148f718`.

Decision: technical, ISA, and moving-market authority for PW-0165. AMD binds
RX 9060 XT to 103-TFLOPS dense half-precision Matrix and 205-TFLOPS structured-
sparse Matrix rates, 16 GB GDDR6, 160-W typical board power, one 8-pin input,
and a 450-W minimum PSU. The ISA distinguishes dense BF16/F32-accumulate WMMA
from the sparse form requiring two zero elements per four. The release binds
16-GB SEP to `$349`; the market row records a new in-stock `$449.99` card with
free shipping before unknown tax and is not delivered purchase authority.

## Affordable Xe2 envelope authorities

- Intel, Arc B-series desktop product specifications, accessed and captured
  2026-08-10:
  <https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc/desktop/b-series.html>.
- Intel, *AI Data Types and Native Hardware Support*, accessed and captured
  2026-08-10:
  <https://www.intel.com/content/www/us/en/support/articles/000098346/graphics.html>.
- Intel, *Intel Xe GPU Architecture*, accessed and captured 2026-08-10:
  <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html>.
- Intel, *Intel Arc B-Series Graphics Quick Reference Guide*, and official
  B-series launch release, accessed and captured 2026-08-10:
  <https://download.intel.com/newsroom/2024/client-computing/Intel-Arc-B-Series-Graphics-Quick-Reference-Guide.pdf>
  and
  <https://download.intel.com/newsroom/archive/2025/en-us-2024-12-03-intel-launches-arc-bseries-graphics-cards.pdf>.
- Intel Graphics Compiler commit
  `2eefea9414f2064b2250045305b28a2f73d4f644`, pinned 2026-08-10:
  <https://github.com/intel/intel-graphics-compiler/tree/2eefea9414f2064b2250045305b28a2f73d4f644>.

The product page hashes to
`f823f01910776e04f4ac5b3bb151b960cb857c96630ca9cbead15a86986679c8`;
the datatype page hashes to
`79c9b9a32ccb7d1869777d85384cd06ddc4b2238218eead74cd03c978a40f3d1`;
the Xe architecture page hashes to
`ae4b7eaa179b7eabb5383b951f7b6bd8ae27058f727c724a534acc835899881f`;
the QRG hashes to
`6957f49863018e0226b126f5500a97304ff7cab2a9fe61e75019cc7db51b1d4e`;
and the launch release hashes to
`597c943c6a4a7ab6d929a4f47c6731fe45427d1b5715bd7369765f8b3437e934`.
At the pinned IGC commit, the DPAS specification hashes to
`79ba16ab6716e9099aaaf88875d7213c1a2581601aae8fd7e20fcd70d7737170`
and the Xe2 scheduling table hashes to
`17502f5b5050ec5538ae3424d09d07a6aea5d32f92b01d71b221bb58f60800c6`.

Decision: technical and launch-price authority for PW-0166. Intel binds B580
to Xe2-HPG, 20 Xe cores, 160 XMX engines, 12 GB, 190-W total board power,
233 peak INT8 XMX TOPS, BF16 and INT8 XMX support, and a `$249` launch price.
The pinned compiler sources bind BF16 and INT8 DPAS operations per channel and
the precision-independent Xe2 scheduling premise used to derive a
source-oriented BF16 ceiling. They do not establish delivered cost, installed
oneAPI performance, fit, cabling, cooling, or purchase authority.

## Affordable Xe-HPG envelope authorities

- Intel, Arc A770 16-GB product specifications, accessed and captured
  2026-08-10:
  <https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html>.
- Intel, *Intel Xe GPU Architecture* (Xe-HPG section), accessed and captured
  2026-08-10:
  <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/xe-arch.html>.
- Intel, *Intel Arc A-Series Graphics — Desktop Quick Start Guide*, accessed
  and captured 2026-08-10:
  <https://www.intel.com/content/www/us/en/support/articles/000091128/graphics/intel-arc-dedicated-graphics-family.html>.
- Intel, oneAPI 2026 system requirements, accessed and captured 2026-08-10:
  <https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-toolkit/2026.html>.
- Supermicro FAQ 42887, H11SSL-i Resizable BAR support, accessed and captured
  2026-08-10:
  <https://www.supermicro.com/en/support/faqs/faq.php?faq=42887>.
- Dated semantic transcriptions of eBay items `358221920938` and
  `137194058039`, observed sold at `$245` and `$215.50` respectively on
  2026-08-10 after direct fetches returned HTTP 403.

The A770 product page hashes to
`b4691de4514c938e8c0d386a6d1fa6583b96479b4c11ad4aed2726ac1527eccd`;
the Xe-HPG architecture page hashes to
`0bc5fddeb681428ce63a8972b6b5eb53a002ea4e7eb6541fab52898f62771d0b`;
the Arc quick-start page hashes to
`c7402269d97f457527b7de660dc87adbd62e1284ce0afc57a97fc246f0fb9133`;
the oneAPI requirements page hashes to
`58c984149a1e39359c0211826da1a288a610bc97e0b2646668b988d79cb8cec2`;
the Supermicro FAQ hashes to
`4441674b68b105dcd82df6bfb938e7781af6e84aa45e0ac566b3ff7ba9b36794`;
the owned-host inventory hashes to
`b8b84a557eabea9c1781186357cf0f2f4fbf75ca7c8a74656beff26fac15978b`;
and the explicitly weaker sold-market transcription hashes to
`e1774ad682ed47ee897831df47719164850dc0ba6c8136e9c3821cd708ee4385`.

Decision: technical, platform, and historical-price authority for PW-0167.
Intel binds A770 to Xe-HPG, 512 XMX engines, 262 dense INT8 XMX TOPS, 16 GB,
560 GB/s, PCIe 4.0 x16, 225-W total board power, and oneAPI support. Xe-HPG's
published datatype ratio supports the derived 131-TFLOPS BF16 ceiling.
Supermicro binds the owned H11 generation to absent native ReBAR support;
Intel binds ReBAR and supported client-GPU Linux prerequisites. The sold rows
do not establish active inventory, a delivered BOM, installation, or measured
oneAPI performance.

## Active A770 Photon exact-board authorities

- GUNNIR, A770 Photon 16G OC product page and official specification panel,
  accessed and captured 2026-08-10:
  <https://www.gunnir.cn/home/product?id=8052c305-0ca4-4630-aafb-97ba53463d98&modelid=d5512409-d36e-475e-8343-37a5301eb47f>.
- eBay item `127017511242`, dated semantic transcription observed 2026-08-10
  after a direct command-line fetch returned HTTP 403:
  <https://www.ebay.com/itm/127017511242>.

The official product HTML hashes to
`44830fe78ed6971bca45a19df127175419256486318b89147f67202f291a8e1d`;
the specification panel hashes to
`75149fac3b91f3447967121a4ea704b31f7be289611924f442ce2870f7a313e7`;
its image-bound semantic transcription hashes to
`bc34e4a4e2bb6c76186d82318f3df92c1ad3d8fc8ebfdeb0f959dc15656921d0`;
and the explicitly weaker moving-market transcription hashes to
`c8f040b06ac9e6d776b5ce0d4333090b4bf7087569c73dee0670c8f2c0773836`.

Decision: exact-board identity, dimension, connector, TBP, and dated active-
market authority for PW-0168. GUNNIR binds this Photon to 16 GB, two 8-pin
inputs, 285-W TBP, and 300x118.5x50-mm dimensions. The listing records `$411`
plus `$20` shipping, four available, and import fees included, but it does not
authenticate destination tax, delivered complete cost, physical fit, original
PSU cables, cooling, purchase authority, or installed performance.

## Active A770 Limited Edition authorities

- Intel support article 000092554, Limited Edition dimensions, accessed
  2026-08-10:
  <https://www.intel.com/content/www/us/en/support/articles/000092554/graphics.html>.
- Intel support article 000092523, Limited Edition power connectors and TBP,
  accessed 2026-08-10:
  <https://www.intel.com/content/www/us/en/support/articles/000092523/graphics.html>.
- eBay item `168591709192`, direct active listing observed 2026-08-10:
  <https://www.ebay.com/itm/168591709192>.

Command-line fetches returned HTTP 403, so the two direct official-page
semantic captures hash to
`c7656a01a4aa734b6488309d430eaaf61ad6b48df353d3aeca7a6a357a9eece5`
and
`369554f262e5409f7795823b1904ef6767a7af69f7a5c41fba2d445a668450b1`.
The explicitly weaker moving-market transcription hashes to
`dd2551749fd8c508d76deea4ea7810ac7ca76a5c181c59df09f9d47e7070d080`.
The four original listing images hash respectively to
`da65a27c20d54a6e00d2058b71c518521d14ee8e8f7841d178e7a814dc9319a3`,
`dd9b6a26ea5e8f06aaacd40c314ce46a6bb16026a5e8a21f3a7a6ad191789330`,
`9658830c4bbea8c094318ec770c4ba712d7220874147bc02b483d84ac3ad2f99`,
and
`1711a34b9557a00579906f8762f31840e17c511879391a7deae1f94352517f50`;
their image-bound semantic transcription hashes to
`f8555eab28b8e5bade3aef7b29b7a04b9edff7101f75c52b092d6de7bf1d8d41`.
PW-0169 additionally authenticates PW-0167's raw Intel product capture.

Decision: exact reference-card dimensions, power, connector, and dated active-
market authority for PW-0169. Intel binds the A770 Limited Edition to 225-W
TBP, required 8-pin plus 6-pin inputs, and 279.9x126.36x42-mm maximum extents.
The used listing identifies MPN `21P01J00BA`, shows `$300` plus `$11.71`
rendered shipping, domestic location, seller-described working order, and no
seller returns. The box and card images independently bind the 16-GB product
code and Limited Edition form but do not prove function. It does not prove
actual-destination checkout, component
health, fit, original PSU cables, cooling, purchase authority, or installed
performance.
