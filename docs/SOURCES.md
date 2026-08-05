# Source ledger

Every source is pinned to the revision used for a decision. A link to a moving
branch is convenient navigation, not identity evidence.

## MiMo-V2.5

- Repository: `XiaomiMiMo/MiMo-V2.5`
- Hugging Face revision:
  `63651580ca774f8504f676040460aed3e1244ac1`
- Local immutable source root:
  `/Volumes/Elements/mimo-prismwing/checkpoints/MiMo-V2.5-63651580`
- Machine-readable file identity: `spec/model.lock.json`
- Semantics: pinned `configuration_mimo_v2.py`, `modeling_mimo_v2.py`,
  tokenizer, processor configuration, and chat templates.
- Decision: authoritative source checkpoint and component semantics.

The upstream revision contains 39 files totaling 315,714,053,402 bytes. Its 18
safetensors files use upstream LFS SHA-256 identities. Local verification is
incomplete until the external-disk download and full hash pass finish.

## OpenRouter MiMo-V2.5

- Model slug: `xiaomi/mimo-v2.5`
- Initial provider: Parasail, individually pinned with fallbacks disabled.
- Discovery date: 2026-08-04.
- Decision: only external whole-model reference; PW-0001 proves scoreable text,
  image, multi-image, audio, video, and mixed requests.
- Limitation: provider implementation and future availability are external and
  require frozen endpoint metadata plus drift canaries.

## Predecessor runtimes

- TurboFieldfare: `7a99f2a635e3adf7ed0720b882d2edb600f2f0da`
- Swiftlet: `d0cf7021b0544bf1ba4f264c592386a54bc49a00`
- Decision: architecture and workflow references only. No code has been copied
  into the Prismwing runtime at this milestone.
