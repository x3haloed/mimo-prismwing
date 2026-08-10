# PW-0002 — Pinned checkpoint census

- Status: completed
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: local census executed from clean
  `83ef8aa589e1740edb5c0b2fe7536d22309e461f`; release binary SHA-256
  `8d3cc1029690f77e2508f3cb965c9f5a4be0d5bea2161af7eea3f971354dc2ba`
- Checkpoint/processor/reference hashes:
  `63651580ca774f8504f676040460aed3e1244ac1`; see `spec/model.lock.json`
- Hardware, OS, compiler, storage, memory pressure: Macmini9,1; 16 GiB; macOS
  26.4.1 (25E253); Rust 1.96.0; source on USB platter disk
- Related records: PW-0001

## Hypothesis and mechanism

The pinned source indices and safetensors headers assign every tensor exactly
once and allow the physical source representation to be divided into routed
experts, dense layer zero, attention/norms, routers, embeddings/head, MTP,
vision, audio, projectors, and residual categories without loading tensor data.

## Contract

Target-faithful source inspection, L0 artifact identity. The census fails on an
unknown schema, missing file, bad SHA-256, malformed header, duplicate tensor,
index disagreement, reversed offset, or unassigned indexed tensor.

Pass: all 39 upstream files verify, every source tensor is assigned exactly
once, header-derived bytes reconcile with shard sizes and source metadata, and
the capacity/traffic constants are updated from the result.

Kill: unresolved artifact or semantic mismatch blocks runtime performance work.

## Baseline and candidate

The pinned upstream tree contains 39 files totaling 315,714,053,402 bytes,
including 18 safetensors files. The main index declares 73,081 tensors and
315,031,102,208 bytes with `save_format: fp8` and `tp_size: 4`.

Before full headers arrived, the indexed expert names and pinned configuration
already established 47×256 experts. Each expert has three FP8
4096×2048-equivalent matrices plus three f32 inverse-scale grids at 128×128
blocks: 25,171,968 bytes per expert, 302,869,118,976 routed-bank bytes, and
9,464,659,968 cold routed bytes for eight experts across 47 layers. These
derived values remain provisional until reconciled against every header.

A pinned remote-header pass now covers all 18 safetensors files without
claiming local payload verification. It found 73,530 tensors: 73,081 in the
main index, 48 standalone MTP tensors, and 449 standalone audio-tokenizer
tensors. Tensor data totals 315,683,674,448 bytes and reconciles with
315,693,004,496 safetensors file bytes plus 9,330,048 bytes of headers, padding,
and other non-tensor storage.

Exact category data bytes from those headers are:

| Category | Tensors | Bytes | GiB |
| --- | ---: | ---: | ---: |
| Routed experts | 72,192 | 302,869,118,976 | 282.0688 |
| Attention and norms | 280 | 6,094,778,240 | 5.6762 |
| LM head | 1 | 1,249,902,592 | 1.1641 |
| Token embeddings | 1 | 1,249,902,592 | 1.1641 |
| MTP | 48 | 1,189,400,448 | 1.1077 |
| Vision encoder | 364 | 1,457,188,864 | 1.3571 |
| Audio tokenizer | 449 | 652,572,240 | 0.6078 |
| Audio path | 95 | 522,254,336 | 0.4864 |
| Dense layer zero | 6 | 201,375,744 | 0.1875 |
| Routers | 47 | 197,132,288 | 0.1836 |
| Other language/projector | 47 | 48,128 | 0.00004 |

The remote pass exposed and repaired an implementation omission: the local
Rust census admitted the standalone audio tokenizer but not the standalone MTP
file, and categorized audio-tokenizer tensors by ambiguous tensor names. The
local path now includes and names both explicitly.

The Rust `prismwing census` command reads bounded safetensors headers, validates
the main index assignment, includes the standalone audio-tokenizer weights, and
emits per-tensor plus grouped byte totals. The Python checkpoint-lock tool pins
all upstream file identities and performs streaming local SHA-256 verification.

## Isolated attribution

Remote-header census command:

```sh
python3 tools/remote_checkpoint_census.py \
  --lock spec/model.lock.json \
  --index /Volumes/Elements/mimo-prismwing/checkpoints/MiMo-V2.5-63651580/model.safetensors.index.json \
  --output /Volumes/Elements/mimo-prismwing/evidence/PW-0002/remote-header-census.json
```

Evidence SHA-256:
`8ac4a179c7b0a06baee05e380dc76acd0a1a64cff4d3e2abe9572ce59afb5c52`.

## End-to-end result

The completed local Rust census independently opened all 18 local safetensors
headers and assigned all `73,530` tensors. It reports
`315,683,674,448` tensor-data bytes, `315,693,004,496` safetensors-file bytes,
and `9,330,048` header/padding bytes. Every category count and byte total is
identical to the earlier pinned remote-header census, including the standalone
48-tensor MTP file and 449-tensor audio tokenizer.

The authoritative local census is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0002/local-census-001.json`,
SHA-256
`82a5916a13d3859b7ad47bea41c5827733b0eb05c3e5b2bb32d6e9244ad4bc17`.
It is a roughly 19-MB per-tensor evidence artifact and remains outside Git.

## Correctness result

The create-new local verification receipt at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0049/checkpoint-verification.json`,
SHA-256
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`,
binds repository `XiaomiMiMo/MiMo-V2.5`, revision
`63651580ca774f8504f676040460aed3e1244ac1`, all 39 locked files, every byte
count and SHA-256, `complete=true`, `require_complete=true`, and no missing
files. The independent local Rust header census exactly reproduces the pinned
remote totals and contains no unassigned category.

## Decision

Close the PW-0002 L0 artifact gate for the pinned revision. Promote the local
verification receipt as payload identity authority and the local Rust census
as tensor/category authority. Fail closed on any future revision, receipt,
layout, count, or byte-total change.

The source-FP8 byte constants already recorded in the throughput model are
unchanged because the local census confirms rather than revises them. This
experiment reports zero accepted tokens, no endpoint TPS, and no performance
claim.
