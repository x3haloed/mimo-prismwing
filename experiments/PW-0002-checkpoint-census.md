# PW-0002 — Pinned checkpoint census

- Status: running
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation commit pending; tensor download active
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

The Rust `prismwing census` command reads bounded safetensors headers, validates
the main index assignment, includes the standalone audio-tokenizer weights, and
emits per-tensor plus grouped byte totals. The Python checkpoint-lock tool pins
all upstream file identities and performs streaming local SHA-256 verification.

## Isolated attribution

Pending complete download.

## End-to-end result

Pending complete download.

## Correctness result

The lock currently verifies every downloaded non-weight artifact. Full shard
verification and census remain pending.

## Decision

Pending.
