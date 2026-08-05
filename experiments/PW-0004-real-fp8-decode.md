# PW-0004 — Sampled-real FP8 block decode

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation commit pending
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; `model_mtp.safetensors`
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`
- Hardware, OS, compiler, storage, memory pressure: Macmini9,1; 16 GiB; macOS
  26.4.1 (25E253); Rust 1.96.0; source on USB platter disk
- Related records: PW-0002, PW-0003

## Hypothesis and mechanism

The source `F8_E4M3` bytes can be decoded independently and combined with the
checkpoint's inverse scale for their 128×128 block to reproduce a trusted
safetensors/PyTorch read of the same real tensor positions.

## Contract

Target-faithful source representation, L0 byte identity and component-local
numeric parity. The fixture records raw bytes rather than only converted
floats, the exact source tensor and scale tensor, the source shard SHA-256, and
the 128×128 block layout pinned by `config.json`.

Pass: all 32 sampled bytes decode exactly to the library FP8 values and scaled
values match within `1e-9` in f32.

## Baseline and candidate

The fixture samples rows 0–3 and columns 0–7 of
`model.mtp.layers.0.mlp.gate_proj.weight` and scale `[0,0]`. Python safetensors
0.7.0 with PyTorch 2.13.0 supplies the capture-side decoded values. Rust uses a
direct E4M3FN bit decoder with no tensor library.

## Isolated attribution

Fixture SHA-256:
`b23ce21531bf872868ec4aec8fb84e8b06bb11a32dfd27d7b628458c607e52d9`.

The exhaustive 256-pattern format fixture has SHA-256:
`feb5d20d36a561e9011563edf6896216f49cbea6023a8689c58be39ce3c21a67`.

## End-to-end result

Real pinned checkpoint bytes travel through independent FP8 decoding and block
scaling to asserted dequantized values in the Rust test suite.

## Correctness result

All 32 real values pass exact decoded-FP8 comparison and `1e-9` dequantized
comparison. A second fixture compares the f32 output bits for all 256 possible
FP8 encodings with PyTorch 2.13.0, including subnormals, signed zero,
maximum-finite values, and signed NaNs; every encoding passes exactly.

## Decision

Promote the decoder as complete format-level and sampled-real correctness
evidence for source FP8 reads. Do not yet promote it as a matrix kernel or
performance path; production-shape GEMV/GEMM and accumulation parity remain
unproven.
