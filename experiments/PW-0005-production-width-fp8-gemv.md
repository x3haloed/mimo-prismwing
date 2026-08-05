# PW-0005 — Production-width real FP8 GEMV slice

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
- Related records: PW-0002, PW-0004

## Hypothesis and mechanism

Correct FP8 byte decoding is insufficient unless block-scale indexing and
production-width accumulation also agree. Four real 4096-column MTP projection
rows should reproduce a trusted f32 matmul when every 128-column block uses the
matching inverse scale.

## Contract

Target-faithful component semantic using real pinned bytes. The input width,
32 scale blocks, raw bytes, deterministic activation, and expected outputs are
frozen in the fixture. This is a correctness experiment, not accepted-TPS
evidence.

Pass: all four row outputs agree with the safetensors/PyTorch f32 oracle within
absolute error `2e-7`; malformed dimensions fail closed.

## Baseline and candidate

The capture uses rows 0–3 of
`model.mtp.layers.0.mlp.gate_proj.weight`, all 4096 columns, its 32 corresponding
scale values, and a deterministic low-amplitude sinusoidal activation. The
oracle is safetensors 0.7.0 plus PyTorch 2.13.0 f32 matmul. The candidate is a
dependency-free Rust scalar block-FP8 GEMV.

## Isolated attribution

Fixture SHA-256:
`3f544c2bf5f6273cf695af18ce570bb9e67af937feaae67e98c279df190ba8a6`.

## End-to-end result

Each fixture row traverses all 4096 real FP8 bytes, 32 real block scales,
activation multiplication, and f32 accumulation to an asserted row output.

## Correctness result

All four production-width rows pass at `2e-7`. Tests also reject zero block
width, non-divisible input widths, wrong scale counts, and row-width mismatch.

## Decision

Promote the Rust scalar block-FP8 GEMV as the readable production-width oracle.
It is intentionally not a performance default. The next rung is an accelerated
kernel compared against this oracle at full expert shapes and real offsets.
