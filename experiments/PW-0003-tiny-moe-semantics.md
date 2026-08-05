# PW-0003 — Tiny noaux_tc SwiGLU MoE semantics

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation commit pending
- Checkpoint/processor/reference hashes:
  `63651580ca774f8504f676040460aed3e1244ac1`
- Hardware, OS, compiler, storage, memory pressure: Macmini9,1; 16 GiB; macOS
  26.4.1 (25E253); Rust 1.96.0; no material memory pressure
- Related records: PW-0002

## Hypothesis and mechanism

MiMo's routed expert transition can be represented independently of PyTorch as
sigmoid router scores, correction-biased `noaux_tc` group and expert choice,
normalization of the uncorrected chosen scores, and a weighted mixture of
SwiGLU experts.

## Contract

Target-faithful source-derived component semantic. The fixture uses seeded,
hand-declared tiny tensors and the equations in the pinned Xiaomi implementation
at `MiMoV2MoEGate`, `MiMoV2MLP`, and `MiMoV2MoE`. Correction bias influences
selection but not mixture weight. SwiGLU is `silu(gate(x)) * up(x)` followed by
`down`.

Pass: an independently implemented Rust scalar path reproduces selected expert
IDs, normalized weights, and outputs within `1e-14` in f64. Unknown fixture
semantics or inconsistent dimensions fail closed.

## Baseline and candidate

The Python generator implements a readable equation-level oracle without
PyTorch or NumPy. The Rust implementation owns the first executable model
semantic. The fixture has two tokens selecting different expert groups so one
hard-coded hot path cannot pass both cases.

## Isolated attribution

Fixture SHA-256:
`ce27b5e27dc2b3e4325ca0f42d424cc257b899a93911d7c5b60416baac47e044`.

## End-to-end result

Both inputs travel from real fixture JSON through routing, expert evaluation,
mixture accumulation, and asserted observable output in the Rust test suite.

## Correctness result

`cargo test` passes the two-token fixture at `1e-14`. Python tests prove the
committed JSON is reproducible from the generator. Rust rejects unknown schema,
semantic name, and inconsistent dimensions.

## Decision

Promote this scalar semantic as the first rung of the correctness ladder. It is
not evidence for FP8 decoding, production shapes, accelerated parity, a full
layer, or whole-model correctness; those remain later rungs.
