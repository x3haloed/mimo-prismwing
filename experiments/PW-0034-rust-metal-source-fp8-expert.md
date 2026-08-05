# PW-0034 — Rust-owned Metal source-FP8 complete expert

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `1b061ab`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; locked layer-43/expert-32
  gate/up extract SHA-256
  `ca02748075edd889014c1e5beb4a2ce2abd96c1a2adebe5bd3faf278aa724276`;
  complete locked `model_pp0_ep1_shard1.safetensors`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal;
  source artifacts read-only on external platter
- Related records: PW-0015, PW-0019, PW-0031, PW-0032, PW-0033

## Hypothesis and mechanism

PW-0033's validated source-FP8 Metal projection can be made reusable inside one
Rust process and compose a complete MiMo routed expert: gate projection, up
projection, SiLU-gated product, and down projection. This provides a
target-faithful native expert reference and a production-shaped cost before the
common EP0 shard is available.

## Contract

Add a macOS-only Rust `metal-fp8-expert` path. Pass only if:

1. gate, up, and down weights and scale grids come exclusively from
   `MappedSafetensors`; gate/up are `F8_E4M3 [2048,4096]`, down is
   `F8_E4M3 [4096,2048]`, scale grids are exact 128×128-block F32 layouts, and
   all existing PW-0033 validation and create-new rules remain authoritative;
2. the runtime executes the exact source equation
   `down(silu(gate(x)) * up(x))`, where `silu(z) = z * sigmoid(z)`, with no
   quantized replacement, topology change, omitted projection, Python runtime,
   or MLX runtime on the candidate path;
3. add a deterministic tiny independent correctness fixture with signed and
   large-magnitude inputs for the new Metal SwiGLU kernel before promotion;
   every accelerated boundary and all 4,096 final expert outputs must be
   finite and checked against an independent source-FP8 reference;
4. use PW-0015's exact deterministic batch-one FP16-rounded input. Complete
   output relative L2 versus independent Torch source-FP8 matmul must be at
   most `3e-5`, maximum absolute error at most `2e-8`, and two complete
   candidate processes must produce byte-identical output artifacts;
5. after five warmups, 30 serialized resident-buffer complete-expert wall
   measurements report median/p10/p90 and must have median at most 3 ms. Report
   projection/SwiGLU dispatch composition and an explicitly idealized serial
   `8 experts × 47 routed layers` diagnostic; neither is endpoint TPS;
6. report full cold process wall, runtime compile time, logical source bytes,
   batch one, concurrency one, accepted tokens, `A`, `U`, hardware, commit,
   cache state, and exact source/output hashes. No target-fidelity or endpoint
   throughput promotion is permitted.

Passing promotes a native target-faithful complete-expert primitive. It does
not promote a routed layer, batching schedule, full model, or endpoint.

## Baseline and candidate

The correctness baseline is independently dequantized source FP8 evaluated by
Torch using the exact PW-0015 input and expert equation. The candidate is a
single Rust-owned Metal process with persistent source-FP8 buffers and explicit
SwiGLU composition.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0034`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
