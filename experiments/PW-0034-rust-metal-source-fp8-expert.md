# PW-0034 — Rust-owned Metal source-FP8 complete expert

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `188c7d7`; implementation dirty
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

The candidate reads `8,388,608` FP8 bytes plus `2,048` scale bytes for each of
gate, up, and down. Including the 4,096-wide F32 input and output, the reported
logical source-and-I/O footprint is `25,204,736` bytes. One serialized command
buffer dispatches gate FP8 GEMV, up FP8 GEMV, F32 SwiGLU, and down FP8 GEMV.

After five warmups, the first process's 30 complete-expert measurements have a
1.0209 ms median, 0.9107 ms p10, and 1.2391 ms p90. The repeat process has a
1.0788 ms median, 0.9670 ms p10, and 1.3084 ms p90. First-dispatch times are
2.909 and 3.280 ms. Runtime compilation is 183.1 ms first and 1.74 ms repeat;
complete process wall is 0.74 and 0.10 seconds.

Repeating that first batch-one cost serially for eight experts across 47 routed
layers gives 2.605 routed-only token positions per second. This is deliberately
pessimistic and is not accepted TPS: it omits expert batching/reuse and also
omits routing, the dense spine, attention, logits, storage, MTP, and endpoint
work. It establishes that a naive source-FP8 serial schedule cannot be the
performance architecture.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All 4,096 final outputs agree with the independently dequantized Torch
source-FP8 oracle at `4.69754e-7` relative L2 and `4.45652e-11` maximum absolute
error. The new signed/large-magnitude SwiGLU Metal fixture agrees with its f64
oracle within `1.97745e-7`. All values are finite and the thresholds pass.

The output SHA-256 is
`1fb7fb1755eff0c72ab3a0de7744bf62455fc3177a6f6e0b35617978ca97247e`;
two complete candidate processes are byte-identical. The independent fixture
generator also repeats byte-identically, and an existing output path is
rejected before source mapping or dispatch. Rust has 15 passing tests, Python
has 21, and clippy is clean with warnings denied.

The exact local down tensor and scale hash to
`75706d115d6706950c6a6b147959ab64cb8bb4cfc0004bad467ace9b413f7495`
and `db951c18ed0788b74171ce09bc523689055f82dc5787bc21d85569d2b328d06e`.
The complete local shard size matches its `3,490,619,024`-byte lock. Its whole
local hash was deliberately deferred so a 3.49-GB platter scan would not
compete with the owner's active durable checkpoint download; the pinned LFS
SHA remains `fd89388271eac237e06ace68a832156357b42f85820856afee24da7bb36d9dcc`.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0034`.
Its `SHA256SUMS` manifest hashes to
`51fe7fdfd2c7f0e07b204f92211c4fe1a545301d56f3d481fcbd631c36ac34b3`.

## Decision

Promote the Rust-owned Metal source-FP8 complete-expert primitive as the native
target-faithful routed-expert reference. The runtime now owns a complete real
gate/up/SwiGLU/down semantic path without Python, MLX, or a quantized substitute
on the candidate path.

Kill the naive batch-one serial extrapolation as a performance architecture.
Do not promote a routed layer or endpoint claim: the next routed branch must
batch positions by expert and measure heterogeneous top-eight execution, then
compose routing and weighted reduction. EP0 remains required for actual
base-layer hidden states and accumulated target-fidelity evidence.
