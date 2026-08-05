# PW-0030 — Complete learned MTP decoder block

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `8fb2555`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; locked complete
  `model_mtp.safetensors` SHA-256
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2 plus Swift/Metal; checkpoint
  read-only on external platter
- Related records: PW-0019, PW-0026, PW-0029

## Hypothesis and mechanism

The complete MTP file contains three learned dense decoder blocks. Layer zero
can close the current causal gap from deterministic production-width hidden
state through RMSNorm, fused FP8 QKV, affine8 Metal attention, BF16 output
projection, residual, pre-MLP RMSNorm, learned FP8 SwiGLU MLP, and final
residual. This is an MTP decoder-block fixture, not a base-model layer.

## Contract

Use context 17 and the exact deterministic PW-0026 hidden states. Build both a
source-cache baseline and modified affine8 candidate for MTP layer zero. Pass
only if:

1. the full source file SHA and all used tensor names, dtypes, shapes, and byte
   offsets match exactly: input/pre-MLP norms, fused QKV/scales, sink, output
   projection, gate/up/down FP8 weights, and all three scale grids;
2. BF16 tensors are independently raw-bit decoded exactly; sampled QKV,
   gate/up/down projection values agree with float64 scalar dots at maximum
   absolute error `2e-4`;
3. source and candidate preserve RMSNorm epsilon `1e-5`, layer-zero SWA
   attention semantics, output projection, both residual edges, SiLU gate,
   elementwise gate/up product, and down projection without omitting or
   replacing any dimension;
4. the candidate consumes the actual 8,192-float Metal output emitted by the
   PW-0029 learned affine8 kernel. Its artifact identity, length, finiteness,
   guards, and packed-scalar parity must be checked before the output enters
   the BF16 projection;
5. independently hash source and candidate attention, post-attention residual,
   normalized MLP input, MLP output, and final 4,096-wide block state. Report
   candidate/source relative L2 at each boundary; no component-error threshold
   is promoted to target fidelity from this one fixture.

No performance or endpoint TPS claim is in scope. Passing promotes the complete
MTP decoder block as a correctness reference and executable-foundation slice.
It cannot substitute for the base-layer gate once EP0 becomes available.

## Baseline and candidate

Baseline changes no learned tensor and uses uncompressed source K/V attention.
Candidate changes only the K/V representation and attention execution to
PW-0029 WHT-affine8 Metal. Both then traverse the same actual output
projection, residual, pre-MLP norm, source-FP8 dense SwiGLU weights, and final
residual.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0030`.

## Isolated attribution

The complete 1.1-GB MTP source and pinned model source pass their locked
SHA-256 checks. Every used layer-zero tensor matches its predeclared name,
dtype, shape, and byte offsets. Independent raw-bit decoding exactly matches
the tensor library for input/pre-MLP norms, all 64 sinks, and the complete BF16
output projection.

Float64 scalar-dot maximum absolute errors are:

| Projection | Max absolute error |
|---|---:|
| fused QKV | `1.91264e-6` |
| MLP gate | `9.05227e-7` |
| MLP up | `6.56720e-7` |
| MLP down | `4.02478e-7` |

All are more than two orders of magnitude below the `2e-4` gate. The actual
Metal artifact contains exactly 8,192 finite little-endian F32 values and
hashes to `fdb8ea39872939fd44bd01d383f384239ed074b156ee8cbbffe078fdba9a6108`.
The producing run retained all 64 head guards and reports relative L2
`2.79970e-7` versus the independently packed scalar candidate.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All five conditions pass. Candidate/source relative L2 evolves across the
complete learned path as follows:

| Boundary | Relative L2 |
|---|---:|
| affine8 Metal attention | 0.9841% |
| BF16 projection plus attention residual | 0.4608% |
| pre-MLP RMSNorm | 0.4289% |
| learned FP8 SwiGLU MLP output | 0.6273% |
| final decoder-block state | 0.6203% |

The source final state hashes to
`d89e298c4c0944d0d06f4fe6f62e8d86800f8e7163e74919a9099c93f8686814`;
the candidate hashes to
`42d997477cadc8dafefadc9feba6fddc7191a7464f7498c545ce416559dc9d05`.
A second complete run is byte-identical.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0030`.
Its `SHA256SUMS` manifest hashes to
`d03880910b61fcff4035d0badbca6c7f7f1d5d64230371700bf9ad0b2f3d7fab`.

## Decision

Promote the complete learned MTP decoder block as the current correctness
reference and executable-foundation slice. PW-0029's Metal output now
causally drives actual downstream projection, residual, normalization, dense
SwiGLU, and final state rather than ending at an isolated attention tensor.

Do not promote target fidelity or base-layer parity from the 0.6203% result.
It is one deterministic final-token MTP block; it excludes accumulated layers,
base MoE routing, logits, hosted-reference parity, and endpoint timing. Once
EP0 shard1 completes, reproduce this ladder on an actual base layer and join it
to the heterogeneous MoE substrate. Until then, MTP layer chaining and native
checkpoint/runtime foundation remain honest independent work.
