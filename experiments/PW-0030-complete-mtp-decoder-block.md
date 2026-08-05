# PW-0030 — Complete learned MTP decoder block

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `7582798`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
