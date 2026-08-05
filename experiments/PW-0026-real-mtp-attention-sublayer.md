# PW-0026 — Real learned MTP attention sublayer

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `84fbb39`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; local
  `model_mtp.safetensors` SHA-256
  `a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2 plus Swift/Metal; source file
  read-only on external platter, derived fixtures on evidence storage
- Related records: PW-0004, PW-0024, PW-0025

## Hypothesis and mechanism

The complete local MTP file can supply the first learned-tensor attention
sublayer while the target common shard downloads. Its actual RMSNorm, fused
FP8 QKV projection/scales, BF16 output projection, and learned 64-head SWA sink
can drive PW-0025's compressed attention path from deterministic production-
width hidden states to an observable 4,096-wide sublayer output.

## Contract

Target-source MTP attention tensors and MiMo attention semantics; deterministic
synthetic activations; modified Turbo4 KV cache. Pass only if:

1. fail closed unless all tensor names, dtypes, shapes, offsets, source SHA,
   and source revision match: input norm `[4096]` BF16, QKV `[14848,4096]`
   FP8 plus `[116,32]` F32 scales, sink `[64]` BF16, output projection
   `[4096,8192]` BF16;
2. use context 17, hidden width 4,096, 64 Q heads, eight KV heads, Q/K head
   width 192, V width 128, partial RoPE 64/base 10,000, value scale 0.707,
   learned sinks, and the exact final-token SWA causal path;
3. sampled decoded QKV projection values and outputs agree with an independent
   scalar FP8/block-scale oracle at maximum absolute error `2e-4`; BF16 norm
   and output-projection decoding must be bit-exact;
4. Turbo4 Metal attention agrees with a scalar packed-cache reference at
   relative L2 at most `4e-4` and maximum absolute error `7e-4`;
5. produce and hash both uncompressed-source and Turbo4 4,096-wide attention
   sublayer outputs. Quantization error is diagnostic only and cannot promote
   fidelity. No performance or endpoint TPS claim is in scope.

Passing advances learned attention into the transformer-layer fixture. It does
not substitute MTP weights for base-layer weights or prove whole-model parity.

## Baseline and candidate

Baseline uses decoded source QKV/KV attention/output projection. Candidate
changes only K/V storage and attention execution to PW-0025 Turbo4. Both share
the same normalized hidden states, learned tensors, RoPE, value scale, sink,
and output projection.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0026`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no performance or endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
