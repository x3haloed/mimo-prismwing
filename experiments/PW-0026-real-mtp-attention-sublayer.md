# PW-0026 — Real learned MTP attention sublayer

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `b4a0089`; implementation dirty
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

The complete 1.1-GB source file hashes to its locked SHA-256. Header metadata
matches every predeclared dtype, shape, and offset. Independent raw-bit BF16
decoding exactly matches the tensor library for norm, sink, and all 33,554,432
output-projection elements.

The four QKV sample rows `[0, 1, 12288, 13824]` cover Q, K, and V partitions.
MLX's production projection differs from float64 scalar dots by at most
`1.91264e-6`, passing the `2e-4` gate.

No performance measurement or TPS is reported. Fixture generation includes
source-file hashing, tensor decode, installation, both baselines, and artifact
serialization, so its wall time is not a decode benchmark.

## End-to-end result

Out of scope; no performance or endpoint TPS claim is permitted.

## Correctness result

All five conditions pass. The learned context-17 Metal output agrees with the
scalar packed-cache reference at relative L2 `2.28478e-7` and maximum absolute
error `6.55651e-7`; both are far below the predeclared limits. All 128 head
guards remain intact.

The learned quantization diagnostics are material:

- Turbo4 attention output relative L2 versus uncompressed source: `0.185848`.
- Projected 4,096-wide sublayer relative L2: `0.194277`.

Output identities:

- source attention: `aff26ef6640e73974467f7c957c466bb15543a7030ad399d427ece6d903c1258`
- Turbo4 attention: `0313cb602d94e1b303ee20b30e62147a77737a7c05c5a5f0fad31d2d8aa17c20`
- source projected sublayer: `b1b2cae21fc72bbd989284138a6f90c69be6e6eb315b6932b8ae34222a984dce`
- Turbo4 projected sublayer: `7462245a5132f23a7fcfbcbfd8f504e77d52d7d6e4f1252495d08821bbb5ab38`

Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0026`. The SHA-256 of its
`SHA256SUMS` manifest is
`34c1f950c1a61e6799ba536f12a149300e668bb7b72b0294facda780b9f159f4`.

## Decision

Promote the real learned MTP attention sublayer fixture into the correctness
ladder. Do not promote uniform Turbo4 for fidelity: nearly 19.4% sublayer error
on deterministic learned projections is a caution result, not acceptable
model evidence.

The next cheapest fidelity experiment compares mixed precision—especially
higher-precision K with Turbo4 V—on this exact learned fixture before any
whole-layer integration. The MTP result cannot substitute for a base-layer
gate once the common shard becomes available.
