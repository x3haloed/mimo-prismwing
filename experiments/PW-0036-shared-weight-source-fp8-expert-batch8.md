# PW-0036 — Shared-weight source-FP8 expert batch eight

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `03f792c`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; exact PW-0035 sources, inputs,
  and independent Torch output
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust release plus runtime-compiled Metal;
  source artifacts read-only on external platter
- Related records: PW-0034, PW-0035

## Hypothesis and mechanism

PW-0035 reduced dispatch repetition but assigned each position to a separate
threadgroup, so every weight byte was decoded and read eight times. A
threadgroup per output row can load/decode each weight once, accumulate eight
position sums per lane, and reduce all eight sums in 2 KiB threadgroup memory.
This realizes physical weight reuse rather than merely naming a batch.

## Contract

Add `block_fp8_gemm8_shared_weight_lut_blocked` and a Rust CLI schedule. Pass
only if:

1. one 64-lane threadgroup owns one output row, each source weight is decoded
   once per threadgroup, each lane accumulates eight explicit F32 sums, and
   reduction uses exactly `64×8×4 = 2,048` threadgroup bytes;
2. the same PW-0035 128×128 batch-eight scalar fixture passes at maximum
   absolute error `2e-4`; all `8×4,096` complete-expert outputs pass the same
   independent Torch gates (`3e-5` relative L2, `2e-8` maximum absolute);
3. output finiteness, length, create-new behavior, hashes, and byte-identical
   repeated complete processes remain mandatory;
4. measure paired control/candidate process orders after five warmups and 30
   serialized resident-buffer complete-expert measurements per process. The
   candidate's paired mean median must be at most 3.5 ms, at least 1.5× faster
   than PW-0035 controls, and at least 2.5× faster per position than PW-0034
   batch one;
5. report p10/median/p90, cold process/dispatch, compile time, logical and
   threadgroup bytes, batch eight, concurrency one, idealized `A=8`, `U=1`,
   hardware, commit, and cache state. No endpoint TPS claim is permitted.

Passing reverses only PW-0035's rejected schedule by changing the causal
weight-sharing mechanism. It promotes neither representative route reuse nor a
heterogeneous MoE layer.

## Baseline and candidate

Control is PW-0035's flattened batch-row kernel in the same final runtime.
Candidate is identical except one row threadgroup applies each decoded weight
to all eight positions and reduces eight sums.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0036`.

## Isolated attribution

The candidate uses one 64-lane threadgroup per output row. Each lane decodes a
source FP8 weight once, multiplies it by all eight position activations, keeps
eight F32 sums, and reduces the sums in exactly `64×8×4 = 2,048` threadgroup
bytes. The control retains one threadgroup per position/output-row pair and
therefore rereads the weight eight times.

Paired process orders produce:

| Order | First median ms | Second median ms |
|---|---:|---:|
| candidate → control | 1.9259 | 5.1745 |
| control → candidate | 5.0903 | 1.9437 |

Candidate paired mean median is 1.9348 ms versus 5.1324 ms for control, a
2.653× gain. Candidate p10/p90 are 1.874/1.995 ms and 1.914/2.085 ms. Mean
per-position cost is 0.24185 ms, 4.221× faster than PW-0034 batch one. All
three predeclared timing gates pass with stable order direction.

The idealized `A=8`, `U=1` perfect-reuse routed-only diagnostic is 10.997 TPS
across 47 routed layers. It excludes routing, dense weights, attention, logits,
storage, MTP, and endpoint work and assumes the best possible eight-expert
union at every layer; it is not endpoint or representative-route throughput.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All four candidate/control artifacts are byte-identical with SHA-256
`8c198563b12f73a7c5fd181e2d173ffa55692c64ddd1d5386cb0ccdfac2a1393`.
All `8×4,096` values remain at `1.62608e-6` relative L2 and `2.47383e-10`
maximum absolute error versus independent Torch source FP8. The synthetic
1,024-output shared-weight kernel fixture is byte-exact to the Rust scalar
oracle, all values are finite, and create-new rejection passes.

Rust has 15 passing tests, Python has 21, and clippy is clean with warnings
denied. Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0036`. Its `SHA256SUMS` manifest
hashes to
`f1dea1da7fe4cfb265168b6b8e9cf8d0f3f62dff8b3180790ee13a263368b891`.

## Decision

Promote the shared-weight source-FP8 batch-eight complete-expert component.
This reverses PW-0035's rejected schedule because the causal mechanism changed:
weights are now actually decoded once and applied to eight positions.

Do not promote a heterogeneous MoE layer or endpoint result. The next slice
must run the actual PW-0016 nine-expert union with its uneven 8/5/3 position
batches, routing weights, and reduction. Fixed batch eight is an upper-bound
reuse component, not a claim that real routes fill every expert batch.
