# PW-0036 — Shared-weight source-FP8 expert batch eight

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `979c5a7`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
