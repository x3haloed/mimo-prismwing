# PW-0029 — Affine8 shared-KV Metal attention

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `6da266b`; contract dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; locked PW-0026 MTP file and
  PW-0020 WHT source
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift/Metal runtime compiler; internal SSD;
  learned source read-only on external platter
- Related records: PW-0025, PW-0026, PW-0028

## Hypothesis and mechanism

PW-0028's joint WHT-affine8 representation can reuse PW-0025's shared-KV
topology with simpler per-value dequantization. Despite moving 1.912 times the
Turbo4 cache bytes, it should remain within 2.25 times Turbo4's complete GQA
attention-core cost while reproducing its scalar representation exactly.

## Contract

Add an explicit modified `wht_affine8` format: each independent 128-value K or
V block contains one little-endian FP16 symmetric scale followed by 128 signed
8-bit codes in `[-127,127]`. Codes use ties-to-even rounding after the exact
locked WHT. Pass only if:

1. a deterministic packing fixture verifies block size, FP16 scale, signed
   code layout, zero handling, fail-closed format selection, and CPU
   reconstruction before Metal is measured;
2. shared-KV Metal covers global 64Q/4KV contexts 128, 1,024, and 8,192 plus
   SWA 64Q/8KV context 128, preserving PW-0024 RoPE, V scale, GQA, learned or
   deterministic sinks, online softmax, and inverse WHT;
3. every synthetic Metal output agrees with an independent packed scalar
   reference at relative L2 at most `4e-4` and maximum absolute error at most
   `7e-4`, all 64 head guards remain intact, and no tensor dimension is
   inferred from buffer length;
4. a newly generated locked-MTP context-17 affine8 fixture passes the same
   learned-source identity and tensor gates as PW-0026, then Metal agrees with
   its packed scalar expected attention at the same error limits;
5. at global context 8,192, two paired process orders (`Turbo4, affine8` then
   `affine8, Turbo4`) each use 10 warm-ups and 30 measurements. The ratio of
   mean affine8 GPU medians to mean Turbo4 GPU medians must not exceed 2.25;
6. report cold and warm wall/GPU median/p95, batch one, concurrency one, one
   accepted token, logical bytes, hardware, commit, and packed-buffer state.
   `A` and `U` are not applicable. Also report the nine-global-plus-39-SWA
   component diagnostic at context 8,192 without calling it TPS.

Passing promotes affine8 as the learned-fidelity attention implementation
candidate. Failure retains its representation evidence but rejects this Metal
schedule. Neither result establishes accumulated model fidelity, full-layer
performance, or endpoint TPS.

## Baseline and candidate

Baseline is PW-0025's Turbo4 shared-KV kernel and exact packed scalar oracle.
Candidate changes only packed K/V representation and dequantization; query
rotation, tiling, GQA mapping, sinks, online softmax, and output reconstruction
remain common.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0029`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
