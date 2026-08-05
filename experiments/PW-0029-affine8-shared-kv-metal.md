# PW-0029 — Affine8 shared-KV Metal attention

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `b2ac3bb`; implementation dirty
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

The deterministic packing fixture passes: each affine8 block is exactly 130
bytes, zero blocks encode as 130 zero bytes, the FP16 scale and signed-code
round trip is bit-deterministic, invalid bit depths and payload lengths fail
closed, and Python reconstruction exactly matches the packed representation.

All component runs use batch one, concurrency one, one accepted token, 10
warm-ups, and 30 measurements. Packed application buffers are warm with no
model or storage I/O; `A` and `U` are not applicable.

| Mode | Context | Logical bytes | GPU median / p95 ms | Wall median / p95 ms |
|---|---:|---:|---:|---:|
| global | 128 | 265,216 | 0.370 / 0.376 | 0.575 / 0.632 |
| global | 1,024 | 1,662,976 | 1.629 / 1.670 | 1.833 / 1.893 |
| global | 8,192 | 12,845,056 | 13.258 mean | 13.604 mean |
| SWA | 128 | 465,152 | 0.371 / 0.376 | 0.607 / 0.639 |

Cold GPU/wall values across recorded synthetic candidate runs range from
0.378/0.959 ms through 20.169/21.211 ms. The long-context paired medians are:

| Order | Turbo4 GPU median ms | affine8 GPU median ms |
|---|---:|---:|
| Turbo4, affine8 | 13.4795 | 13.2554 |
| affine8, Turbo4 | 13.4747 | 13.2609 |

Mean affine8/Turbo4 time ratio is `0.983752`, passing the `2.25` limit in both
orders. Affine8 reads 12,845,056 logical bytes versus Turbo4's 6,750,208, so
the result indicates simpler signed-byte dequantization offsets the additional
traffic in this 8K shared-KV component; it does not show that cache traffic is
free at longer contexts or in the full runtime.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All six contract conditions pass. Global 128/1,024/8,192 and SWA-128 preserve
every head and guard. Worst synthetic packed-scalar relative L2 is
`1.44178e-6`; worst maximum absolute error is `5.96046e-7`.

The regenerated locked-MTP fixture is byte-identical across complete runs and
retains PW-0028's affine8 projected-sublayer relative L2 of `0.0105761` versus
source. Metal agrees with its independently packed Python scalar attention at
relative L2 `2.79970e-7` and maximum absolute error `1.66893e-6`. The existing
Turbo4 learned fixture also remains green after the kernel extension.

Nine mean global-8,192 cores plus 39 SWA-128 cores give a `133.804 ms`
attention-only diagnostic. This excludes QKV/output projections, norms, KV
append, MoE, MTP, sampling, storage, and all endpoint orchestration; it is not
TPS. Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0029`.
Its `SHA256SUMS` manifest hashes to
`2363bcc3ff284cab0040c6046cf8bc732720f64c9076d96b94a9e8b252a49ff6`.

## Decision

Promote WHT-affine8 shared-KV Metal as the learned-fidelity attention
implementation candidate. It preserves PW-0028's roughly 1.06% learned
projected error while matching or slightly beating Turbo4's 8K component time
on this M1 schedule.

This does not promote target fidelity or endpoint performance. The next causal
step is a complete learned transformer-layer fixture using actual base-layer
weights and activations, affine8 attention, and the selected affine8 MoE
substrate. If the common EP0 shard remains unavailable, continue executable
foundation and route/format work that does not pretend MTP weights are base
weights.
