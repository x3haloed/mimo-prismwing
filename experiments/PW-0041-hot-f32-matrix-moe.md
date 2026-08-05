# PW-0041 — Hot exact-F32 matrix MoE backend

- Status: complete
- Disposition: rejected
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `51b144c`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0039 exact router, input,
  selected source-FP8 tensors, and Torch source-FP8 reference
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2 exploratory backend; selected
  exact tensors on external platter
- Related records: PW-0016, PW-0039, PW-0040

## Hypothesis and mechanism

PW-0040 shows dispatch reordering does not improve the custom FP8 GEMV path.
The source representation is only 25.2 MB per expert, but its scalar FP8 decode
and reduction realize about 13 GB/s across the real union. Expanding a bounded
hot expert working set once to exact F32 values and using Apple's tuned matrix
backend may trade resident memory for enough arithmetic efficiency to reduce
the 17.1 ms dynamic-MoE component cost without changing stored weights or model
equations.

## Contract

Add a deterministic exploratory MLX benchmark for the exact PW-0039 fixture.
Pass only if:

1. validate every source artifact and tensor authority exactly as PW-0039.
   Expand each FP8 value and its pinned F32 block scale into F32; no requantized,
   F16, BF16, or approximate expert representation is permitted;
2. independently fixture source expansion, native noaux-tc selected sets and
   normalized weights, expert gather, F32 gate/up/SwiGLU/down, and weighted
   reduction. Complete parity must meet `4e-5` relative L2 and `3e-8` maximum
   absolute error versus PW-0037's independent Torch source-FP8 output;
3. time one dynamically routed batch-eight request using the resident expanded
   expert union. Include router, decision, gather, all expert matrix operations,
   weighting, and reduction; exclude and separately report source validation,
   expansion, and installation wall;
4. use two paired candidate/control process orders. Each uses ten warmups and
   30 measurements for MLX candidate and PW-0039's declared five/30 protocol
   for native control. Promote the backend branch only if candidate mean median
   is at least 20% faster than control and at most 14 ms;
5. report p10/median/p90, complete process wall, cold/install/warm state, source,
   expanded, executable and peak bytes, batch eight, concurrency one, accepted
   tokens, `A=8`, observed `U=1.125`, and routed-only diagnostic. No endpoint
   TPS, whole-model residency, or storage-cold claim is permitted;
6. fail closed on non-finite data, route mismatch, artifact mismatch, output
   mismatch, create-new overwrite, or unknown MLX version/device. Repeated
   canonical reports must bind all outputs and source hashes.

Passing promotes the exact-F32 hot-cache matrix mechanism for native bridge
work, not Python as the inference binary and not a whole-model default. Failure
rejects expanded-F32 MLX matmul as the near-term expert backend on this M1.

## Baseline and candidate

Control is PW-0039's promoted Rust/Metal dynamic source-FP8 path. Candidate is
the same exact source values expanded to an MLX F32 resident hot set and
executed with dynamically recomputed routes.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0041`.

## Isolated attribution

The first attempt failed before loading model tensors because MLX 0.31.2 does
not expose `__version__` on its module. The implementation now fails closed
using installed-package metadata; the failure is preserved.

The candidate validates all 12 selected artifacts, expands every exact FP8 and
F32 block-scale pair to F32, installs nine experts plus the F32 router, and
recomputes routing and heterogeneous expert execution per timed request. In
candidate/control then control/candidate order, candidate medians are
`21.865042` and `21.441500` ms versus controls `17.298667` and `17.254000` ms.
Candidate mean is `21.653271` ms versus `17.276334` ms: a 0.7979× speedup, or
25.33% slowdown. Candidate p10/p90 pairs are `21.319500/24.475417` and
`20.808791/21.998500` ms.

`226,547,712` selected source-tensor bytes expand to `905,969,664` expert bytes;
router-inclusive executable bytes are `910,164,992`. MLX reports 919,809,540
peak bytes; complete process peak footprint is 1,089,491,712 bytes. The first
platter/OS-cache-uncontrolled install took 6,591 ms; paired warm-cache installs
took 752 and 555 ms. Complete paired process wall is 2.74 seconds.

The resulting fixed-fixture routed-only `A=8`, `U=1.125` diagnostic is 7.8608
TPS. It is not endpoint TPS.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Deterministic tests independently verify exact `128×128` block-scale expansion
and F64 error metrics. Native selected sets match on every request and maximum
route-weight absolute error is `2.9802322e-8`.

Complete candidate output is deterministic with SHA-256
`e942447375c2bfb10f94d05a36f7448024f4f73125da3dda8e1371249e15e69d`.
It has relative L2 `1.769381e-6` and maximum absolute error `7.116796e-11`
versus independent Torch source FP8. Create-new rejection exits 1. Rust has 15
passing tests, Python has 23, and clippy is clean with warnings denied.

Raw evidence is under `/Volumes/Elements/mimo-prismwing/evidence/PW-0041`.
Its `SHA256SUMS` manifest hashes to
`f47ac7ad799e70cd9f2d2f4cb8fd72a73e4f932858a8cf61fe62efca2777617a`.

## Decision

Reject resident exact-F32 MLX matmul as the near-term expert backend on this M1.
The arithmetic is faithful, but batch eight is too small to amortize matrix
backend scheduling and F32 weight traffic; it is materially slower and uses
roughly four times the source expert bytes.

Retain the deterministic benchmark as a backend control. Do not build the C++
bridge for this mechanism. PW-0039's direct source-FP8 Rust/Metal schedule
remains promoted; the next speed branch must reduce useful work/bytes or improve
the direct FP8 kernel rather than expanding the hot set.
