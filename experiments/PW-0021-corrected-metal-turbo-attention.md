# PW-0021 — Corrected MiMo Turbo KV Metal attention

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `0d83af4`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0020 locked Atomic
  source revision `074bf826e1b06005a51737d29387e36657f41bf7`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift/Metal runtime compiler; internal SSD;
  live checkpoint download excluded
- Related records: PW-0020

## Hypothesis and mechanism

A minimal Prismwing-owned Metal path with the actual effective MiMo dimensions
K=256 (192 logical plus 64 zero padding) and V=128 can repair PW-0020's missing
`dk256_dv128` dispatch without importing the fork's surrounding runtime.
Fusing packed Turbo3/Turbo4 dequantization with single-token causal attention
should establish the accelerated-parity rung and expose realistic context
scaling before whole-layer integration.

## Contract

Target-faithful attention shapes and causal softmax; modified KV
representation. Implement both compiled 128-value layouts retained by PW-0020:
Turbo3 at 50 bytes and Turbo4 at 68 bytes. This experiment passes only if:

1. a deterministic scalar reference and Metal candidate use logical K=192,
   padded K=256, V=128, the locked WHT signs/centroids, and exactly the packed
   byte layouts from PW-0020;
2. the context-17 scalar result reproduces PW-0020's score/output diagnostics
   within `2e-5` absolute and the Metal result agrees with the scalar reference
   at output relative L2 at most `1e-4` and maximum absolute error at most
   `2e-4`;
3. source layouts fail closed on any dimension, stride, format, or packed-size
   mismatch, and guard bytes around output remain unchanged;
4. runtime Metal compilation and execution succeed for contexts 17, 128,
   1,024, and 8,192 for both formats without non-finite output;
5. warm measurements use batch size one, concurrency one, one accepted token,
   10 warm-ups, and 50 measured runs. Report wall and GPU medians/p95, bytes
   read, hardware, cache state, and commit. Record the first cold dispatch
   separately. `A` and `U` are not applicable to this attention component.

No latency threshold is predeclared because the first kernel is a walking
skeleton, not a promoted performance default. Any result may guide the next
kernel, but neither format advances to fidelity status without real attention
activations, whole-layer state parity, local logits, and hosted-reference
gates. Kernel-only timing is diagnostic and cannot be reported as accepted
endpoint TPS.

## Baseline and candidate

Baseline is a deterministic scalar CPU implementation derived independently
from the locked layout and transform definition. Candidate is runtime-compiled
Metal consuming the same packed buffers. No model files or network access are
used.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0021`.

## Isolated attribution

Both formats pass every required context. Warm single-thread kernel results are:

| Format | Context | Bytes read | GPU median / p95 ms | Wall median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Turbo3 | 17 | 3,574 | 2.119 / 2.992 | 2.307 / 3.173 |
| Turbo3 | 128 | 20,224 | 14.896 / 15.207 | 15.376 / 16.522 |
| Turbo3 | 1,024 | 154,624 | 116.723 / 117.621 | 117.141 / 118.026 |
| Turbo3 | 8,192 | 1,229,824 | 933.765 / 937.063 | 934.192 / 937.660 |
| Turbo4 | 17 | 4,492 | 2.013 / 2.675 | 2.240 / 2.891 |
| Turbo4 | 128 | 27,136 | 14.102 / 14.283 | 14.487 / 15.065 |
| Turbo4 | 1,024 | 209,920 | 109.813 / 110.626 | 110.169 / 110.949 |
| Turbo4 | 8,192 | 1,672,192 | 879.384 / 883.125 | 879.793 / 883.599 |

The first cold GPU/wall dispatches are respectively 3.979/5.418 ms and
2.026/3.714 ms at context 17, rising to 942.460/943.960 ms and
887.558/889.275 ms at context 8,192 for Turbo3 and Turbo4. Each process uses
10 warm-ups and 50 measurements with batch one, concurrency one, and one
accepted token. `A` and `U` are not applicable.

The kernel is intentionally one Metal thread. Its nearly linear scaling is a
causal baseline and a clear parallelization target, not an optimized result.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All five conditions pass. Context-17 scalar diagnostics reproduce PW-0020:

| Format | Score relative L2 vs FP32 | Output relative L2 vs FP32 |
| --- | ---: | ---: |
| Turbo3 | 0.134098797 | 0.227303483 |
| Turbo4 | 0.099595100 | 0.230426987 |

Across all eight runs, Metal-versus-scalar relative L2 ranges from
`2.73e-7` to `8.33e-7`, versus the `1e-4` limit. Maximum absolute error is at
most `2.99e-7`, versus the `2e-4` limit. Every output is finite and both guard
values remain unchanged.

The candidate handles K=256/V=128 directly, including query WHT, packed
dequantization, online causal softmax, weighted V accumulation, and inverse
WHT. It therefore repairs PW-0020's missing `dk256_dv128` causal path without
copying the fork runtime.

Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0021`. The SHA-256 of its
`SHA256SUMS` manifest is
`39a5c284bf34c6f8fcb61b40939d7105e1c077f53a521b17ccf840f92eab6f46`.

## Decision

Promote this kernel only as the accelerated-correctness reference for the real
MiMo shape. Kill its serial schedule as a performance branch: 879–934 ms for
one head at context 8,192 cannot contribute to a viable endpoint.

The next cheapest experiment is a parallel online-softmax reduction over KV
tokens, retaining this kernel as the scalar oracle. Turbo4 remains the first
quality-oriented candidate and Turbo3 the compact candidate. Neither has model
fidelity status; synthetic output errors remain roughly 20–30% at longer
contexts and require real-activation evaluation.
