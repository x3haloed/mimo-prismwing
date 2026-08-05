# PW-0021 — Corrected MiMo Turbo KV Metal attention

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `015cd31`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
