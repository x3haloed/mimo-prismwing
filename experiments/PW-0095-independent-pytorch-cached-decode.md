# PW-0095 — Independent PyTorch cached decode

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0092 run 001
  `18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25`;
  PW-0094 PyTorch-28/Rust-28 comparison
  `1c98bb1ce5086d519ce2a3d63079f7dda11c97f5abc3cb9f91fc3fc7c9960b00`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  independent PyTorch prefill plus retained per-layer K/V
- Related records: PW-0060, PW-0091 through PW-0094

## Hypothesis and mechanism

The PW-0092 Rust endpoint's second distribution differs from the exact 28-row
whole-sequence distribution because one-row source matrix reductions differ
from 28-row reductions, not because its retained K/V is corrupt. Build a
readable independent PyTorch cache oracle: evaluate the frozen 27-token prompt
through all layers while retaining each layer's source K/V, then embed only
token 264 and propagate that one row through the 48 retained caches. Use the
same pinned source operations, BF16 boundaries, dynamic FP8 activation
quantization, RoPE position 27, global/SWA policies, router, and LM head as the
cleared whole-sequence oracle.

Do not import, call, or translate Rust cache state. The only shared inputs are
the verified checkpoint, committed token fixture, and source semantics. Add
deterministic tests for cache append position, global/SWA visible ranges,
shape/head authority, and corruption rejection before execution.

## Gates

The PyTorch prefill route rows must reproduce the cleared 27-row oracle. After
one incremental row, every cache must contain 28 positions with the pinned 4/8
KV-head schedule. The incremental row's expert sets/order and route weights in
all 47 routed layers must match PW-0092 step two; route-weight error may not
exceed the existing `5e-7` source threshold. Its complete 152,576-value F32
logit vector must be byte-identical to PW-0092 step two and choose token 13.
All identities, hashes, shapes, positions, values, and evidence schemas fail
closed. Any mismatch is preserved and localized before production changes.

Enforce normative Gate 8 at checkpoint open, every prefill layer, every
incremental layer, LM head, and capture boundary: minimum memory-free 20%,
peak/current process memory at most 8 GiB, post-release resident memory at most
4 GiB, swap growth at most 512 MiB, no new throttled pages, allocator relief,
and continued health of ChatGPT, WindowServer, `nxnode`, and Syncthing.
Preserve stopped evidence. One independent cached oracle is authorized; a new
Rust trace is allowed only if the comparison fails and layer-state localization
is necessary.

This is a correctness experiment with `accepted_tokens=0`; no wall time or
component rate is accepted TPS or a promoted performance default.

## Result

Unexecuted.

## Decision

Unexecuted.
