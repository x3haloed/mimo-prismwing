# PW-0090 — Specialized BF16 LM-head reduction

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0089 Rust manifest
  `0e8b14621a5e3e3715c8136bbef53ae94da674df9a0e9435e3ae881fb5d11f80`;
  PW-0089 comparison
  `6f00f95147aebf4f7c941893fe4aa1224f9874e74934cd8be6d42fc634cc82b8`;
  pinned PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust endpoint
- Related records: PW-0060, PW-0072, PW-0073, PW-0089

## Hypothesis and contract

PW-0089 proves the complete transformer and final RMSNorm are bit-exact. The
remaining 45 logit differences arise because PyTorch applies BF16 GEMV to the
last normalized row and contiguous 4,096-wide LM-head weight rows, while Rust
decodes BF16 to F32 and delegates a 27-row SGEMM to Accelerate. Token 15,745
discriminates the current endpoint output (`-4.59375`) from PyTorch
(`-4.5625`); the pinned specialized BF16 vector-dot topology rounds to the
PyTorch value.

Freeze the exact hash-bound final-norm row and token-15,745 weight row with
PyTorch BF16 result, specialized raw F32 result, alternative reductions, and
the mismatching PW-0089 logit. Change only the LM-head path: compute only the
last row required for next-token decoding and use the gated specialized BF16
dot for each contiguous weight row. Preserve BF16 output rounding, tensor
authority checks, ledgers, release, and all prior fixtures.

Pass the complete test suite, then repeat the full-prefix trace against the
immutable PW-0060 oracle. Embedding, all layers, final norm, every logit,
route, and expert set/order must meet existing gates. Retain normative Gate 8,
batch 1, concurrency 1, accepted tokens 0, buffer release, allocator relief,
and complete wall time. This is a correctness experiment and cannot count as
TPS or alter any threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
