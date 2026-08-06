# PW-0090 — Specialized BF16 LM-head reduction

- Status: complete
- Disposition: rejected partial repair; gate-clearing diagnostic
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

The 4,096-value fixture is deterministic with hash
`33e0a1deacb9b5dba1c7fad504598fbad3fad377836d9b646d5fdf5aacb124b5`.
It proves the specialized BF16 dot rounds token 15,745 to the oracle value,
and the implementation passes all 38 Rust tests, 42 Python tests, strict
Clippy, and fixture regeneration.

The full replay falsifies the broader operator hypothesis. The frozen oracle's
`bf16_linear` explicitly widens BF16 activations and weights to F32, performs
the complete matrix multiply, then rounds the result to BF16. A standalone
BF16 contiguous dot is not that operator. It removes 25 of PW-0089's 45 logit
differences, but leaves 20: equality is 99.9869%, relative L2 is
`6.744156149382939e-6`, and maximum error is `0.00390625`.

All existing formal gates nevertheless pass: `first_failure` is null,
`full_prefix_provisionally_cleared` is true, both hosted-chosen logits are
exact, every transformer capture is exact, and routing is exact. The run
completed in 799.776 seconds—no material full-path speedup over PW-0089—peaked
at 3,924,803,584 bytes RSS, ended with a 2,662,243,264-byte footprint, retained
at least 81% free memory, grew no swap, observed no throttling, and kept every
protected service healthy. Evidence hashes:

- Rust manifest:
  `038107d4a368560ce034073aa9302c1d90d6bb214716614787dd9b8f5e2a205c`
- Comparison:
  `efd792d768264db4a2c73d365c2a003fa352664abaab3545ab48d620a8840d7f`

## Decision

Do not promote the specialized BF16 dot as LM-head authority despite formal
gate clearance. Preserve the result as evidence that removing 26 unnecessary
prompt-row projections is semantically safe but not yet a measured full-path
gain. Supersede the arithmetic with the oracle-faithful F32 matrix path applied
only to the required last row, then repeat the full trace. No gate changes.
