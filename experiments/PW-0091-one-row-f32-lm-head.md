# PW-0091 — One-row F32 LM-head matrix path

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0090 comparison
  `efd792d768264db4a2c73d365c2a003fa352664abaab3545ab48d620a8840d7f`;
  frozen PW-0060 oracle
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  PyTorch 2.13.0 CPU oracle, production Rust endpoint
- Related records: PW-0060, PW-0089, PW-0090

## Hypothesis and contract

PW-0090 proves the complete prefix formally clears but its specialized BF16
dot is not the frozen oracle operation. Oracle `bf16_linear` widens BF16 input
and weights to F32, performs the matrix multiply, then rounds to BF16. The
original Rust endpoint did the same representation conversion but projected
all 27 prompt rows; its last-row reduction therefore used a different SGEMM
shape from the oracle's one-row LM-head call.

Use the existing production F32 matrix backend and BF16 rounding on only the
last normalized row required for decoding. Preserve exact tensor layout
checks, logical-byte and matrix ledgers, transient release, the rejected
PW-0090 fixture as historical evidence, and every prior semantic fixture.
Do not introduce a new arithmetic kernel.

Pass the complete test suite, then repeat the full-prefix trace against the
immutable PW-0060 oracle. Prefer bit-exact logits; at minimum, the already
cleared formal gate, exact chosen logits, transformer parity, routes, and
expert sets/order may not regress. Retain normative Gate 8, batch 1,
concurrency 1, accepted tokens 0, release, allocator relief, and complete wall
time. This cannot count as TPS or alter any threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
