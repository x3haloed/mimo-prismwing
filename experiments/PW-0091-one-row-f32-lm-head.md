# PW-0091 — One-row F32 LM-head matrix path

- Status: complete
- Disposition: correctness-repair; full-prefix parity
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

The one-row F32 matrix path passes all 38 Rust tests, 42 Python tests, strict
Clippy, and every preserved fixture. The full production replay is bit-exact
against the frozen PyTorch oracle for embedding, all 48 layer finals, final
RMSNorm, all 152,576 last-token logits, every route weight, and every expert
set/order. `first_failure` is null and `full_prefix_provisionally_cleared` is
true.

This confirms matrix shape was the remaining arithmetic boundary: preserving
the F32 widening, full-matrix backend, and BF16 output rounding while reducing
the endpoint from 27 unnecessary prompt rows to the one authoritative decoding
row reproduces the oracle exactly. The rejected PW-0090 specialized-dot path
remains historical evidence and is not production authority.

The run completed in 798.639 seconds, effectively unchanged from PW-0089 and
PW-0090, so no full-path performance gain is promoted. It peaked at
3,928,424,448 bytes RSS, ended with a 2,667,289,472-byte footprint, retained at
least 81% free memory, grew no swap, observed no throttling, and kept every
protected service healthy. Evidence hashes:

- Rust manifest:
  `87466b59480a5a5b4256c490f1dfe670fe09f28d21d169085ab13bb1b4b7ab59`
- Comparison:
  `34f1d6e28622d66409d46e7407a9e54532e03821ea7dd36e65e94b50045216db`

## Decision

Promote the source-faithful one-row F32 LM-head matrix path as the correctness
repair. The complete local prefill and next-token logit path is now bit-exact
to the frozen PyTorch oracle without weakening any gate. Retain last-row-only
projection as an embodiment simplification, but do not claim speedup from
these non-interleaved full walks. Advance to complete generation semantics and
end-to-end cold/warm throughput measurement under the unchanged target.
