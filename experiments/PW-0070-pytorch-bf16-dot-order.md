# PW-0070 — PyTorch aarch64 BF16 dot-product order

- Status: complete
- Disposition: promoted correctness repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0069 comparison
  `01f212e3f16e0f3d5f03899a9608272c770691a1d0ae4777d186635fa792ba98`
- Hardware/runtime: Apple M1 shared 16 GiB host; installed PyTorch 2.13.0
  commit `cf30153c4c131c8164ee7798e5022d810682e2cb`; Rust native runtime
- Related records: PW-0065, PW-0066, PW-0068, PW-0069

## Hypothesis and contract

PW-0069's only first-stage mismatch is a length-192 BF16 query/key dot. The
pinned PyTorch fallback GEMM source computes reduced-precision dots as four
independent F32 partial sums, then adds partials 1, 2, and 3 into partial 0.
Rust currently performs one forward F32 sum. On the failing pair, the two sums
straddle a BF16 rounding boundary before attention scaling.

Create a small deterministic fixture from the hash-bound PW-0069 oracle query,
key, and sink captures. It must contain the exact 192 BF16 input bit patterns,
the pinned-source four-lane and old forward F32 replay bits, the PyTorch
post-dot BF16 bits, the scaled BF16 score bits, the row maximum, and the
centered-score bits. Record source manifest and capture hashes; do not commit
full real-model captures.

First prove that a source-derived four-lane replay matches the installed
PyTorch bits and the failing centered score while forward accumulation does
not. Then implement one named four-lane dot helper, gate it with the fixture,
and replay the complete layer-7 trace. The repair is promoted only if all
preexisting Rust/Python tests and strict Clippy pass, the complete layer-7
comparison is exact or advances the first boundary, and every shared-host
safety stop remains enforced.

Generated evidence stays external. This experiment changes no model identity,
hosted threshold, acceptance threshold, or throughput constant.

## Result

The hash-bound fixture reproduces the exact discriminator. The old forward F32
sum is `0x41a77fff`; the pinned-source four-lane sum is `0x41a78001`. BF16
rounding therefore produces `0x41a7` versus PyTorch's `0x41a8`, despite only a
two-ULP F32 difference. The four-lane replay yields PyTorch's scaled score and
centered `0xbe68` score exactly. Fixture SHA-256:
`c46f9b963e5b32f1e767e6278b39473155ee729f5d6d565cb84ea433e4786b59`.

The runtime now uses one named four-part F32 reduction for BF16 attention dots.
All 33 Rust tests, 42 Python tests, strict Clippy, deterministic fixture
regeneration, release compilation, Metal fixture checks, and the MLX smoke test
pass. The optimized fixture test proves the compiler preserves the required
bits.

The complete layer-7 replay finished in 118.330 seconds. All 21 captured
tensors are bit-exact against the unchanged PW-0069 oracle, including centered
scores, probabilities, attention, projection, both residuals, router logits and
scores, every selected-expert tensor, scatter, and final state. Expert sets and
order are exact; JSON route-weight values differ by at most
`7.428169246370686e-9`, below the `5e-7` gate. There is no remaining layer-7
failure.

The run peaked at 720,453,632 bytes RSS and 606,757,056 bytes physical
footprint, returned to 124,440,704 bytes after captures, retained at least 81%
system-free memory, grew no swap, observed no throttling, and retained every
protected service. Evidence hashes:

- Rust manifest:
  `626fc218e02e9d66b1f2f2359202f87e2a71d3c4dd66ee7cfa3551804aeaf8ca`
- Comparison:
  `62ea2df8ba4494959e5fdb9544af7ac758b31e54a73b7508bdc17212dd14d472`

## Decision

Promote the pinned-source four-lane BF16 dot reduction as a correctness repair.
It removes PW-0069's causal first difference and clears layer 7 without
weakening a gate. The next cheap discriminator is one full-prefix replay from
the frozen oracle to locate the next accumulated boundary. No throughput or
hosted threshold changes.
