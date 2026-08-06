# PW-0073 — PyTorch specialized BF16 vector-dot order

- Status: complete
- Disposition: promoted correctness repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0072 comparison
  `eb764c15082cbe78c61ab59d80af6c4607ce502702e8bc99e1b427c79c52bc9d`
- Hardware/runtime: Apple M1 shared 16 GiB host; installed PyTorch 2.13.0
  commit `cf30153c4c131c8164ee7798e5022d810682e2cb`; Rust native runtime
- Related records: PW-0070 through PW-0072

## Hypothesis and contract

PW-0072's single first-stage mismatch is explained by the pinned PyTorch
reduced-precision GEMV fast path: eight four-element F32 vector accumulators
over 32-element blocks, pairwise accumulator reduction, then ARM horizontal
addition. The current four-lane helper represents a different fallback path.

Create a deterministic fixture from the hash-bound PW-0072 oracle query/key
captures at position 22, head 3, source token 16. Record the 192 BF16 inputs,
source-replayed vector-dot F32 bits, old four-lane F32 bits, PyTorch BF16 dot,
scaled score, row maximum, and centered score. Bind source manifest and capture
hashes; commit no full-model tensors.

Implement one named specialized vector-dot helper and select it for global
attention while retaining the four-lane SWA helper. The fixture must distinguish
both schedules in debug and optimized builds. Promote only if the full test and
strict-Clippy gates pass and a production layer-11 replay makes every capture
exact or advances the first boundary. Retain all normative Gate 8 stops.

This is correctness-only. It changes no model, hosted, capability, fidelity,
cost, power, safety, or throughput threshold.

## Result

The fixture distinguishes the schedules exactly. The specialized source replay
and PyTorch result cross the BF16 boundary required for centered score
`-1.421875`; the four-lane schedule yields `-1.4296875`. Fixture SHA-256:
`563387025dd8cc21d6d4b9f395e8d5601729b11bc937653d85adcefbbb5305d2`.

The first full regression run rejected the initial width-192-only helper: a tiny
global-attention fixture exercised PyTorch's vector/scalar tail behavior at a
non-32-aligned width. The implementation was not extended further until both
the 8-element vector tail and scalar tail were source-matched. The repeated
gate then passed 34 Rust tests, 42 Python tests, strict Clippy, deterministic
fixture regeneration, release compilation, Metal fixture checks, and the MLX
smoke test. Both debug and optimized builds preserve the exact discriminator.

The production layer-11 replay completed in 180.887 seconds. All 21 captures
are bit-exact against unchanged PW-0072 oracle run 002, including global
attention scores/probabilities, projection, both residuals, router internals,
every selected-expert tensor, scatter, and final state. Expert sets/order are
exact; JSON route-weight values differ by at most
`7.568359383647305e-9`, below the `5e-7` gate.

The run peaked at 749,305,856 bytes RSS and 658,890,944 bytes physical
footprint, ended at 355,574,016 bytes, retained at least 81% system-free memory,
grew no swap, observed no throttling, and retained every protected service.
Evidence hashes:

- Rust manifest:
  `15576207f5d58b098aab5ccf4eb412aac26790f56c5d0e8de45778f5c17dcd13`
- Comparison:
  `4f400322f1a25bb5469ca35836104d67a920b0f24648343ecfe83efb19238a17`

## Decision

Promote the specialized PyTorch BF16 vector-dot reduction for global attention
while retaining the four-lane SWA path. The exact accumulated frontier is now
through layer 11. The next cheap discriminator is one frozen full-prefix replay
to find the next boundary; no throughput or threshold changes.
