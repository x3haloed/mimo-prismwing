# PW-0073 — PyTorch specialized BF16 vector-dot order

- Status: in progress
- Disposition: unexecuted
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

Unexecuted.

## Decision

Unexecuted.
