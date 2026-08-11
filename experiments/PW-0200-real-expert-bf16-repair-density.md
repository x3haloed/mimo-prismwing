# PW-0200 — Real-expert BF16 repair density

- Status: completed
- Disposition: promoted to selective exact-repair falsification
- Date: 2026-08-10
- Execution mode: target-faithful numerical diagnostic
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0101, PW-0196, PW-0197, PW-0199

## Hypothesis

Although an uncorrected wide transaction fails a cancellation-sensitive routed
relative gate, only a sparse subset of BF16 projection outputs may disagree
with the source authority on real model activations. If mismatch density is
small, a fast Metal projection plus selective source-calibrated repair can be
both exact and materially cheaper than full reproducible accumulation.

## Contract and gate

Audit all gate, up, and down projections for PW-0101's eight real selected
layer-4 experts against their frozen per-projection authorities. Preserve exact
dynamic-FP8 input preprocessing, BF16 output staging, original-shard page-
rounded no-copy binding, and the existing numerical thresholds. Diagnostic
mode may retain failed candidates but must report the frozen gate result,
mismatched BF16 count/fraction, error metrics, and zero accepted tokens. It may
not be presented as a passing projection or throughput result.

Promote selective repair only if mismatch density is low enough that correcting
the observed outputs cannot dominate the existing wide Metal cost. Otherwise
reject sparse repair and move to tile-level certified accumulation.

## Result

The audit covers all 65,536 outputs from 24 real PW-0101 projections. Only nine
BF16 values differ, an aggregate mismatch fraction of `0.0001373291` (0.0137%).
Fifteen projections are byte-exact and 19 of 24 pass the unchanged projection
gate. All tensors use original-shard no-copy binding. The manifest hashes to
`7ecc133ed4fa6319b6fda0ef3c74bc4c93e0428b29a583e4df6ab62b0a7748a5`.

The five failed gates are caused by one to three isolated BF16 disagreements;
the worst is expert-9 down with three mismatches. Promote a falsifier that asks
whether a high-precision dot selects the frozen source BF16 value at all nine
sites. Timing from this audit remains diagnostic, zero tokens are accepted, and
no endpoint constant changes.
