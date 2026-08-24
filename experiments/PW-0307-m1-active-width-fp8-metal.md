# PW-0307 — M1 active-width block-scaled FP8 Metal

- Status: running
- Disposition: unexecuted
- Date: 2026-08-24
- Owner: Codex
- Starting commit: clean `cb3c06f6a747f9436f34acc95daeda7a237182f1`
- Hardware: Apple M1 Mac mini (`Macmini9,1`), 16 GiB unified memory, macOS
  26.6.1 (25G76)
- Related records: PW-0195, PW-0205, PW-0207, PW-0306; stronger-worker PW-0305
  handoff at `/private/tmp/mimo-exact-0.2tps-worker-handoff/`

## Hypothesis and mechanism

The wide FP8 linear currently pads every one-to-seven-row dense projection to
eight rows. Existing generic kernels already specialize widths one through
eight, while the stronger-worker handoff supplies single-row full-QKV and
sliding-window-QKV kernels. Dispatch the exact active width for generic dense
FP8 projections and the new single-row kernel for QKV; retain batch eight for
QKV widths two through eight.

Unlike the handoff's selected dequantized one-row path, this candidate preserves
SGLang's separate activation codes/scales and its per-block reduction topology.
It removes unused positions without changing arithmetic or adding resident
model state.

## Contract

Target-faithful L1, function-preserving scheduling. Before endpoint use, add
byte-exact fixtures for generic widths one and four and production-shaped full
and sliding-window QKV single-row kernels against their batch-eight controls,
all after the existing BF16 output stage. Widths two through eight generic and
QKV batch-eight behavior must remain available unchanged.

Kernel-only timing is diagnostic. Promotion requires identical endpoint tokens,
`A`, `U`, logical bytes, and host safety plus a repeatable interleaved complete
accepted-TPS gain on this 16 GiB M1. The handoff's dequantized association and
large resident caches are explicitly out of scope.

## Baseline and candidate

Pending implementation commit and exact control.

## Isolated attribution

Pending.

## End-to-end result

Pending.

## Correctness result

Pending.

## Decision

Pending target-machine evidence.
