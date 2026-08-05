# PW-0047 — Texture-native executable weight codec

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: unassigned
- Commit and dirty state: proposal based on clean `4eedd12`; no execution
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; candidate encoder and encoded
  artifact hashes must be recorded
- Hardware, OS, compiler, storage, memory pressure: Apple M1 target; not measured
- Related records: PW-0008, PW-0011, PW-0034, PW-0039, PW-0043; black-swan
  budget in `docs/EXPERIMENTS.md`

## Hypothesis and mechanism

Apple GPU fixed-function texture decode and texture caches may embody low-bit
expert tiles more efficiently than ALU-based FP8 or INT4 unpacking. Encode a
production-shaped projection into a supported compressed texture
representation, fetch decoded blocks through Metal texture operations, and
consume them directly in the dot-product path without expanding a large
temporary matrix.

The proposition is not that texture decompression is fast in isolation. It is
that fixed-function decode plus complete projection moves fewer executable
bytes and executes faster at acceptable error.

## Contract

This is an explicitly modified L3 representation named `texture-weight`; it is
never reported as source FP8 or exact. The investigation is limited to one week
and the fixed kill test below. Unsupported devices or formats fail closed; no
silent buffer-kernel fallback may be timed as the candidate.

Pass only if:

1. the encoder, block layout, channel interpretation, normalization, address
   mode, filtering mode, and texture format are deterministic and captured in
   a tiny scalar-decodable fixture before the production kernel;
2. the benchmark covers an actual 4,096×2,048 or 2,048×4,096 expert projection
   and then one complete gate/up/SwiGLU/down expert at batch one and eight;
3. reports distinguish stored bytes, texture allocation bytes, decoded values,
   temporary buffers, bytes fetched, ALU work, cache state, compile/install
   time, energy when available, and complete wall time;
4. the complete expert is at least 1.5x faster than the promoted PW-0034/PW-0036
   control at the same batch and reduces executable representation bytes by at
   least 40%;
5. local complete-expert relative L2 is at most 1%, with downstream routed-MoE
   and layer-local gates required before further integration; and
6. cold and warm paired controls confirm that the gain survives outside the
   texture cache.

Kill immediately if the format requires materializing an expanded weight
buffer, if complete projection does not beat the control by 20% in the first
production-shaped test, or when the one-week budget expires without the full
gate.

## Baseline and candidate

Baselines are the promoted direct source-FP8 Metal expert and the strongest
measured low-bit buffer kernel at matching batch. Candidate is texture decode
inside the complete arithmetic path. A texture-only bandwidth number cannot
promote or extend the experiment.

## Isolated attribution

Unexecuted. Verify actual device and allocation bytes rather than inferring
them from nominal format bits per pixel.

## End-to-end result

Unexecuted. No endpoint or accepted-TPS claim exists.

## Correctness result

Unexecuted. A new fixture is required before any new texture indexing or decode
semantic is trusted.

## Decision

Unexecuted. This is deliberately time-boxed black-swan work; preserve a
negative result and return to the main path when its gate fails.
