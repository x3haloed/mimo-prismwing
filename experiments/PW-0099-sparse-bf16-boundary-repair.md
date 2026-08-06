# PW-0099 — Sparse BF16 boundary repair

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0098 oracle manifest
  `5884217fbc804a7a34bc76534b985eb7e6fe90f5e49e27e6328bda8584607cda`;
  PW-0098 rejected evidence
  `1ffa33d7a7f4d2742e142db65f4267e5ee7f9691c7c6666dbad4e140aa30c3c0`
- Hardware/runtime: Apple M1 shared 16 GiB; bounded Metal plus source-exact
  sparse CPU correction
- Related records: PW-0096, PW-0097, PW-0098

## Hypothesis and mechanism

PW-0098 localizes routed-row numerical failure to expert 182 while seven
controls already pass. Capture expert 182's gate, up, SwiGLU, and down BF16
boundaries independently. Compare the pre-round Metal F32 value to its nearest
BF16 midpoint and derive a conservative uncertainty interval from observed
projection error plus an explicit margin. Recompute only uncertain output rows
from source FP8 weights using the source-exact CPU matrix reduction, replace
their BF16 values, and propagate normally.

The uncertainty predicate must be value-derived and fixed before the final
timed runs. It may not encode expert IDs, row IDs, expected values, oracle
hashes, or route outputs. Expert 182 is the discovery fixture; at least the
seven PW-0098 experts and PW-0097 expert 32 are mandatory holdouts.

## Gates

First add independent gate/up/SwiGLU/down captures and report exactly where
expert 182 first diverges. Preserve raw hashes and reject any unexplained
upstream mismatch. Add tiny fixtures for BF16 midpoint distance, conservative
error intervals, uncertain-row selection, sparse row decode, source-exact
reduction, replacement order, empty/full repair sets, and fail-closed shapes.

On discovery and every holdout expert, repaired final expert output must meet
PW-0097's unchanged gates: relative L2 at most `5e-4`, maximum absolute error
at most `2e-2`, and BF16 identity at least 99%. The complete PW-0098 routed row
must then meet its unchanged output and route gates in two clean processes and
produce byte-identical output. Report the uncertain/recomputed row count and
fraction separately for gate, up, and down; no oracle-derived row selection is
allowed during candidate execution.

Include uncertainty selection, sparse source decode/reduction, and replacement
inside five-warmup/30-measurement timing. Both routed-row medians must remain at
most 100 ms and at least 10x faster than the 3,180 ms CPU attribution. Record
logical bytes, decoded bytes, batch 1, concurrency 1, accepted tokens 0,
`A=0`, `U=8`, commit, cold/warm state, and interleaved uncorrected controls.

Apply the complete Gate 8 contract at compile, warmup, timed-series, and
post-release boundaries. This remains a component experiment; passing only
authorizes complete-token candidate integration and is not accepted TPS.

## Result

Unexecuted.

## Decision

Unexecuted.
