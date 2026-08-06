# PW-0097 — BF16-staged Metal one-row expert

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0034 input
  `ad261a98cc64c34277a40168f45654cabb1c1059e88771c3c71092ae6ffee5ba`;
  checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
- Hardware/runtime: Apple M1 shared 16 GiB host; Rust-owned Metal plus CPU
  source staging; real layer-43 expert-32 source tensors
- Related records: PW-0033 through PW-0036, PW-0095, PW-0096

## Hypothesis and mechanism

The existing Metal source-FP8 projection kernel can replace per-matrix F32
expansion for a production-shaped one-row expert while preserving the source
execution contract around it. Quantize/dequantize each input activation group
exactly as the endpoint, project gate and up directly from source FP8, round
both to BF16, apply the source SwiGLU staging, round its result to BF16, project
down directly from source FP8, and round the final expert output to BF16.

Compile one reusable Metal executor. Each timed execution must still install
the real expert's weight/scale views into bounded shared Metal buffers; do not
hide a 25 MB per-expert copy in setup or retain hundreds of experts. Use the
frozen PW-0034 one-row input and a newly generated independent PyTorch oracle
with the exact dynamic-FP8/BF16 source boundaries.

## Gates

Add deterministic tests for dynamic activation bytes/dequantization, BF16
gate/up/SwiGLU/down staging, shape/layout rejection, non-finite rejection, and
create-new output. Gate/up/down tensor names, shapes, source bytes, scale grids,
input hash, checkpoint revision, and reference hash fail closed.

All 4,096 outputs must be finite. Against independent PyTorch, relative L2 must
be at most `5e-4`, maximum absolute error at most `2e-2`, BF16 equality at least
99%, and the topological error must not exceed the existing target-faithful
component gates. Two clean candidate processes must produce byte-identical
output.

After one untimed compile and five warmups, run 30 complete expert executions
including dynamic activation staging, real tensor-to-Metal buffer installation,
gate/up dispatch, CPU BF16/SwiGLU staging, down dispatch, completion waits, and
readback. Median must be at most 100 ms in both processes and improve at least
10x over PW-0096's 3,180 ms average routed layer divided by eight experts
(`397.5 ms/expert`). Record cold first execution, p10/median/p90, bytes, RSS,
batch 1, concurrency 1, accepted tokens 0, `A=0`, `U=1`, device, compiler,
commit, and cache state.

Enforce Gate 8 before compilation, after warmups, after the timed series, and
after buffer release. This component cannot become accepted TPS or a
performance default. Endpoint integration requires a later real routed-layer
and complete-token experiment.

## Result

Unexecuted.

## Decision

Unexecuted.
