# PW-0098 — Bounded Metal one-row routed layer

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0037 manifest
  `a2b30be7ab767c754fd4680887420246dcd28d314bff242b144f664a8ff12470`;
  input `ac6776035eee0537ab0d7d7975d4ad92e08bf67930b58d47a4d9f2e051113150`
- Hardware/runtime: Apple M1 shared 16 GiB host; Rust-owned Metal plus bounded
  CPU source staging; real layer-43 row-zero route
- Related records: PW-0037, PW-0039, PW-0049, PW-0096, PW-0097

## Hypothesis and mechanism

PW-0097's reusable source-faithful executor can process all eight experts of
one real routed row without retaining a model-wide expert bank. Use row zero
of PW-0037's frozen real layer-43 input and independently recompute its native
router decision. For experts `[182,208,185,3,152,8,63,98]`, install one
gate/up/down tensor set at a time, execute exact dynamic-FP8 and BF16 staging,
apply the computed route weight, scatter-add in F32, round the routed result to
BF16, release buffers, then advance to the next expert.

The independent PyTorch oracle must read the verified full checkpoint, not the
selected artifacts or candidate output. The candidate may use the pinned
selected tensor artifacts only after validating every artifact and tensor
identity from the PW-0037 manifest.

## Gates

Fail closed on manifest, input, artifact, tensor, route, dtype, shape, scale
grid, non-finite, duplicate-expert, top-eight-boundary, output, commit, or
create-new mismatch. Add fixtures for generalized expert naming, manifest row
selection, weighted scatter order, BF16 final staging, and bounded executor
reuse before execution.

The candidate-selected expert order must equal the independent oracle exactly;
maximum route-weight error must remain at most `3e-8`. All 4,096 routed outputs
must be finite. Relative L2 against independent PyTorch must be at most `5e-4`,
maximum absolute error at most `2e-2`, BF16 equality at least 99%, and two
clean processes must produce byte-identical output.

Compile once, perform five warmups and 30 complete routed-row measurements.
Each measurement includes native routing, all eight experts' dynamic activation
staging, tensor-to-Metal installation, dispatch/wait/readback, BF16/SwiGLU,
weighted scatter, final BF16 round, and buffer release. Median must be at most
100 ms in both processes and at least 10x faster than PW-0096's 3,180 ms mean
routed-layer attribution. Record cold and warm OS/application state, p10,
median, p90, bytes, batch 1, concurrency 1, accepted tokens 0, `A=0`, `U=8`,
hardware, compiler, commit, and source hashes.

Enforce Gate 8 before compilation, after warmups, after the timed series, and
after buffer release: at least 20% free memory; current/peak process footprint
at most 8 GiB; post-release footprint at most 4 GiB; swap growth at most
512 MiB; zero new throttled pages; ChatGPT, WindowServer, nxnode, and syncthing
remain healthy. This component cannot become accepted TPS or the endpoint
default. A passing result only authorizes complete-token candidate integration.

## Result

Unexecuted.

## Decision

Unexecuted.
