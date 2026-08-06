# PW-0098 — Bounded Metal one-row routed layer

- Status: complete
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract `9cc629e`; final diagnostic/safety
  implementation `e8a12e69a46b08bcc5780b78c2fd88cfbf892448`; clean tree
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

The independent verified-checkpoint oracle is under
`/Users/chad/Models/mimo-prismwing/evidence/PW-0098/reference-002`.
Its manifest hashes to
`5884217fbc804a7a34bc76534b985eb7e6fe90f5e49e27e6328bda8584607cda`
and its BF16-widened routed output to
`6e3e3fe76c20b8ebb88ca7bafe212ae4d041cc914083dd3af3d1d64bf6c52779`.
It completed in 2,237.637 ms, peaked at 369,262,592 resident bytes, retained
79% free memory, caused no swap growth or throttling, and preserved all four
protected services.

Two authority defects failed closed before the numerical run. First, the old
PW-0039 helper returned descending top-k order while the source uses PyTorch
`topk(sorted=False)`; the existing source-exact bridge repaired the causal
route order. Second, PW-0037's route weights belonged to its FP16-rounded
fixture input. PW-0098 applies the target-faithful BF16 boundary, so its
verified-checkpoint oracle became route-weight authority while PW-0037 remains
input, selected-set, artifact, and tensor inventory authority. One expert also
legally split a weight and scale across two artifacts; the final loader
validates those mapped views jointly. No gate was relaxed.

The final candidate recomputes the exact source unsorted order
`[182,208,185,3,152,8,63,98]` and reproduces all eight BF16 route weights
bit-for-bit. Two clean numerical runs report identical correctness diagnostics
and 55.9013/55.8679 ms medians, a mean of 55.8846 ms and a 56.90x diagnostic
gain over PW-0096's 3,180 ms routed-layer attribution. The timing threshold
passes, but correctness does not:

- routed output relative L2 `9.59021e-4` versus the `5e-4` gate;
- maximum absolute error `1.19209e-7`, well inside the `2e-2` gate;
- BF16 identity 92.2363% versus the 99% gate;
- no accepted output file is written.

Per-expert captures localize the failure. Experts 208, 185, 3, 152, 8, 63,
and 98 match 4,092--4,096 of 4,096 BF16 values. Expert 182 matches only 2,992,
with maximum absolute error `4.76837e-7`. All six expert-182 weight/scale raw
tensor ranges were independently compared between selected artifacts and the
verified SSD checkpoint and are byte-identical, excluding artifact corruption.
The residual is the 64-lane Metal accumulation crossing source BF16 boundaries
for that expert.

The final fail-closed artifact is
`candidate-007/error.txt`, SHA-256
`1ffa33d7a7f4d2742e142db65f4267e5ee7f9691c7c6666dbad4e140aa30c3c0`.
It records a 55.9615 ms median, 79% minimum free memory, 253,345,792-byte peak,
29,509,568-byte post-release footprint, zero swap growth, zero new throttled
pages, and successful protected-service checks. Thus the rejected path also
closes Gate 8 after buffer release.
The updated throughput model hashes to
`6777a425ee5344ed89e17bfacb901a7ba71572198caba8d2e68a972358b681cb`.

## Decision

Reject the uncorrected 64-lane source-FP8 Metal routed-row candidate. Its
performance and bounded-memory mechanism are real, but PW-0097's single-expert
success does not generalize across expert numerical distributions. Do not
integrate this path into the complete token or claim its component rate as TPS.

Promote the precise risk frontier instead: expert 182 requires a production-
bounded BF16 boundary repair or a closer source reduction topology. The next
experiment should capture gate/up/SwiGLU/down boundaries for that expert,
identify the minimum uncertain-row set, and test sparse source-exact row
recomputation while retaining the 100 ms routed-row budget. Preserve the seven
passing experts as controls.
