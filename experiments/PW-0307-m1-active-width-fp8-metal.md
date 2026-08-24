# PW-0307 — M1 active-width block-scaled FP8 Metal

- Status: complete
- Disposition: rejected as production default; exact kernels and fixtures retained
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

The candidate implementation is clean commit
`55eaea1d59d92bced550c4989129c82013cac057`. The matched control is clean commit
`781b965f493137a84133a837b71b16a968250d25`: it retains the new kernels and
fixtures but forces the established batch-eight FP8 dispatch. After the
experiment, production dispatch was restored to that control behavior while
the exact research kernels and fixtures were retained. The rejected single-row
pipeline objects are compiled only in test builds, so production pays no eager
pipeline-compilation cost for them.

All three endpoint runs use checkpoint revision `63651580`, model-lock SHA-256
`df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050`,
checkpoint-verification SHA-256
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`,
kernel SHA-256
`0ab9e7647e19b4fb0bea20d97ea3158ef42cec52a44d7cd2ac2b8b2a48148060`,
the PW-0208 ordinary prompt, seven requested and accepted tokens, cold process
start, batch one, concurrency one, and verifier width four.

## Isolated attribution

The summed layer-zero wall over the two target transactions is 552.689 and
552.680 ms for the candidates versus 619.720 ms for control. The candidate
median improves this production-shaped interval `1.121290x`. Complete target
verification is 53,684.278 and 53,452.703 ms versus 54,271.876 ms; its candidate
median improves `1.013131x`. These are causal diagnostics because the external
CPU proposer is unchanged and measured separately.

Candidate post-prefill proposal-plus-verification walls are 79,394.063 and
72,973.290 ms. Their 76,183.676-ms median improves over control's 76,736.144 ms
by `1.007252x`. This smaller slice result does not replace the complete-path
gate.

## End-to-end result

Candidate-control-candidate complete walls are 339,042.852, 330,763.011, and
326,432.534 ms. Candidate spread is 3.790%; their 332,737.693-ms median is
`0.0210376` accepted TPS versus control's `0.0211632`. The complete accepted-TPS
ratio is `0.994065x`, a 0.597% regression, so the predeclared promotion gate
fails even though Candidate 2 alone is faster than control.

Prefill walls are 259,648.789, 254,026.866, and 253,459.244 ms. External CPU
proposal walls are 24,391.722, 21,169.215, and 18,787.973 ms. Those independent
variations explain why the lower-level target gain does not establish a
repeatable complete-path gain.

## Correctness result

All 107 Rust tests pass on the target, including generic block-scaled CPU parity
at active widths 1, 2, 4, 9, 26, and 32 and a production-shaped exact fixture
for the 13,568-by-4,096 full-QKV and 14,848-by-4,096 sliding-window-QKV one-row
kernels. The latter compares single-row and batch-eight controls byte for byte
after BF16 output staging. The release build passes.

Every endpoint emits token IDs `[32,3283,646,11941,7949,7324,8628]`, preserves
`A=[3,3]`, `U=[6.053191489361702,5.377659574468085]`, and moves
427,197,245,056 logical source bytes. Process reads are 420,418,093,056,
421,416,771,584, and 420,170,924,032 bytes. Peak resident-byte ledgers are
4,502,667,264, 4,508,778,496, and 4,512,661,504 bytes. All runs record zero
swap growth and zero new throttled pages.

Candidate 1 report and progress hashes are
`d04bd8f6762802b7cf9b01db6965a039c4b9faa7202a5067f6481d2b39323be5` and
`50ebff282c79268d5836daf96ce685ea08c6b7e1dfb5b27ee2e995eb25adc5d3`.
Control hashes are
`0a466cfd4f4848c4807d9dd27d14f6125529c1d9123d0714973aeba7b1f417b8` and
`1e2cfc647b58cda6ab2fbee4737ccd472e088abfd35de3d84a8e85e41ac09dd2`.
Candidate 2 hashes are
`56e271f0e70600638a1208b8389c05fc9f1212bfcaad34df669e9ccbf2285758` and
`812a1e1760f8b3c3817467a26f1540cf1385f09ecf158f9ace588bffe2b724d4`.

## Decision

Reject active-width FP8 selection as a production default and restore the
established batch-eight dispatch. Preserve the exact kernels and correctness
fixtures as research material: they prove a real target-local opportunity, but
this realization is too small and noisy to improve repeatable full-path TPS.

This supersedes the untested belief that the stronger worker's active-width FP8
dispatch could be imported directly as a target default. It does not change a
throughput-model constant or the promoted PW-0306 BF16 result. The handoff's
dequantized association remains excluded because it changes arithmetic, and
the resident-cache branch remains target-inadmissible.
