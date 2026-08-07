# PW-0119 — Best-rank real-expert activation control

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0118 analysis
  `25f71aabb3d66f3142c8ff8447c451c61d7f527b79a02638b256916fe0db778e`
- Hardware/runtime: Apple M1 shared 16 GiB; NumPy Accelerate SVD, PyTorch
  source-FP8 oracle; verified internal-SSD checkpoint
- Related records: PW-0045, PW-0115, PW-0116, PW-0117, PW-0118; E5

## Question and causal mechanism

Before optimizing shared bases, establish the strictly stronger independent
per-expert low-rank control required by PW-0045. Every identity-basis expert
matrix has rank at most `r`; an independently optimal truncated SVD at the same
rank has no sharing constraint and is therefore an optimistic matrix-space
control. Measure its real activation behavior, not only singular-value energy.

This control does not by itself lower-bound activation-weighted training: a
rank-`r` fit optimized specifically for the finite corpus could trade matrix
error for lower routed error. It does establish the baseline that a shared
candidate must beat at matched rank and exposes whether the frozen ranks are
even near a plausible source-weight reconstruction regime.

## Frozen sample and execution

Use the predeclared PW-0116 hot/rare identities:

- layer 4: hot expert 64 (181 placements), rare expert 10 (1);
- layer 24: hot expert 23 (167), rare expert 101 (1); and
- layer 46: hot expert 28 (212), rare expert 0 (1).

For every expert, authenticate and dequantize gate, up, and down source-FP8
matrices. Canonicalize down by transposing it to `[2048,4096]`. Compute complete
singular spectra and exact best truncated reconstructions at ranks 128, 512,
and 768. Report relative Frobenius residual energy for every projection.

Gather the exact PW-0116 `moe_input` rows and expert-major `expert_down` truth.
Reproduce the source expert with PyTorch's frozen dynamic-FP8 group-128 input,
decoded source weights, BF16 projection boundaries, BF16 SiLU, and BF16
product. Fail if its down output differs materially from the Rust capture.
Then replace all three matrices with each F32 truncated-SVD control while
preserving the same activation quantization and BF16 boundaries. Report
expert-down relative L2 and maximum absolute error overall and separately for
train `0..111`, validation `112..167`, and pilot holdout `168..223` whenever
the expert occurs.

Use one process, release every expert's SVD matrices before the next, and apply
Gate 8 after source open, each projection spectrum, each expert evaluation,
release, and final service health. Write metrics only; never write reconstructed
weights.

## Gates and interpretation

1. All checkpoint, corpus, tensor, schedule, position, and payload hashes must
   match; each captured expert row must map to exactly one scheduled position.
2. The source PyTorch oracle must match captured expert-down BF16 values with
   relative L2 at most `1e-3` and maximum absolute error at most `0.02`; record
   exact equality fraction rather than claiming bit identity.
3. Singular values must be finite, descending, and reconstruct total matrix
   energy; residual errors must decrease monotonically with rank.
4. Candidate expert-output errors must be finite and reported for every
   available frozen partition without dropping rare rows.
5. Gate 8 must pass below 8 GiB peak/current, below 4 GiB after release, below
   512 MiB swap growth, with no throttling or lost protected services.

This experiment cannot promote or finally kill shared bases. It chooses the
first fitting target: if rank 128 is already competitive with rank 768 on real
expert outputs, fit `(128,32)` first; otherwise the shared optimizer must begin
with the lowest-error eligible rank and prove it can approach the independent
control. If even rank 768 has large errors, record that the weight-MSE path is
weak and add activation-weighted fitting before spending on a full bank. No
kernel, endpoint output, accepted token, or TPS claim is allowed.

## Result

Unexecuted.

## Decision

Unexecuted.
