# PW-0119 — Best-rank real-expert activation control

- Status: complete
- Disposition: conditional
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

The clean implementation at `ef5de9f1261e1e0d3b17f5a315c4991eb8dc67de`
completed all 18 full singular decompositions and six expert activation
controls in 94,299.533 ms. The PyTorch source oracle reproduced every captured
BF16 expert-down value bit-for-bit, closing the corpus/orientation/dynamic-FP8
control before interpreting any approximation.

The independent rank controls separate the unusually compressible early layer
from the deeper model:

| Layer | rank 128 relative-L2 range | rank 512 range | rank 768 range |
| ---: | ---: | ---: | ---: |
| 4 | 0.0838--0.2131 | 0.0305--0.0595 | 0.0198--0.0208 |
| 24 | 0.9633--0.9926 | 0.8277--0.8705 | 0.7098--0.7832 |
| 46 | 0.8701--1.0512 | 0.6748--0.8999 | 0.5694--0.7143 |

Errors decrease monotonically with rank for every sampled expert, but rank 128
is never competitive with rank 768: its routed-output error is 1.27--10.26x
higher. More importantly, even independent best rank 768 leaves 56.9--78.3%
relative L2 in the layer-24/46 controls. Their projection-space rank-768
Frobenius residuals remain 43.2--56.3%, compared with 8.59--19.03% at layer 4.
This is strong evidence that ordinary source-weight MSE/SVD is a weak fitting
objective for the deeper expert sample. It does not lower-bound a fit trained
on the frozen activations.

Gate 8 passed with 81% minimum free memory, 1,036,451,840-byte peak RSS,
339,954,816-byte maximum physical footprint, 224,594,560 bytes at the largest
release boundary, zero swap growth, zero throttling, and stable protected
services. The raw report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0119/run-001.json` hashes to
`3e7729dfff3d9ab6793d8e74d29ad20bb3c877bea328ae53d9325737c717c8fb`.
The independent analysis at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0119/analysis-001/manifest.json`
hashes to
`166f56b0b56c82099520acd6696647d8bc350b52d5b33d8649d51a7971cf7a34`.
There are zero accepted tokens and no TPS claim.

## Decision

Do not begin with the cheapest `(r=128,m=32)` weight-MSE bank merely because
PW-0118 proved that optimizer shape fits. Begin with a bounded rank-768
activation-weighted pilot on the predeclared experts and PW-0116 partitions,
after separately proving that rank-heavy optimizer embodiment passes Gate 8.
The candidate must compare against this independent rank-768 control and may
not infer broad representation quality from layer 4.

If activation-weighted rank 768 cannot materially improve the deeper held-out
expert outputs, kill the current low-rank identity-basis family before fitting
a full layer. If it can, only then test whether shared bases approach that
independent activation-weighted control. No result here promotes a stored
artifact, kernel, endpoint, or performance default.
