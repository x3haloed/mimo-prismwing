# PW-0126 — Routed-residual subspace oracle

- Status: completed
- Disposition: negative
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation
  `892fb4c3ee274467e7b9164e6b6fb030f5fec8e1`; clean tree at execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0125 analysis
  `b49bfe3082cc2a81ba87c717f9f493f22b7fb9204b6b586699bcce559c1b8fe8`
- Hardware/runtime: Apple M1 shared 16 GiB; NumPy Accelerate SVD; no model
  execution or checkpoint weight reads
- Exactness: explicitly modified L4 `mixture-compiled` capacity oracle;
  source-FP8 remains the target-faithful control
- Related records: PW-0045, PW-0114 through PW-0125; E5

## Question and changed premise

PW-0123 through PW-0125 reject two identity-basis families before executable
construction. They still represent selected experts individually. PW-0045's
deeper premise is that the transformer consumes only the complete weighted
routed residual, so a layer may compile that output directly without retaining
expert identities as executable matrices.

Before designing a router-conditioned coefficient network, test the necessary
representation floor of the simplest direct compiler: one mean plus a learned
linear output dictionary per routed layer. Grant an impossible oracle the true
best coefficient vector for every validation or holdout residual. If this
oracle cannot meet the local fidelity gates, no predictor over that fixed
dictionary can meet them. If it passes, only coefficient prediction—not output
subspace capacity—is authorized for the next experiment.

## Frozen data and selection

Use PW-0116 `routed_output` at layers 4, 24, and 46. For each layer:

1. Fit the F64 training mean and economy SVD of centered positions `0..111`
   only. Validate finite values, exact capture shape `[224,4096]`, and the
   manifest payload hash before fitting.
2. Evaluate oracle orthogonal projections at ranks `16,32,64,96,111` on
   validation positions `112..167`. Compute aggregate relative L2 and the same
   metric on two predeclared route-coverage slices: positions touching at
   least one expert absent from training, and positions whose selected experts
   are all present in training. Empty slices are reported explicitly and do
   not pass or fail.
3. Select the smallest rank whose validation aggregate is at most 1% and every
   nonempty slice is at most 2%. If none passes, stop that layer at rank 111 and
   do not evaluate its holdout.
4. Only after all three layers select a validation-passing rank, unseal
   positions `168..223` once at those fixed per-layer ranks. Apply the same
   aggregate and route-coverage slice gates. Do not report holdout curves or
   tune ranks from holdout results.

Route coverage is derived only from each layer's frozen
`selected_experts_by_position`: the training expert set is the union over
positions `0..111`. This deliberately exposes layer-4 and layer-24 route
distribution shift instead of dropping unseen experts.

Add deterministic fixtures proving centered projection, rank monotonicity,
selection without holdout access, unseen-expert slice construction, empty-
slice reporting, manifest/hash failure, and exact byte/compute algebra.

## Physical ledger and limits

Report a conservative F32 mean-plus-basis artifact:

`artifact bytes = 4 * 4096 * (rank + 1)`

and oracle output synthesis work:

`multiplications = rank * 4096`.

The source comparison is one layer's complete routed bank:
`256 * 25,171,968 = 6,444,023,808` bytes, and one ordinary selected mixture's
source projection work: `8 * 3 * 2048 * 4096 = 201,326,592`
multiplications. Coefficient prediction, routing, quantization, BF16 staging,
and executable wall time are explicitly absent, so these are necessary
capacity bounds rather than a runnable representation or performance claim.

## Gates and interpretation

1. Every training reconstruction error decreases monotonically and the full
   centered numerical rank reconstructs training within F64 roundoff.
2. Every layer selects a validation-passing rank no greater than 111 without
   holdout access.
3. At the frozen selected ranks, holdout aggregate relative L2 is at most 1%
   and every nonempty route-coverage slice is at most 2%.
4. The F32 dictionary occupies at most 25% of its source layer bank and oracle
   synthesis uses at most 25% of selected source multiplications.
5. Gate 8 passes with the usual 20% free-memory, 8-GiB process,
   4-GiB-release, 512-MiB swap-growth, zero-throttling, and protected-service
   limits. Report zero accepted tokens, `A=0`, and no TPS.

If validation fails at rank 111, reject the fixed linear residual dictionary
without reading holdout. If holdout fails, reject it from untouched evidence.
Neither failure kills nonlinear direct compilation or PW-0045 generally. A
pass authorizes only a separately frozen router/input-conditioned coefficient
predictor at the selected ranks, compared with an input-only control. It does
not authorize recovery training, artifact promotion, a kernel, endpoint
integration, or any fidelity/performance claim beyond this three-layer English
capacity oracle.

## Result

Completed in `923.978 ms`. Every layer's rank-111 centered dictionary
reconstructs its 112 training residuals to `2.03e-15--2.22e-15` relative L2,
closing the SVD and implementation control. Generalization fails before
holdout:

| Layer | Rank 16 validation | Rank 64 validation | Rank 111 validation |
| ---: | ---: | ---: | ---: |
| 4 | `0.242393` | `0.067742` | `0.052437` |
| 24 | `0.352379` | `0.308693` | `0.270086` |
| 46 | `0.570177` | `0.443040` | `0.384575` |

All maximum-rank results miss the 1% aggregate gate by 5.24--38.46x. At layer
24, the 15 validation positions touching a training-unseen expert reach
`0.245325`, while the 41 all-seen positions are worse at `0.350668`; route
novelty alone does not explain the failure. Since no layer selected a rank,
the runner correctly left positions `168..223` sealed and emitted no holdout
metric.

The representation is physically tiny but causally insufficient: a rank-111
F32 mean/basis is 1,835,008 bytes, `0.028476%` of the source layer bank, and
oracle synthesis is `0.225830%` of selected source multiplications. Those
figures exclude coefficient prediction and executable wall time and do not
constitute a runtime path. Gate 8 passes at 79% minimum free memory,
172,902,528-byte maximum physical footprint, zero swap growth or new throttled
pages, and stable protected services. Raw evidence hashes to
`7a36bba9d8e6fc24cce802341ecfd56933aa05f7f4c07471004662ac414a5ffe`;
independent analysis hashes to
`e940d38d84a43332a408b41d6d6f005e9bf24bd3c5950dd61ecfc8d15bf6b1bc`.

## Decision

Reject a fixed linear routed-residual output dictionary on validation without
unsealing holdout or training a coefficient predictor. Perfect knowledge of
the coefficients cannot overcome the output-subspace error. This kills only
the fixed linear dictionary, not nonlinear/input-conditioned direct mixture
compilation or PW-0045 generally. Do not build an executor, recovery path, or
endpoint from this representation. No throughput-model constant or endpoint
TPS changes.
