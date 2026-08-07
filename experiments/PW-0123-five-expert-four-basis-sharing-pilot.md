# PW-0123 — Five-expert/four-basis sharing pilot

- Status: complete
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0119 analysis
  `166f56b0b56c82099520acd6696647d8bc350b52d5b33d8649d51a7971cf7a34`;
  PW-0122 analysis
  `5b5a21be9438e81e9b05a155ca365cd0dc4180be1b06a18a873362e88f60e0eb`
- Hardware/runtime: Apple M1 shared 16 GiB; NumPy Accelerate SVD, PyTorch
  2.13 MPS optimization and CPU source-FP8 evaluation; internal SSD
- Exactness: L4 fitted diagnostic; unchanged source controls
- Related records: PW-0045, PW-0115 through PW-0122; E5

## Question and causal mechanism

PW-0121/PW-0122 show that independent activation-weighted rank-768 factors
generalize better than global SVD at middle and late layers. They do not test
the storage-changing premise: can multiple experts use a smaller shared basis
family without surrendering those gains?

Test the smallest sample that forces actual sharing for the frozen `(r=768,
m=4)` identity-basis form. Four or fewer experts could receive one private
basis each and falsely pass. Use five layer-46 experts with strong coverage:

| Expert | Train | Validation | Pilot holdout | Total |
| ---: | ---: | ---: | ---: | ---: |
| 28 | 100 | 56 | 56 | 212 |
| 249 | 90 | 56 | 56 | 202 |
| 213 | 94 | 48 | 46 | 188 |
| 125 | 57 | 48 | 56 | 161 |
| 57 | 17 | 56 | 56 | 129 |

Every expert must participate in every split. Holdout remains unopened until
both independent and shared model selection are frozen.

## Representation and controls

For each gate, up, and canonical transposed down projection, first reproduce
the PW-0121 independent activation-weighted rank-768 procedure separately for
all five experts. These matched independent controls use each expert's own
balanced `L_e,R_e` factors and select only on its validation rows.

Then fit the shared candidate

`M_e = A_e @ (sum_j c[e,j] * B_j)`

with `A [5,2048,768]`, `B [4,768,4096]`, and unconstrained finite
`c [5,4]`. For each projection this is 20,447,252 F32 values (81,789,008
bytes) and a 327,156,032-byte parameter/gradient/Adam semantic live set. The
five-expert/four-basis topology prevents a private one-basis-per-expert
solution, though it does not prove behavior for the other 251 experts.

Initialize four `B_j` from the four most frequent experts' independently fitted
right factors, initialize their coefficients to the corresponding unit vectors,
and initialize `A_e` from their left factors. Initialize expert 57 by choosing
the single basis with lowest **train** normalized MSE and its independent left
factor; no validation or holdout value may choose initialization. Jointly
optimize all five `A_e`, four bases, and coefficients with an equal-expert mean
of per-expert normalized MSE so frequency does not erase the fifth expert.

## Frozen training and evaluation

Use seed `260123`, Adam `0.0005`, at most 150 joint steps per projection,
validation every five steps, and six-check patience. Select one shared
checkpoint per projection by equal-expert mean validation normalized MSE.
Train projections sequentially and release MPS state between them. Reconstruct
the complete expert equation with true dynamic-FP8 quantization and BF16
boundaries only after all shared projection checkpoints are frozen.

Report per expert and equal-expert aggregate train, validation, and untouched
holdout metrics for:

1. authoritative global-SVD rank 768;
2. matched independent activation-weighted rank 768; and
3. five-expert/four-basis activation-weighted sharing.

Hash but do not persist independent or shared factors. Record the full-shape
PW-0115 storage ledger and PW-0117 transaction multiplication ratio for
`(768,4)` separately from this five-expert training embodiment. No pilot byte
count may be presented as the full-bank ratio.

Add fixtures proving the shared equation, equal-expert loss, train-only fifth-
expert initialization, coefficient gradients, `experts > bases`, immutable
specification, and complete expert evaluation. Apply the PW-0121 Gate 8 and MPS
0.60 cap at source acquisition, every SVD, independent projection fit/release,
shared allocation/backward/checkpoint/release, complete evaluation, and final
service health.

## Gates

1. Every source oracle is bit-exact, every SVD authority matches its frozen
   control, every independent projection improves validation NMSE, and the
   already measured expert-28 independent result reproduces PW-0122 within
   `1e-6` relative L2.
2. Shared validation selection reads no holdout. Every shared projection's
   equal-expert validation NMSE must be at most `1.25x` the matched independent
   aggregate, and no expert may exceed `1.50x` its independent validation NMSE.
3. On complete expert outputs, the shared candidate's equal-expert validation
   and untouched-holdout relative L2 must each be at most `1.25x` the matched
   independent control. Every expert must remain at least 25% better than its
   own global-SVD holdout relative L2; no aggregate may hide an expert failure.
4. The full 256-expert `(768,4)` form must retain PW-0115's at-most-25% bank
   byte eligibility and PW-0117's at-most-50% transaction multiplication gate
   under an explicitly named executable dtype/layout hypothesis. These are
   algebra/size gates, not achieved runtime claims.
5. Gate 8 must pass; MPS current allocation returns to zero after every fit and
   final release. Report zero accepted tokens, `A=0`, no endpoint timing, and
   no TPS.

Passing authorizes a broader same-layer expert/corpus fit and executable-
representation derivation, not a full bank or runtime promotion. Failure kills
this four-basis sharing mechanism at rank 768 under the current objective; do
not add bases, change experts, or tune on holdout without a new contract.

## Result

The clean implementation at `6bf2d72bdee51435731d8a384960fa0a17bc4451`
completed in 180,934.883 ms. All five source oracles were bit-exact, all 15
independent projection controls improved validation, and expert 28 reproduced
PW-0122's complete metrics within `1e-6`. The shared coefficient gradients
were finite and nonzero, all three shared objectives improved from
initialization, and every MPS phase released current allocation to zero.

The forced-sharing result is sharply localized:

| Projection | Shared/independent equal-expert validation NMSE | Expert 57 ratio | Frozen gate |
| --- | ---: | ---: | --- |
| gate | 1.937x | 3.983x | fail |
| up | 2.320x | 5.053x | fail |
| down | 2.087x | 4.863x | fail |

Experts 28, 249, 213, and 125 each began with a private basis and retained
per-projection ratios no worse than 1.445x; expert 57 was the first identity
actually forced to use another expert's basis family and was the sole
per-expert projection failure. Equal-expert complete-output averages would
have hidden this: shared/independent relative L2 is only 1.112x on validation
and 1.138x on holdout, both within their aggregate gates.

The required per-expert holdout gate exposes the failure:

| Expert | Global-SVD holdout relative L2 | Independent fitted | Shared | Shared improvement vs SVD |
| ---: | ---: | ---: | ---: | ---: |
| 28 | 0.545815 | 0.288128 | 0.289267 | 47.00% |
| 249 | 0.698467 | 0.432653 | 0.450714 | 35.47% |
| 213 | 0.684109 | 0.372535 | 0.359454 | 47.46% |
| 125 | 0.599189 | 0.381405 | 0.364761 | 39.12% |
| 57 | 0.745367 | 0.524725 | 0.811428 | **-8.86%** |

The fifth expert is worse than both its independent control and global SVD.
The late validation tails continue descending slowly, but not at a scale that
supports a cheap duration retry: over steps 120--150, expert-57 up NMSE falls
only `0.098450 -> 0.096895` and down only `0.237832 -> 0.235617`, while their
1.5x gates require approximately `0.0288` and `0.0727`. This is diagnostic,
not a mathematical convergence bound; it makes an unchanged longer run a poor
next falsification.

The prospective full-bank physical algebra remains eligible under the named
FP8-factor/F32-scale/F16-coefficient hypothesis: 415,339,520 bytes per
projection versus 2,148,007,936 source bytes (`19.336%`), and transaction
multiplications remain `37.537%` of source. These are unachieved size/compute
hypotheses because the shared representation failed fidelity and was never
quantized or made executable.

Gate 8 passed with 69% minimum free memory, 1,746,518,016-byte peak RSS,
1,957,483,392-byte maximum physical footprint, 1,872,679,296-byte maximum
release-boundary footprint, zero swap growth/throttling, stable services, and
a 269,864,960-byte final footprint. The raw report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0123/run-001.json` hashes to
`e0f682e77d3f9ca79b762fae52534820af963b3a0478d5d4fa9944694ce5bbc2`.
Independent analysis at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0123/analysis-001/manifest.json`
hashes to
`4d4469184eda8717a12643a58b111d0a4fd6ac72585eb6aaabcfc6c187ab6438`.
There are zero accepted tokens and no TPS claim.

## Decision

Reject rank-768/four-basis sharing under this objective and corpus. Do not add
bases, swap experts, tune on the holdout, or rerun the same initialization for
more steps as if aggregate success erased expert 57.

This rejects the current sharing mechanism, not all learned executable-byte
reduction. The result also confirms that PW-0116's single English trace is now
the representation bottleneck: expert 57 has only 17 train placements versus
56 validation and 56 holdout rows. Before another shared fit, acquire a broader
frozen activation corpus with multilingual and modality traces, route-stratify
train/validation/holdout so at least five same-layer experts have substantive
coverage in every split, and keep the current five-expert result as the
unchanged control. If broader coverage still forces one expert below global
SVD, kill four-basis sharing more generally rather than expanding a full bank.

No throughput-model constant or endpoint TPS changes. The branch remains
unqualified for artifact, kernel, whole-prefix, or hosted evaluation.
