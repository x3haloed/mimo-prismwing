# PW-0123 — Five-expert/four-basis sharing pilot

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

Unexecuted.

## Decision

Unexecuted.
