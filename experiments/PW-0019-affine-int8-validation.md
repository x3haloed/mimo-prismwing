# PW-0019 — Predeclared affine-INT8 block validation

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `9103991`; clean before this contract
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; committed INT4 and INT8 block
  fixtures
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2
- Related records: PW-0016, PW-0018

## Hypothesis and mechanism

Affine INT8 preserves the complete source-FP8 routed-block output much better
than INT4 while remaining close enough in wall time to become the default MLX
research representation.

## Contract

Predeclared L3 component gate. Use PW-0016's actual layer-43 router, nine
heterogeneous selected experts, exact position schedule, normalized route
weights, and weighted sum. Compare affine group-128 INT4 and INT8 in three
paired process repetitions with order `4,8`, `8,4`, `4,8`. Each process uses
10 warm-ups and 30 measurements. Source load and installation quantization are
excluded; all selected buffers are warm.

INT8 passes only if all conditions hold:

1. every run matches the committed INT8 fixture and source router selection;
2. relative L2 versus source FP8 is at most 0.02 and cosine at least 0.9998;
3. mean of the three INT8 wall medians is no more than 1.20 times the mean of
   the paired INT4 wall medians;
4. all six runs complete without numerical or integrity failure.

Passing promotes INT8 only as the default quality-oriented research substrate.
It does not pass whole-layer, whole-model, distributional, capability, or
endpoint gates. Failure retains INT4 for performance work and returns INT8 to
exploratory status.

## Baseline and candidate

Baseline is affine INT4. Candidate is affine INT8. Commands use
`tools/mlx_moe_block_benchmark.py --bits BITS --fixture FIXTURE` and distinct
immutable external evidence paths under PW-0019.

## Isolated attribution

| Pair/order | INT4 median ms | INT8 median ms |
| --- | ---: | ---: |
| 1 (`4,8`) | 10.0533 | 11.1099 |
| 2 (`8,4`) | 10.1354 | 11.2376 |
| 3 (`4,8`) | 10.1577 | 11.0321 |

Mean INT4 median is 10.1155 ms; mean INT8 median is 11.1265 ms. The
candidate/control ratio is 1.09995, a 9.995% slowdown and comfortably inside
the predeclared 20% limit.

## End-to-end result

All six runs completed. No endpoint claim is in scope. Using the validated
INT8 mean as a repeated-layer diagnostic gives 15.29 routed-only TPS for this
fixture before every non-MoE cost.

## Correctness result

Every run verified its committed precision-specific fixture and matched source
router selection. INT8 relative L2 is 0.0102613, below the 0.02 threshold, and
cosine is 0.9999474, above 0.9998.

Evidence SHA-256:

- pair 1 INT4: `388c18ad9a4183ad4549bd2ad519d52f17ca425f87b6b25e4dd69d3122eda067`
- pair 1 INT8: `7a4781ab479c4b856fa7b2560d7695d64dfa27a8ba2452267cae5c6c7e91b107`
- pair 2 INT8: `525011d7c3c6486c4508238315207acc1f37f97448edc1d2566d23e6c15b8dbb`
- pair 2 INT4: `17fa32336819d1918b1be1d5858f000732e10769b8b5b000cd700e7992a669c8`
- pair 3 INT4: `cb60aabb0f0038cf1c8d647da51cd8e27391ab9d25e7f6fdb98f46beb0045ba9`
- pair 3 INT8: `375a070eaa43c08ac6f54532be6ca418d56438ffe57ebd10f3fd2cfcb6603c48`

External evidence root: `/Volumes/Elements/mimo-prismwing/evidence/PW-0019`.

## Decision

All four predeclared conditions pass. Promote affine INT8 as the default
quality-oriented MLX research representation and retain INT4 as the compact
performance branch. The promotion is conditional and component-scoped: INT8
remains L3 and must pass real-activation whole-layer, local-logit, hosted
distributional, capability, and endpoint gates before any target claim.
