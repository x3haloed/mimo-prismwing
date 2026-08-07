# PW-0124 — Coverage-rebalanced forced-sharing control

- Status: completed
- Disposition: negative
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation commit
  `9167f22e04a845bb76863919b12a8f732bf30fd7`; clean tree at execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0123 analysis
  `4d4469184eda8717a12643a58b111d0a4fd6ac72585eb6aaabcfc6c187ab6438`
- Hardware/runtime: Apple M1 shared 16 GiB; NumPy Accelerate SVD, PyTorch
  2.13 MPS optimization and CPU source-FP8 evaluation; internal SSD
- Exactness: L4 fitted diagnostic; unchanged source controls
- Related records: PW-0045, PW-0115 through PW-0123; E5

## Question and changed premise

PW-0123 rejects four-basis sharing because expert 57—the first identity forced
to share—has projection errors 3.98--5.05x its independent controls and a
holdout result worse than global SVD. That expert has only 17 training rows but
56 development-validation rows in PW-0116. Before paying for another
approximately 22-minute source-weight corpus walk, test the cheapest coverage
explanation using only the already-unsealed development prefix.

Keep global positions `168..223` as the identical untouched pilot holdout.
Within positions `0..167`, sort each expert's occurrences by global position
and assign occurrence indices divisible by five to validation; assign the other
four of each five to training. No original holdout position may move or be read
during selection. Frozen counts become:

| Expert | Rebalanced train | Rebalanced validation | Unchanged holdout |
| ---: | ---: | ---: | ---: |
| 28 | 124 | 32 | 56 |
| 249 | 116 | 30 | 56 |
| 213 | 113 | 29 | 46 |
| 125 | 84 | 21 | 56 |
| 57 | 58 | 15 | 56 |

This is a causal coverage control, not a new representative corpus. A pass
would show that PW-0123's failure was development-split scarcity and authorize
broader multilingual/modality acquisition. It could not promote sharing from
the same correlated English trace.

## Frozen reuse and gates

Parameterize the PW-0123 executor with an immutable partition specification,
evidence class, parent hash, seed, and decision labels. Reuse the exact five
experts, four bases, rank 768, source targets, balanced independent controls,
private-four/train-selected-fifth initialization, equal-expert objective,
Adam settings, 150-step maximum, factor non-persistence, physical ledger, and
Gate 8 behavior. Use seed `260124`. Add fixtures proving the exact rebalanced
counts and that changing positions `168..223` cannot affect train/validation
indices or fifth-basis initialization.

The frozen acceptance gates remain deliberately unchanged:

1. all source oracles are bit-exact and every independent projection improves
   on its new validation split;
2. each shared projection is at most `1.25x` the independent equal-expert
   validation NMSE and at most `1.50x` for every individual expert;
3. complete shared validation and unchanged holdout equal-expert relative L2
   are each at most `1.25x` their matched independent controls;
4. every expert's unchanged shared holdout remains at least 25% better than its
   global-SVD holdout, including expert 57 at most
   `0.5590250442301753`; and
5. full-bank byte/compute algebra and Gate 8 pass, with zero MPS current
   allocation after every fit and release. Report zero accepted tokens, `A=0`,
   no endpoint timing, and no TPS.

Passing kills only the scarcity explanation and authorizes a broader corpus
before another representation decision. Failure rejects four-basis sharing
more strongly on this trace: do not acquire a broad corpus merely to rescue
this exact parameterization unless another independent use justifies the
corpus. Neither outcome authorizes artifacts, kernels, or runtime changes.

## Result

Completed in `183,010.327 ms`. The frozen rebalancing produced exactly the
predeclared counts and all independent projection controls improved. Source
oracles remained bit-exact. Expert 57's complete holdout relative L2 improved
from global SVD's `0.745367` to `0.515999`, so the composed 25% improvement
gate now passes.

The representation gates nevertheless fail more strongly. Shared-to-
independent aggregate projection NMSE ratios are `2.291x` gate, `3.158x` up,
and `1.856x` down, all above `1.25x`. Expert 57's corresponding per-expert
ratios are `5.241x`, `6.817x`, and `4.329x`, all above `1.50x`; expert 28 also
misses the up-projection tail gate at `1.568x`. Complete validation and
holdout means pass at `0.953x` and `1.136x`, again demonstrating that complete
aggregate behavior conceals projection-level tail failure.

The prospective physical algebra remains eligible at `19.336%` of source
projection bytes and `37.537%` of source multiplications, but is not an
achieved artifact or runtime result. Gate 8 passed with 70% minimum free
memory, 1,958,122,368-byte maximum physical footprint, zero swap growth or new
throttled pages, stable protected services, and zero final MPS allocation.
Raw evidence hashes to
`086cd06b66aa79117e44f3b17e3f1b18b751640d1696e3ce6f3045a769586077`;
independent analysis hashes to
`a6c98d0469e2e788e5c54833975277ebcffa822a3d0b426a8bb39dbf3606d32a`.

## Decision

Reject development-split coverage scarcity as the explanation for PW-0123's
four-basis failure. Do not acquire a broader corpus solely to rescue this exact
identity-basis/rank-768 parameterization. Preserve the positive independent-fit
and complete-holdout observations as separate facts; neither overrides the
failed frozen projection gates or authorizes an artifact, kernel, endpoint
claim, or TPS change.
