# PW-0152 — Wide-proposer acceptance prerequisite

- Status: completed
- Disposition: rejected
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0151 analysis
  `d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1`;
  PW-0150 analysis
  `72051c021ae1d93989508b0423ab1b0811072c24799b8e986d4543b4a513f04e`;
  DFlash arXiv v2 PDF
  `ffa514e6ce180eb1f7a39c49372f3b8170b99f8bc142d4a4daa0f087bf2ceb91`
- Hardware: analytical prerequisite only; no accelerator, storage, or memory
  purchase and no endpoint execution
- Implementation commit and dirty state:
  `aa9e6389a5ceb42d31bc293148b8fc99ebcf42f4`, clean
- Related records: PW-0044, PW-0102, PW-0112, PW-0150, PW-0151

## Question and risk frontier

PW-0151 leaves one narrow direct-FP32 companion envelope: on its conservative
four-by-2.5-GB/s storage grant, a `q=137` target transaction needs at least
`A=86` accepted positions for the useful 34.3-TPS horizon and `A=125` for
Prismwing 50. Determine whether the published DFlash block mechanism, the
supplied width-eight proposer, or ordinary chaining can causally provide that
single-transaction acceptance before spending compute on proposer training.

The risk frontier is proposer shape, not target arithmetic. A verification
transaction amortizes its routed expert union only across positions evaluated
together. Draft blocks separated by a target verification cannot be added and
reported as one `q=137` transaction.

## Exactness and red-line check

This is target-faithful greedy L2 analysis. Draft candidates may be approximate,
but the pinned base target remains the only acceptance authority. The experiment
does not modify target weights, routes, tokenizer, modalities, or acceptance
thresholds. It reports no proposed token as accepted and no analytical ceiling
as endpoint TPS.

## Contract

1. Authenticate PW-0151's clean `run-003` report, PW-0150's exported-mask
   control, and the camera-ready DFlash arXiv v2 PDF by SHA-256. Fail closed
   unless the reports retain the supplied proposer's `A=1/8`, `q=137`,
   `A>=86` for 34.3 TPS, and `A>=125` for 50 TPS in the named four-lane
   envelope.
2. Bind DFlash's published semantics: one cycle proposes `gamma` tokens,
   `tau` includes the target bonus token, and `tau <= gamma + 1`; training uses
   a clean target-produced bonus token as the first block position and predicts
   only `block_size - 1` following positions. The main experiments use block
   size 16; the shipped MiMo proposer uses width eight.
3. Prove structural maxima before empirical comparisons. A width-`b` linear
   block has `A<=b`. Chaining conventional blocks requires the intervening
   target bonus/anchor and therefore creates separate target transactions. A
   `q`-node greedy tree or lattice can accept no more positions than its
   deepest root-to-leaf path.
4. For `q=137`, report the minimum tree depth and maximum off-path node budget
   at both targets. Report the block count and target transaction count needed
   to span 137 positions with widths eight and 16.
5. Use the constant independent conditional-acceptance model only as a named
   diagnostic: solve `sum(p**i, i=0..q-1) = A`. Do not treat its inferred `p`
   as a measured MiMo acceptance rate or as an impossibility proof.
6. Compare the prerequisite with the strongest published DFlash Table 6
   acceptance (`tau=6.33`, eight draft layers, block size 16), the published
   five-layer rows (`5.99`, `4.94`, `3.37`), and the supplied MiMo result
   `A=1/8`. Cross-model published values are scale evidence, not a bound on a
   newly trained MiMo proposer.
7. Reject conventional width-eight/16 DFlash and ordinary chained blocks if
   their structural maximum cannot reach the required `A` in one target
   transaction. Retain a newly trained `q>=137` block or depth-at-least-125
   base-aligned proposer only as a distinct unproven architecture, not as a
   continuation of the rejected implementation.
8. Apply Gate 8 safety stops while authenticating and writing evidence. Report
   zero accepted tokens, zero endpoint TPS, and every analytical limitation.

## Promotion and kill rule

This record cannot promote a runtime. If the published/supplied embodiments
fail their structural maxima, kill proposer training that preserves their
width and target-boundary shape. A long-block or long-depth replacement may
proceed only through a new experiment with a cheap base-aligned calibration
gate and a complete draft-compute, target-union, memory, and training-data
ledger.

## Result

The authoritative `analysis-003` manifest hashes to
`68783813c30d08aabb6c23971d65b2579655314819ea8d6e1aef8b19328bc686`.
It authenticates PW-0151, PW-0150, and the camera-ready DFlash v2 PDF, and
reproduces PW-0151's `A>=86/137` requirement for 34.3 TPS and `A>=125/137`
requirement for 50 TPS.

The published/supplied block shapes fail structurally before draft quality is
considered. A width-eight block can accept at most eight positions in one
target transaction and needs 18 target transactions to span 137 positions. A
width-16 block can accept at most 16 and needs nine. Conventional chaining
does not compose these into one amortized transaction: the next block requires
the clean bonus/anchor emitted by the preceding target verification.

A tree does not create a wide escape hatch at the fixed node budget. At
`q=137`, the 34.3-TPS branch requires a root-to-leaf path of at least 86 nodes
and leaves at most 51 off-path nodes. Prismwing 50 requires depth at least 125,
puts 91.24% of all candidate nodes on the accepted path, and leaves only 12
nodes for every alternative branch.

The explicitly diagnostic constant-independent-match model requires
conditional match probability `0.9925414` for expected `A=86` and `0.9986313`
for expected `A=125`. The strongest published DFlash Table 6 row is
`tau=6.33/16`; its corresponding diagnostic probability is `0.8548765` and
would asymptote to only `6.8907` expected positions at `q=137`. Reaching the
50-TPS prerequisite would require about `106.03x` less conditional mismatch.
These paper results use other targets and are scale evidence, not an
impossibility bound on a newly trained MiMo proposer.

Gate 8 passes across five snapshots with 66% minimum free memory,
30,867,456-byte peak RSS, 19,318,080-byte maximum physical footprint, zero
swap growth, zero new throttled pages, and stable protected services. The
first invocation rejected an incorrect commit identity, the second exposed a
mistyped PW-0150 evidence class, and the third exposed an obsolete safety API
call; all failed before manifest publication and their implementation defects
were corrected in subsequent pushed commits. No endpoint ran, accepted-token
count is zero, and no throughput-model constant changes.

## Decision

Reject training or runtime work that preserves the supplied width-eight or
published width-16 DFlash transaction shape, and reject ordinary chaining as a
way to satisfy PW-0151's single-transaction requirement. This kills that
specific PW-0044 prerequisite embodiment.

Do not generalize the result to every proposer. Retain only a separately named,
base-aligned `q>=137` block or depth-at-least-125 proposer as logically open.
It is a new architecture, not a continuation of conventional DFlash, and must
first pass a cheap calibration with a complete draft-compute, target-union,
memory, and training-data ledger. The next embodiment question is whether
resident expert DRAM removes enough of the extreme acceptance prerequisite to
fit the owned host and complete `$500` BOM.
