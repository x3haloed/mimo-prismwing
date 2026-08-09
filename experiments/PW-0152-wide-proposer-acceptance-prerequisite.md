# PW-0152 — Wide-proposer acceptance prerequisite

- Status: proposed
- Disposition: unexecuted
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

Unexecuted.

## Decision

Unexecuted.
