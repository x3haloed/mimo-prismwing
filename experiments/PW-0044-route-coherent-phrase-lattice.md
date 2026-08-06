# PW-0044 — Route-coherent phrase-lattice verification

- Status: proposed; width prerequisite revised by PW-0110
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: unassigned
- Commit and dirty state: proposal based on clean `4eedd12`; no execution
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; future trace corpus must bind the
  complete model lock, draft lock, prompts, and generator configuration
- Hardware, OS, compiler, storage, memory pressure: not measured
- Related records: PW-0009, PW-0010, PW-0017, PW-0036, PW-0039

## Hypothesis and mechanism

Autoregressive tokens are logical outputs, not necessarily the best physical
execution unit. Build a tree or lattice of plausible future token sequences,
then choose a bounded subset that maximizes target probability mass covered per
predicted unique expert byte. Verify the selected nodes together with a tree
attention mask and execute their routed positions expert-major.

Unlike ordinary speculation that optimizes draft accuracy alone, selection
optimizes accepted tokens per expert-set union, `A/U`, while retaining exact
target correction. The intended effect is to turn serial future time into
matrix width without paying for an unconstrained expert union.

## Contract

Target-faithful L2 is the goal: draft and selection may be approximate, but
accepted tokens must follow the pinned target distribution. Begin in greedy
mode. Positive-temperature execution remains disabled until the correction
procedure has a written derivation, exhaustive tiny-distribution tests, and
statistical sampling tests.

PW-0110 supersedes the original implicit native-MTP/phrase-scale width prior
for an unchanged source-FP8 internal-SSD verifier. Before the first trace phase,
the candidate pool must support at least `q=137` target positions for the formal
50-TPS branch or `q=94` for the separately reported 34.3-TPS horizon. Those are
necessary impossible-perfect bounds at `A=q`, `U=1`; measured acceptance below
one or union above one raises the required width. A base-aligned proposer—not
the rejected supplied DFlash or native-MTP first proposal—is now prerequisite.

The first trace-only phase passes only if:

1. a slow complete target path records, for every candidate node, parent,
   token, draft probability, target probability or greedy authority, accepted
   status, and every layer's top-eight IDs and route weights;
2. the ordinary linear speculative baseline and route-coherent candidate use
   the same frozen prompts, candidate pool, node budget, target calls, and draft
   compute budget;
3. train/tuning and held-out prompt partitions are hash-bound and include text,
   multilingual, rare-domain, tool, and every available native modality slice;
4. reports include `q`, accepted prefix `A`, per-layer and aggregate `U`,
   `A/U`, logical and actual expert bytes, dense/attention work, rollback,
   candidate-tree width/depth, and cache state;
5. on held-out traces, routed bytes per accepted token fall by at least 30%
   versus linear speculation, mean accepted prefix is at least 90% of control,
   and no required slice increases routed bytes per accepted token by more than
   10%; and
6. a replay verifier proves that candidate selection never supplies target
   probabilities, logits, routes, or accepted outputs from fixture answers.

Only after the trace gate passes may an executable tree verifier be built. It
is promoted only after paired complete-endpoint runs improve accepted TPS by at
least 25% with no correctness regression. Microbatch width, aggregate branch
throughput, or proposed tokens are not accepted TPS.

Kill the branch if the fixed-pool trace comparison cannot reduce held-out
routed bytes per accepted token by 20%, if `U` grows as quickly as coverage, or
if exact correction removes the modeled gain.

## Baseline and candidate

Baseline is the best ordinary linear native-MTP or compatible draft verifier
available when the complete text endpoint exists. Candidate uses the identical
draft candidate pool but selects a route-coherent tree under the same node and
compute budget. A second diagnostic may enlarge the candidate pool, but it is
not comparable to the fixed-pool attribution result.

## Isolated attribution

Unexecuted. Required output is a content-addressed trace and replay tool that
can recompute selection, acceptance, `U`, and byte ledgers without rerunning the
model.

## End-to-end result

Unexecuted. No endpoint or TPS claim exists.

## Correctness result

Unexecuted. Exact greedy equality is the first gate; positive-temperature
target-distribution preservation requires separate proof and tests.

## Decision

Unexecuted. PW-0110 makes this a much wider and more demanding branch than the
original proposal: `q=16` and `q=32` are now rejected before training or
verifier construction on the source-FP8 internal-SSD premise. Do not
approximate it with PW-0017's correlated synthetic inputs or PW-0039's single
low-union fixture.
