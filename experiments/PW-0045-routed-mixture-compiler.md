# PW-0045 — Routed-mixture executable compiler

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: unassigned
- Commit and dirty state: proposal based on clean `4eedd12`; no execution
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; future activation corpora and
  compiled artifacts must be independently hash-bound
- Hardware, OS, compiler, storage, memory pressure: not measured
- Related records: PW-0016, PW-0018, PW-0039; prospective E5

## Hypothesis and mechanism

The transformer observes the route-weighted sum of eight expert outputs, not
eight independently inspectable expert matrices. A smaller executable may
therefore compile the mapping

`(layer, hidden state, selected IDs, route weights) -> weighted MoE residual`

more efficiently than representing and evaluating every expert separately.
Candidate forms include shared nonlinear basis functions, a router-conditioned
coefficient program, coupled-neuron bases, or a small hypernetwork. Shared work
must be evaluated once per mixture rather than once per selected expert.

## Contract

This is an explicitly modified L3 or L4 mode named `mixture-compiled`; it never
replaces or relabels the target-faithful source-FP8 runtime. Training history,
seeds, source tensors, corpora, artifact hashes, and executable representation
must be locked.

The cheap audit passes only if:

1. activations come from real target routing at representative early, middle,
   and late layers, with common, rare, multilingual, long-context, and every
   available native modality separated;
2. train, validation, and untouched holdout partitions are fixed before model
   selection, and rare experts are not dropped or merged out of evaluation;
3. direct mixture compilation is compared at matched executable bytes and
   FLOPs against per-expert low-rank/factorized controls and source-FP8 truth;
4. the compiled representation occupies at most 25% of the source routed-bank
   bytes for the tested scope, reads at most 25% of source expert bytes per
   mixture, and reduces measured complete-MoE wall time by at least 2x;
5. on held-out layer-local fixtures, weighted residual relative L2 is at most
   1%, the following layer's top-eight agreement is at least 99.5%, and no
   required slice exceeds 2% residual relative L2; and
6. accumulated multi-layer tests measure hidden-state drift, later routing,
   sampled local-logit divergence, and rare-expert behavior before any recovery
   training or endpoint integration.

Passing the audit authorizes recovery training, not runtime promotion. Runtime
promotion still requires the full distributional and capability gates in
`TARGET.md` plus a repeatable full-path gain. Kill the stock-M1 branch if no
candidate meeting the local error gates fits within 25% of source bytes or if
its executable compute loses to source FP8.

## Baseline and candidate

Baselines are source-FP8 PW-0039 semantics and the strongest per-expert
factorization at the same byte/FLOP budget. Candidate predicts the complete
weighted mixture directly. Parameter-count or on-disk compression without
executable-byte and wall-time reductions is a failure.

## Isolated attribution

Unexecuted. The first artifact should cover a few fully verified layers rather
than train a whole-model replacement.

## End-to-end result

Unexecuted. No target fidelity, modality parity, or endpoint TPS claim exists.

## Correctness result

Unexecuted. Local regression metrics cannot promote this L3/L4 branch; they
only decide whether whole-model recovery work is warranted.

## Decision

Unexecuted. Begin only after representative real routed activations exist.
Preserve source experts as the independent teacher and target-faithful control.
