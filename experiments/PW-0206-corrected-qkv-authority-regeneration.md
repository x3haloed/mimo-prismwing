# PW-0206 — Corrected-QKV proposal authority regeneration

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-10
- Execution mode: target-faithful audit plus separately named modified controls
- Related records: PW-0102, PW-0103, PW-0150, PW-0186, PW-0187, PW-0203, PW-0205

## Hypothesis and mechanism

PW-0205 proves that earlier whole-model paths and their Python oracle assigned
the checkpoint's four tensor-parallel `[Q,K,V]` shards to the wrong semantic
rows. Because attention outputs feed later routers and logits, proposal tokens,
posterior tokens, route unions, `A`, and `U` derived through that mapping may
not transfer to the corrected runtime.

The hypothesis is deliberately diagnostic: regenerating the smallest decisive
traces will either preserve the old speculative conclusions or identify which
ones must be rerun. No speedup is claimed.

## Contract

Use the pinned checkpoint, tokenizer, prompts, seeds, and greedy verification.
Deinterleave global and sliding-window QKV exactly as the pinned SGLang loader
does. Keep source-faithful and SGLang-directed block-scaled arithmetic in
separate artifacts. Validate deinterleaving with a deterministic row-identity
fixture before a model walk.

Regenerate, in order:

1. the native-MTP first proposal and target posterior used by PW-0103;
2. the corrected DFlash-mask control used by PW-0150;
3. the Jacobi `q=8` convergence trace underlying PW-0186/PW-0187; and
4. only if materially changed, the smallest route-union slice needed to
   recompute `A/U` and the physical acquisition bound.

## Cheap falsifier and gates

Kill further regeneration if token IDs, accepted prefix, every per-layer route
set, and `A/U` reproduce the old trace exactly. Promote only the statement that
the prior conclusion survived corrected semantics.

If any of those authorities changes, mark the affected old conclusion
non-portable and regenerate its cheapest downstream bound. Do not rerun a full
endpoint until that bound can alter a branch decision. Record old/new token
IDs, `A`, `U`, routes, logical and physical bytes, cold/warm state, and safety.

## Decision

Unexecuted. This audit is the prerequisite for PW-0208 and for any claim that
PW-0203's wide-verifier economics survive PW-0205's semantic repair.
