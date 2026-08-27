# Work Frontier

## Outcome

On the existing 16 GiB Apple M1, deliver a fully local, full-capability,
fidelity-qualified endpoint above one sustained accepted TPS after prefill.
Companion hardware is inadmissible for this run. This is a named Prismwing-1
intermediate; the primary Prismwing-50 target in `TARGET.md` remains unchanged.

## Goal invariants

- Prismwing-1 retains the repository's full capability, fidelity, safety,
  local-inference, and reproducibility gates; only its named throughput tier is
  lower. Prefill timing remains separate from sustained decode timing.
- Target-faithful and modified modes remain distinct. K4 and other approximate
  weights remain L3 even when a bounded distribution slice passes.
- Companion capacity, compute, storage, procurement, or performance evidence
  cannot satisfy or reopen this run.
- PW-0325 opens only an analytical Prismwing-1 envelope: 3,925 selected K4
  identities and an oracle 8 GiB cache reach 1.252--1.373 storage-only TPS by
  category and 1.149 nearest-rank p10, with all compute and common work free.
  No bank construction, endpoint claim, or cache implementation is authorized.
- The envelope requires 75.45% of observed identity-window occurrences to use
  K4 under the legacy bonus-free transaction authority. PW-0326 repairs the
  full-match target-bonus/cache contract with complete deterministic gates but
  accepts zero model tokens. PW-0327 must capture one real corrected q8 target
  transaction per category before broader causal routes are regenerated.

## Prediction errors

- PW-0319's selector stopped assigning value after three K4 hits per routed
  row. Extending that order beyond its original `(3,5)` purpose was not a valid
  byte-minimizing plan; PW-0325 replaces it with a deterministic
  category-balanced byte objective.
- The standard full-match q8 commit omitted the target posterior bonus while
  its acceptance helper reported full width. PW-0326 resolves the contradiction
  as target-faithful `target_bonus_full_match_v1`; legacy evidence remains
  immutable and cannot receive an `A+1` projection or stale-route reuse.
