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
- PW-0324 closes the current evidence-backed onboard portfolio below two TPS,
  not one TPS. Its q8 storage-only ceiling contains a narrow one-TPS opening,
  but only under an unbuilt L3 K4 bank, free perfect cache, and omitted compute.
  PW-0325 must first identify a bank/cache envelope with enough category and
  compute headroom before any construction tranche is authorized.

## Prediction errors
