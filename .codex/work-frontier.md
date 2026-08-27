# Work Frontier

## Outcome

On the existing 16 GiB Apple M1, deliver a fully local, full-capability,
fidelity-qualified endpoint above two sustained accepted TPS after prefill, or
produce decisive reproducible evidence that the milestone cannot be reached
within the authorized constraints. Companion hardware is inadmissible for this
run. The primary Prismwing-50 target in `TARGET.md` remains unchanged.

## Goal invariants

- Prismwing-2 retains the repository's full capability, fidelity, safety,
  local-inference, and reproducibility gates; only its named throughput tier is
  lower.
- Target-faithful and modified modes remain distinct. K4 and other approximate
  weights remain L3 even when a bounded distribution slice passes.
- Companion capacity, compute, storage, procurement, or performance evidence
  cannot satisfy or reopen this run.
- PW-0324 closes the current evidence-backed onboard portfolio below
  Prismwing-2. The best corrected measured complete path is `0.0459781517` TPS;
  q8 fails two TPS even at structural maximum acceptance; and real q64 at
  `A=3` needs another `17.594241525x` byte reduction after a 2,048-identity K4
  bank and free perfect 4 GiB cache. Canonical evidence:
  `97d4d20a4c709d42429973e867138495756ce9d52d417f98a7edd40b282ccff3`.
- PW-0324 is portfolio closure, not a theorem against unknown future
  algorithms. A future run may reopen only on a genuinely new onboard
  representation or proposer premise that first passes a cheap discriminating
  physical and fidelity gate, or on separately authorized companion hardware.

## Prediction errors
