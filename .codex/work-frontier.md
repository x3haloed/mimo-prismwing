# Work Frontier

## Outcome

Advance the primary 16 GiB Apple M1 target in `TARGET.md`; treat stronger-worker
results as candidate mechanisms until they pass target-machine correctness,
safety, and full-path performance gates.

## Goal invariants

None beyond the authoritative contracts in `TARGET.md`, `RED_LINES.md`, and
`docs/WORKFLOW.md`.

## Prediction errors

None unresolved. The stronger worker's large-cache result is target-inadmissible;
its RAM-neutral one-row kernels remain an M1 performance hypothesis under
repeatability testing.
