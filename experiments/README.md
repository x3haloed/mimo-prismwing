# Experiment ledger

This directory is the append-only history of Prismwing experiments. The staged
research queue lives in `docs/EXPERIMENTS.md`; a proposed experiment becomes
evidence only when it has a record here and a raw-evidence manifest or hash.

The architecture-level candidate portfolio and its shared compression contract
live in [`docs/EMBODIMENT_JUMPS.md`](../docs/EMBODIMENT_JUMPS.md). The ledger
currently runs through proposed record PW-0214. The next unreserved experiment
ID is PW-0215.

Use sequential IDs such as `PW-0001-checkpoint-census.md`. Never renumber,
delete, or overwrite a negative result. If stronger evidence reverses a
decision, create a new record and cross-link both records.

## Record template

```markdown
# PW-NNNN — Short title

- Status: proposed | running | complete
- Disposition: production | conditional | rejected | correctness-repair |
  reversed | scope-decision | unexecuted
- Date:
- Owner:
- Commit and dirty state:
- Checkpoint/processor/reference hashes:
- Hardware, OS, compiler, storage, memory pressure:
- Related records:

## Hypothesis and mechanism

What should change, why, and where it should appear in the full-path budget.

## Contract

Target-faithful or modified; exactness class; applicable red lines; success and
kill thresholds fixed before measurement.

## Baseline and candidate

Exact commands, configuration, fixtures, prompts, cache state, seeds, sampling,
and protocol deviations.

## Isolated attribution

Local counters and measurements with units and uncertainty.

## End-to-end result

Interleaved control/candidate results for relevant short, medium, long,
multimodal, and holdout workloads. Include accepted TPS and complete wall time.

## Correctness result

Exact byte/output checks or distributional/capability gates appropriate to the
claim. Include raw-evidence location and content hash.

## Decision

Disposition, confidence, limitations, reusable lesson, and next cheapest
falsification. Explain any reversal of prior evidence.
```
