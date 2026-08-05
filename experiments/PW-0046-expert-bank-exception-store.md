# PW-0046 — Expert bank as exception store

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: unassigned
- Commit and dirty state: proposal based on clean `4eedd12`; no execution
- Checkpoint/processor/reference hashes: inherits a successful, locked PW-0045
  artifact and the pinned MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Hardware, OS, compiler, storage, memory pressure: not measured
- Related records: PW-0002, PW-0039, PW-0045

## Hypothesis and mechanism

A resident mixture-compiled program can handle common routed activations while
the exact source expert bank serves as backing state for uncertain or tail
cases. The runtime predicts both the mixture residual and a calibrated risk;
unsafe positions fault to exact source-FP8 execution. Optional progressive
residual pages may refine an answer before a full exact fault.

This changes the expert bank from the primary instruction stream into an
exception store. It improves physical fitness only if exact faults and residual
bytes are uncommon on genuinely held-out inputs.

## Contract

This is an explicitly modified L3 hybrid mode named `mixture-exception`; exact
fallback does not make accepted approximate positions L2. The target-faithful
runtime remains available as control. Unknown schemas, missing expert pages,
non-finite confidence, and unsupported modalities fail closed to an error or
the exact path, never silently to the approximation.

Do not begin implementation unless PW-0045 passes its predeclared local audit.
The exception-store audit passes only if:

1. the risk gate is trained without the untouched evaluation split and uses
   signals available before deciding whether to fetch exact bytes;
2. evaluation reports false-safe and false-fault rates, calibration, exact
   fault rate, residual-page depth, bytes moved, latency, and output error by
   modality, language, context, rarity, and expert frequency;
3. held-out exact faults plus residual traffic consume at most 25% of the
   always-exact source expert bytes, while at least 99.9% of positions exceeding
   the PW-0045 local error limit fault to exact execution;
4. no required slice consumes more than 50% of always-exact expert bytes or
   exceeds PW-0045's local error limits after fallback;
5. adversarial confidence tests cover distribution shift, rare experts,
   modality conflicts, long context, NaNs, truncated pages, stale artifacts,
   and deliberate uncertainty-model corruption; and
6. complete endpoint promotion requires at least a 25% accepted-TPS gain with
   all `TARGET.md` fidelity and capability gates passing in this explicitly
   modified mode.

Kill the branch if safe thresholds require exact execution on more than half of
positions overall, if false-safe tail errors remain, or if storage/page-fault
latency returns the avoided compute gain.

## Baseline and candidate

Baseline is always-exact source-FP8 execution on identical routed activations.
Controls include always-compiled PW-0045 and a frequency-only fallback policy.
Candidate uses a predeclared calibrated risk gate and immutable exact backing
pages.

## Isolated attribution

Unexecuted. Cache hits and avoided storage reads must not be reported as avoided
executable-memory traffic unless the compute path actually omits those bytes.

## End-to-end result

Unexecuted. No endpoint or accepted-TPS claim exists.

## Correctness result

Unexecuted. Exact fallbacks are correctness evidence only for faulted
positions; every non-faulted position remains governed by L3 validation.

## Decision

Unexecuted and blocked on PW-0045. Do not build an uncertainty system around an
unproven resident approximation.
