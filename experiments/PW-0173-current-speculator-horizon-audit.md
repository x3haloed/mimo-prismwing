# PW-0173 — current speculator horizon audit

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0170 exact `q=137`
  A770/storage envelope
- Execution mode: primary-source architecture audit and arithmetic comparison;
  no model execution or endpoint claim
- Related records: PW-0044, PW-0102, PW-0103, PW-0152, PW-0169 through
  PW-0172; E2 and E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0170 retains only a new base-aligned proposer spanning one `q=137` target
transaction. Its slow-storage A770 branch requires `A=77/137` for the owner's
34.3-TPS horizon and `A=113/137` for Prismwing 50; even the faster rejected BOM
requires `A=56/137` and `A=81/137`. PW-0152 already proves that the supplied
width-eight and published width-16 DFlash shapes cannot span those paths by
ordinary chaining.

Newer trained and tree-structured speculators may have changed that premise.
Before inventing or training a MiMo-specific proposer, audit the exact released
configurations and reported path horizons of EAGLE-3, P-EAGLE, AngelSpec/DFly,
and BASTION. Ask only whether any published configuration can be used directly
at PW-0170's minimum required accepted horizon.

## Exactness and scope

This is target-faithful L2 architecture triage. A compatible proposer would
still require exact target verification and correction. Cross-model acceptance
lengths are a research prior, not a bound on a hypothetical MiMo-trained
proposer. Reported speedups, GPU results, and means are not Prismwing TPS.

## Contract

1. Authenticate PW-0170's analysis manifest and immutable PDF captures of the
   EAGLE-3, P-EAGLE, AngelSpec, and BASTION papers by SHA-256. Record canonical
   paper URLs, versions, dates, and capture hashes outside Git.
2. Extract from primary tables or implementation descriptions, without
   extrapolation:
   - configured speculative depth or maximum accepted path;
   - total tree/token budget where applicable;
   - mean accepted length overall and the largest reported slice;
   - target model, decoding temperature, and workload scope.
3. Normalize only the structural comparison. Grant each published
   configuration its entire stated path, including a target bonus token when
   the source's convention could include one. Do not infer a longer path from
   total tree nodes or from speedup.
4. Compare every granted path with PW-0170's least demanding retained branch,
   `A=56`. If all are below 56, reject the audited configurations as direct
   Prismwing proposers. If any reaches 56, retain that exact configuration for
   a MiMo compatibility and calibration experiment.
5. Separately compare the strongest reported mean accepted length with
   `A={56,77,81,113}` as a diagnostic gap. This comparison may prioritize
   research but cannot reject an unbuilt scaled or MiMo-trained proposer.
6. Record training/deployment prerequisites that are material to direct reuse,
   especially target-specific weights, hidden-state access, block size, tree
   size, and unavailable MiMo checkpoints. Do not convert training hardware or
   engineering effort into runtime embodiment cost.
7. Apply Gate 8 to the local analyzer and source-capture phase. Record zero
   accepted Prismwing tokens, no endpoint TPS, and no purchase authority.

## Promotion and kill rule

If a released configuration grants a path of at least 56 tokens and can be
adapted to MiMo without contradicting its published prerequisites, promote a
separate MiMo compatibility/calibration experiment.

Otherwise, reject every audited released configuration as the missing PW-0170
proposer and preserve only a separately named, newly trained/scaled
MiMo-specific `q>=137` branch. Do not call that residual branch feasible or
impossible from this audit.

## Result

Pending source capture, fail-closed analyzer, and execution from a clean
implementation commit.
