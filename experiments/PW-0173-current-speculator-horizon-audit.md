# PW-0173 — current speculator horizon audit

- Status: completed
- Disposition: rejected for all audited released configurations; unbuilt
  MiMo-specific `q>=137` branch remains unproven
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0170 exact `q=137`
  A770/storage envelope
- Execution mode: primary-source architecture audit and arithmetic comparison;
  no model execution or endpoint claim
- Related records: PW-0044, PW-0102, PW-0103, PW-0152, PW-0169 through
  PW-0172; E2 and E7
- Implementation commit and dirty state:
  `cbaf46798a5b244d91c28a87d4962dfdafda2066`, clean

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

Completed from the clean implementation commit. The authoritative manifest
hashes to
`15ec2cfa3ea80a3914ce500f3cb8288a2149cc1948469aeecde04922f6f7a16d`.
It authenticates PW-0170 and immutable captures of all four primary papers.

Every audited released configuration is structurally too short. The analyzer
grants a target bonus even where source conventions differ:

| Configuration | Published depth | Favorable maximum path | Largest reported slice mean |
| --- | ---: | ---: | ---: |
| EAGLE-3 | 8 | 9 | 7.54 |
| P-EAGLE | 5 | 6 | 4.50 |
| AngelSpec DFly | 7 draft positions | 8 | 6.42 |
| BASTION | block 16 | 17 | 10.60 |

BASTION is the strongest structural control and still falls 39 tokens short
of PW-0170's least demanding `A=56` branch. Its best reported slice mean is
only a cross-model prior: PW-0170's `A={56,77,81,113}` requirements are
`{5.2830,7.2642,7.6415,10.6604}` times that mean. Those ratios do not bound a
hypothetical scaled or MiMo-trained proposer.

Reject EAGLE-3, P-EAGLE, the released AngelSpec block-eight forms, and BASTION
as direct PW-0170 proposers. The only residual speculation branch is a newly
trained or scaled MiMo-specific `q>=137` proposer with exact target correction;
its feasibility is not established. None of the audited projects publishes a
MiMo-compatible draft checkpoint, and all neural forms require target-specific
weights and target hidden-state or distribution access.

Gate 8 passes with 72% minimum free memory, 35,520,512-byte peak RSS,
17,728,896-byte maximum physical footprint, zero swap growth or throttling, an
explicit release boundary, and stable protected services. This source audit
records zero accepted Prismwing tokens, no endpoint TPS, no measured
throughput-model constant changes, and no purchase authority.
