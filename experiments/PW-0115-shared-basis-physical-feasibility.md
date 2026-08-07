# PW-0115 — Shared-basis physical feasibility envelope

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0114 analysis
  `14866caa426287a61d9ed91a441ff6937465da4e542114ba29b773726b332fa6`
- Hardware/runtime: analytical bound for the 16 GiB M1; no model execution
- Related records: PW-0045, PW-0108, PW-0113, PW-0114; prospective E5

## Question and causal mechanism

Exact source-FP8 acquisition remains too slow, exact expert permutation does
not expose useful residual structure, and PW-0114 conditionally retains
repair-free L3 arithmetic. Before spending another full-model walk or training
a replacement, determine which shared-basis representations can meet PW-0045's
physical gates by construction on MiMo's exact expert shapes.

The pinned routed expert has three equal-byte projections:

- gate: `2,048 x 4,096` FP8 plus a `16 x 32` F32 scale grid;
- up: `2,048 x 4,096` FP8 plus a `16 x 32` F32 scale grid; and
- down: `4,096 x 2,048` FP8 plus a `32 x 16` F32 scale grid.

There are 256 routed experts in each of 47 layers and eight experts per token.
The source record is 25,171,968 bytes per expert, 201,375,744 bytes per selected
eight-expert layer, and 302,869,118,976 bytes across the routed bank.

Published MoBE factorizes gate/up while retaining down unchanged. Since all
three MiMo projections have equal source bytes, even free gate/up would leave a
one-third bank floor. Test that applicability bound separately from a deeper
whole-mixture form that factorizes all three projections.

For the deeper form, model each projection using an expert-specific factor and
`m` layer-shared basis matrices at rank `r`, oriented so the eight experts'
latent contributions can be reduced before each shared basis is evaluated.
This is an optimistic physical envelope, not a learned artifact or fidelity
result.

## Exactness and accounting contract

This is a scope-decision analysis for a prospective **L3/L4**
`mixture-compiled` representation. It does not change weights or emit model
output. Source-FP8 remains the target-faithful control.

Fail closed unless the analyzer authenticates the PW-0106/PW-0113 exact source
shape and byte constants. Enumerate `r in {16, 32, 64, 128, 256, 512, 768,
1024}` and `m in {1, 2, 4, 8, 16, 32}`. Use an explicitly optimistic storage
model:

- one byte per factor or basis value, with no scale metadata;
- two bytes per expert/basis coefficient;
- all shared bases resident across 47 layers;
- only the eight selected expert-specific factors streamed per mixture; and
- shared basis work evaluated once per routed mixture, not once per expert.

For hidden width `d=4096`, expert width `p=2048`, experts `n=256`, and selected
experts `k=8`, record:

```text
source values/layer = 3 n p d
factor values/layer = 3 n p r
basis values/layer  = 3 m r d
coefficient bytes/layer = 3 n m * 2
bank ratio = (factor bytes + basis bytes + coefficient bytes) / source values
selected streamed-factor ratio = (3 k p r) / (3 k p d)
optimistic compute ratio = (3 k p r + 3 m d r) / (3 k p d)
resident basis bytes = 47 * 3 m d r
```

The source denominator deliberately omits its scale overhead while candidate
metadata is also omitted except coefficients. This favors the candidate only
slightly and remains a necessary, not sufficient, bound. Report both the exact
source-byte ledger and the idealized value-domain ratios without conflating
them.

## Gates

A configuration is physically eligible for an activation-weighted audit only
if all of these optimistic necessary conditions hold:

1. full routed-bank ratio is at most 25%;
2. selected expert-specific streamed-factor ratio is at most 25%;
3. optimistic complete-mixture compute ratio is at most 50%, preserving a
   mathematical path to PW-0045's 2x wall gate before overhead; and
4. all 47 layers' shared bases occupy at most 4 GiB.

The analyzer must identify the Pareto frontier over rank, basis count, bank
ratio, compute ratio, and resident basis bytes. Preselect at most three
structurally distinct configurations for a later fidelity audit; selection
must favor rank and basis diversity, not an observed quality result.

Reject direct adoption of any form with a hard byte floor above 25%, including
published MoBE if its unchanged-down floor is confirmed. Do not describe that
as rejecting MoBE's reported quality results or learned shared bases generally.

Passing this envelope authorizes an activation-corpus contract and
implementation estimate only. It does not authorize training, a full-bank
artifact, endpoint integration, or a performance/quality claim. If no
all-projection configuration survives, kill the current shared-basis parameter
family before collecting activations and return to a different representation
or companion-hardware premise.

## Result

Unexecuted.

## Decision

Unexecuted.
