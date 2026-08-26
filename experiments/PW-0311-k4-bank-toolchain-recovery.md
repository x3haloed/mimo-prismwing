# PW-0311 — K4 full-bank toolchain recovery and reproduction

- Status: in progress
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
- Parent experiments: PW-0308 through PW-0310

## Question

Can the preserved K4 exporter and source checkpoint be turned into a complete,
reproducible arbitrary-expert construction toolchain, beginning with exact
reproduction of one authenticated K4 expert already present in the PW-0425
bundle?

This is the first construction gate toward a fully local, full-capability,
fidelity-qualified **Prismwing-2** endpoint. The measured K4 executor does not
support a 34.3- or 50-TPS claim, but approximately 2 accepted TPS is an
explicitly valuable first-tier delivery outcome.

## Revised outcome invariant

A fully proved local endpoint around 2 accepted TPS remains materially valuable
even when it cannot satisfy the 34.3- or 50-TPS milestones. Lower-tier delivery
must preserve the same capability, fidelity, safety, and reproducibility
standards; only its named throughput tier differs.

## Hypothesis and mechanism

The causal addendum preserves the exact exporter, bundle builder, QTIP settings,
payload hashes, and source checkpoint. It does not preserve the `mimo_lab`
package, calibration atlas, three helper scripts, or pinned external QTIP
checkout those tools import. Recovering or reconstructing those authorities and
reproducing one bundled expert payload-by-payload is the cheapest fail-closed
proof that arbitrary bank construction is possible.

## Protocol

1. Inventory every imported module, external repository revision, calibration
   input, source tensor, random seed, and numerical setting used by the
   preserved exporter. Separate recoverable content-addressed authority from
   genuinely missing semantics.
2. Prefer authenticated recovery from local archives or public pinned source.
   If reconstruction is necessary, implement each missing function against the
   preserved specs and add deterministic fixtures before using real weights.
3. Select one K4 expert from the authenticated PW-0425 bundle. Re-run all three
   projections from the verified source checkpoint and require exact hashes for
   packed states, signs, scale, row scale, corrections, and bundle payloads.
4. Independently decode the rebuilt payloads and require the existing
   projection and routed correctness gates unchanged.
5. Record construction wall time, peak RSS, disk usage, compiler/runtime
   versions, checkpoint identity, and Gate 8 safety. This is construction and
   correctness evidence, not accepted TPS.

## Decision rule

- If one real expert reproduces payload-for-payload, promote arbitrary expert
  construction and proceed to held-out routes, layer coverage, and bank-scale
  scheduling.
- If outputs are numerically valid but not bit-identical, localize the first
  divergent authority and keep the prediction error open; do not silently mint
  a new representation revision.
- If required semantic authority is genuinely unavailable and cannot be
  reconstructed from source/specification, record the exact missing facts and
  request only that material from the M4 research worker or project owner.

## Claims excluded

- complete bank coverage;
- general fidelity or modalities;
- ordinary endpoint execution;
- accepted-token TPS, 34.3 TPS, 50 TPS, or `TARGET.md` completion.

