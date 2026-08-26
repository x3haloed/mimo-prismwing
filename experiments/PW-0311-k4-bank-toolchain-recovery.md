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

At experiment opening, the causal addendum preserved the exact exporter,
bundle builder, QTIP settings, payload hashes, and source checkpoint, but not
the `mimo_lab` package, calibration atlas, three helper scripts, or pinned
external QTIP checkout those tools import. The verified M4 handoff below closes
that acquisition gap. Reproducing one bundled expert payload-by-payload remains
the cheapest fail-closed proof that arbitrary bank construction is possible.

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

## Authority-handoff preflight

The M4 worker payload is now a verified construction input. The transport ZIP
is `27,726,810,563` bytes and hashes to
`885de973e13ec61281d65fc9d86e9bea6349a2d0aa4de6c3c7bd67463da4cd6a`.
ZIP CRC extraction passed. The package omitted the expected top-level
`SHA256SUMS`, so the target host generated a fresh 10,926-file extracted-tree
ledger on the external disk; that ledger hashes to
`9be0f7f22b47412e4c5b9f2be6d522c0e2e61647e94270108eb8bd34287de131`.

The fresh ledger reconciles every declared payload needed by the construction
path:

- 27 atlas manifests retain canonical manifest-set SHA-256
  `e1b47f45dfa5975e8fff56f114779ba5242bcf6aadb55b5673d8f8a4d22edcf8`;
  all 8,883 capture files and 56,469,159,936 declared bytes match.
- The MRL-0147 anchor manifest retains SHA-256
  `14331fa6e6314d0b82b5a5b7085870e549db2dc9810d03d9251565ca5b281d9a`;
  all 329 capture files and 884,015,104 declared bytes match.
- The PW-0351 source-expert manifest retains SHA-256
  `c567a637e643476820ed07960385a9de84010ab48d9428441a08a84687b29ac8`;
  all 12 artifact/manifest files and 176,228,261 bytes match.
- All 21 PW-0352 projection manifests and 189 referenced payload files match,
  covering 89,098,324 bytes. The exporter, helper scripts, contracts, panel
  report, fixture, index, and TLUT also retain their recorded authorities.

The installed standalone Metal compiler reports Apple Metal `32023.883`.
PyTorch `2.13.0` sees MPS, the authenticated QTIP checkout is exactly commit
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`, official LDLQ and math modules
load, and regenerated TLUT SHA-256 is exactly
`21bab03171fb4ccaf2b4fb86f3b48efb2d7daa526f2b6dd3b01ceef9db95a9d8`.
The original absolute M4 paths resolve through a small symlink to the external
payload, so no 58-GB copy occupies the SSD.

The raw handoff receipt remains outside Git and hashes to
`4387273bf8127600d3b4e61742b78ab1dbd35782e08151c858f734bf0e4f2878`.
Promote the package only to authenticated construction input. Arbitrary-expert
reproduction, bank coverage, fidelity, capability, and endpoint TPS remain
unproven. No throughput-model constant changes.

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
