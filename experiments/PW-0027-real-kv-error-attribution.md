# PW-0027 — Real learned K/V cache error attribution

- Status: complete
- Disposition: scope-decision
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `e869c8a`; generator dirty
- Checkpoint/processor/reference hashes: same locked MTP source and tensor
  contract as PW-0026
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2; source read-only on external
  platter
- Related records: PW-0026

## Hypothesis and mechanism

Uniform Turbo4's 19.43% learned sublayer error may be dominated by K-induced
softmax changes rather than V reconstruction. Separating source-K/Turbo4-V and
Turbo4-K/source-V on the identical learned fixture will identify which cache
side needs higher precision.

## Contract

Use PW-0026's exact tensors, hidden states, context, RoPE, value scale, sinks,
and output projection. Produce four paths: source K/source V, source K/Turbo4
V, Turbo4 K/source V, and Turbo4 K/Turbo4 V. Pass the attribution experiment
only if:

1. the source and uniform-Turbo4 output hashes and relative errors reproduce
   PW-0026 exactly;
2. each mixed path is deterministic and produces finite 8,192-wide attention
   and 4,096-wide projected outputs with SHA-256 identities;
3. report attention and projected-sublayer relative L2 for both mixed paths;
4. promote no fidelity default from one deterministic learned fixture. The
   lower-error side identifies only the next mixed-precision candidate.

No performance or endpoint TPS claim is in scope.

## Baseline and candidate

Baseline and uniform candidate are PW-0026. Mixed candidates change exactly
one cache side at a time while all other equations and learned values remain
fixed.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0027`.

## Isolated attribution

The exact PW-0026 source and uniform-Turbo4 hashes reproduce. Relative L2:

| K cache | V cache | Attention output | Projected sublayer |
| --- | --- | ---: | ---: |
| Source | Source | 0 | 0 |
| Source | Turbo4 | 0.136130 | 0.154924 |
| Turbo4 | Source | 0.126370 | 0.135811 |
| Turbo4 | Turbo4 | 0.185848 | 0.194277 |

K-only and V-only error are comparable. Keeping source V helps slightly more
than keeping source K on this fixture, contrary to a simple K-dominance
assumption.

## End-to-end result

Out of scope; no performance or endpoint TPS claim is permitted.

## Correctness result

All four conditions pass. Source and uniform hashes match PW-0026 exactly; all
mixed outputs are deterministic and finite.

Mixed output identities:

- source-K/Turbo4-V attention:
  `82a56bab00d0a6b14de81dbc662d8e878f8a826a3323cef5b93ef0df8e9f603e`
- source-K/Turbo4-V projected:
  `09d40882289bda97f289abd7cc0f8bf535998c1d7d2eeb5d2da601d51d942f7b`
- Turbo4-K/source-V attention:
  `52a952aa9373d9c47b6f0109dd20f3f4c209a4a4cdea595a345c7c0593c5f719`
- Turbo4-K/source-V projected:
  `37bd5681ea6d88124468ba85943fa5e0e2b3855a2329e33ca6bfe22eacd6b0dd`

Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0027`. The SHA-256 of its
`SHA256SUMS` manifest is
`b922ac3bad0187afd9d92ba608a14f3674e33a3c0579527bc97acd1cfd505909`.

## Decision

Reject a K-only precision upgrade as the default next branch: it would leave
15.49% projected error from V. Reject a V-only upgrade for the symmetric
reason: K alone leaves 13.58%.

Run a joint K/V precision sweep on this learned fixture. This is a
scope decision, not a fidelity claim; one deterministic MTP activation cannot
select the final cache format.
