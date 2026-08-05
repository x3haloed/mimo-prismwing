# PW-0027 — Real learned K/V cache error attribution

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `90ceccf`; contract dirty
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

Pending.

## End-to-end result

Out of scope; no performance or endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
