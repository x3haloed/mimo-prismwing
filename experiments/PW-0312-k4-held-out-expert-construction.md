# PW-0312 — K4 held-out expert construction

- Status: in progress
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0311

## Question

Does PW-0311's bit-exact target-M1 reconstruction generalize to untouched
expert identities, or was expert 114 an accidental single-artifact success?

## Hypothesis and mechanism

The authenticated PW-0352 panel used the same frozen calibration, seeds, QTIP
implementation, and serialization contract for every selected expert while
changing the source weights. Exact reconstruction of experts 41 and 199 would
cross two untouched identities and two distinct PW-0351 mini-shards
(`selected-00` and `selected-04`) without changing the mechanism. That is a
cheap discriminator before constructing an unauthenticated new-layer artifact.

## Exactness and red-line check

The K4 candidates remain L3 modified weights. The construction claim is
bit-exact reproduction of those named candidates; it is not a claim that they
are source-exact. Thresholds, source identities, and PW-0352 payloads remain
unchanged.

## Protocol

1. Run `tools/reproduce_pw0311_k4_expert.py` from a clean pushed commit for
   expert 41, preserving failures and Gate 8 evidence.
2. Repeat independently for expert 199.
3. Require exact candidate-array and packed-state hashes, exact manifests and
   fixtures, exact bytes for every referenced payload, and zero independent
   decode relative L2 for all six projections.
4. Record construction wall, process I/O, RSS, physical footprint, release
   footprint, memory-free floor, swap/throttle growth, services, hardware,
   software, and commit. Construction time is diagnostic and accepts zero
   tokens.

## Decision rule

- If both held-out experts reproduce bit for bit and pass Gate 8, authorize one
  new-layer construction/quality experiment.
- If either differs, preserve the first mismatch and localize whether source
  shard, expert identity, numerical backend, or serialization is responsible.
  Do not create a new representation revision silently.

## Claims excluded

- arbitrary cross-layer construction;
- complete K4 bank;
- general fidelity or modalities;
- ordinary endpoint execution;
- accepted-token TPS or Prismwing completion.
