# PW-0312 — K4 held-out expert construction

- Status: complete
- Disposition: rejected for arbitrary byte portability; semantic portability retained
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

## Reopened prediction error

Expected: expert 41 would reproduce bit for bit when constructed independently
with the authenticated inputs and seeds.

Observed: two fresh M1 processes deterministically produce the same gate
payload, but it differs from the M4 authority in 681 of 2,097,152 packed words.
The decoded candidates have relative L2 `0.002102904`; source-derived signs,
global scale, and validation input remain exact. The first failed report hashes
to `c600db1eaaef99e9a02713a7a9bb3d57de29afe551044f2cbbdf3ba551173d3c`;
the byte-identical repeat report hashes to
`adaf3ed22f501650ba8da3ec887bc3d701ddf1edbf4457948d53434ac002f96b`.

The smallest discriminator replayed expert 114's three projections before
constructing expert 188 in its original second panel slot. All 33 expert-188
files and 29,993,518 bytes then matched the M4 authority bit for bit, with zero
independent-decode relative L2 across gate, up, and down. The report hashes to
`8864837afa5f56d25500f08fbd278f2d49a0cc7a9317d497546c8d497cc19b7b`.

The initial interpretation was that artifact identity depends on the canonical
preceding construction sequence. Commit
`626bf0ba488b425f70e8e128b2076d468c5f0a31` added a prefix-replay diagnostic.

The diagnostic completed in `1017.933092` seconds and accepted zero tokens.
Gate 8 passes at `1,429,094,400`-byte peak RSS,
`1,674,908,672`-byte maximum physical footprint, 62% minimum free memory, zero
swap or throttle growth, and a `358,944,576`-byte release footprint. Promote
canonical-prefix held-out reconstruction; kill independent later-slot artifact
reconstruction. This changes no throughput constant or runtime default.

That inference is now superseded. Replaying all 15 gate/up/down projections for
experts 114, 188, 93, 199, and 248 before expert 41 still fails at expert-41
gate. More decisively, its `packed.u16le` is byte-identical to both independent
M1 attempts, so the replay changed no target state. The failed report hashes to
`39b753d159a7ea1c3f2f838b45b8cd616100daa41c8013a447595cec21d38271`;
the run took `2724.443504` seconds and passed Gate 8 with
`1,391,214,592`-byte peak RSS, 62% minimum free memory, zero swap/throttle
growth, and a `396,873,664`-byte release footprint.

The expert-188 prefix pass lacked an independent expert-188 control and cannot
attribute its exactness to prefix replay. The prediction error is reopened:
whether M1/M4 QTIP-MPS artifact identity is weight-dependent remains uncertain.
The cheapest discriminator is independent expert-188 construction from a fresh
process. If it also matches, kill the sequence mechanism and classify the
expert-41 difference as cross-device, weight-dependent numerical boundary
behavior. If it differs, sequence effects exist for expert 188 but are not a
sufficient general constructor contract.

The fresh no-prefix expert-188 control resolves that question. All 33 files
and 29,993,518 bytes match independently without replay, proving that prefix
work was irrelevant. Its report hashes to
`263b246cee429aec05c98acd28e12b0e6ff7e13a729f06a3b9608b8f509b2e4f`.
Complete wall is `506.037431` seconds; peak RSS is `1,443,725,312` bytes;
maximum physical footprint is `1,665,831,872` bytes; minimum free memory is
69%; swap/throttle growth is zero; and the release footprint is `347,950,848`
bytes.

Expert 199 adds a third, finer classification. Gate reproduces all 11 files and
9,991,465 bytes exactly. Up has the exact M4 candidate-array hash but 64 of
2,097,152 packed words differ. Independent decoding proves the two packed forms
produce every one of 8,388,608 F32 candidate elements identically. This is a
non-canonical trellis encoding difference, not a weight-semantic difference.
The fail-closed report hashes to
`019987400b0db0bec1f468e650a4fb013e51115fc711245ebcbd978bc46f565c`.
It passes Gate 8 with `1,385,594,880`-byte peak RSS,
`1,505,940,224`-byte maximum footprint, 68% minimum free memory, zero
swap/throttle growth, and a `385,863,552`-byte release footprint.

The prediction error is closed. The QTIP/MPS constructor is deterministic on
the target M1, but M1/M4 byte portability is weight- and projection-dependent:
some projections are payload-identical, some use different semantically exact
trellis aliases, and some produce slightly different candidates. Sequence
replay is not causal.

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

## Decision

Reject arbitrary M4-payload reconstruction on the M1. Do not use a payload hash
from one Apple GPU generation as the artifact identity for a reconstruction on
another. Preserve the authenticated M4-built PW-0425 bundle unchanged; its
target Metal execution evidence remains valid because runtime decoding, not
reconstruction, is the relevant path.

Open PW-0313 as an explicit `m1-native-k4-v1` L3 representation. It must create
complete target-native artifacts, reproduce them locally byte for byte,
classify M4 semantic differences without substituting hashes, and pass frozen
source, routed, and Gate 8 checks. No cross-layer, bank, fidelity, capability,
runtime, or TPS claim is promoted here, and no throughput constant changes.
