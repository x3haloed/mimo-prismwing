# PW-0113 — Exact fine-grained neuron canonicalization

- Status: completed
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: `e6cd914bee4b448d04864e2473e4e573698756d3`
- Artifact authority: PW-0106 layer-4 selected-expert artifact
  `fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21`;
  manifest
  `40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8`
- Hardware/runtime: Apple M1 shared 16 GiB host; internal SSD
- Exactness: L1 function-preserving permutation with expanded exact scale
  association; no weight, route, expert, activation, or arithmetic change
- Related records: PW-0013, PW-0045, PW-0106, PW-0108 through PW-0109,
  PW-0111 through PW-0112

## Changed premise

PW-0109 rejects permutations of the 16 existing 128-neuron source-FP8 scale
blocks. It does not test the full exact SwiGLU permutation symmetry. Every one
of an expert's 2,048 hidden neurons can move independently when the same
permutation is applied to gate/up output rows and down input columns. The
source format obscures that freedom by sharing each scale across a 128-by-128
weight block.

Test a deeper L1 representation that preserves every source FP8 code and
scale value but associates the exact scale vector with each neuron. This
duplicates scale metadata rather than requantizing weights: gate and up each
carry 32 F32 input-block scales per neuron, and down carries 32 F32 output-
block scales per neuron. A runtime could consume those records directly; it
need not reconstruct source matrices. The representation changes storage and
kernel addressing, not the expert function.

## Phase A — representation-cost falsification

Before matching or codec work, derive and verify the exact byte ledger:

- one neuron record contains 4,096 gate FP8 codes, 32 gate F32 scales, 4,096
  up codes, 32 up scales, 4,096 down-column codes, and 32 down F32 scales;
- each record is 12,672 bytes and each expert has 2,048 records;
- expanded exact records occupy 25,952,256 bytes per expert versus
  25,171,968 source bytes, an overhead of 780,288 bytes or approximately
  3.10%; and
- a 2,048-entry inverse permutation requires at most 4,096 bytes per expert
  using U16 indices.

Fail closed if source scale groups do not replicate exactly to those vectors,
if the expanded record cannot collapse back to every original weight and scale
byte, or if total representation overhead exceeds 10%. An overhead pass only
authorizes the selected-route codec experiment.

## Phase B — selected-route construction

Use the same eight authenticated PW-0106 layer-4 experts and deterministic
lowest-ID reference expert as PW-0109. For every expert:

1. Extract all 2,048 exact neuron records and prove complete, non-overlapping
   coverage of the six source tensors.
2. Build a deterministic similarity feature for each neuron from all three
   dequantized projections. For each of the 32 scale blocks in gate, up, and
   down, record the F64 sum, absolute sum, squared sum, and maximum absolute
   value in source order; normalize only with a frozen finite formula. The
   feature selects a permutation but is not compression evidence.
3. Compute reference-to-candidate squared feature distance in bounded row
   tiles and solve one minimum-cost 2,048-by-2,048 bijective assignment with
   deterministic index tie-breaking. Record the feature schema, cost hash,
   assignment, and exact assigned XOR popcount.
4. Invert the permutation, collapse replicated scale vectors only after
   proving all replicas in each original group are byte-identical, and
   reproduce all 48 original tensor hashes.

Do not add learned parameters, a common projection basis, sign flips, neuron
merging, scale averaging, requantization, or a generic graph/compiler layer.
Those are different causal mechanisms.

Compare equal-content controls with pinned zstd levels 1 and 19:

1. expanded neuron records in original expert/neuron order;
2. reference plus identity-order exact XOR residuals, neuron-major; and
3. reference plus aligned exact XOR residuals, neuron-major.

The expanded controls have 207,618,048 logical bytes for eight experts; report
compressed bytes both against that ledger and against the original
201,375,744 source bytes. Temporary streams stay outside Git and are removed
after hashes and codec evidence are recorded. Verify every decompression, XOR
reversal, inverse permutation, scale collapse, and source tensor hash.

## Measurement and gates

Run once under normative Gate 8. Record feature/assignment wall, peak
assignment-matrix bytes, extraction, stream construction, compression,
decompression, physical reads, RSS/physical footprint, memory pressure, swap,
throttling, buffer release, and resident-service health. Total temporary and
resident capacity must remain below 4 GiB; process peak/footprint must remain
below 8 GiB and release below 4 GiB.

For the aligned fast stream compute the optimistic physical bound

`58.033833 ms * compressed_bytes / 201,375,744 + decompression_ms`.

This assumes perfect proportional I/O and charges no metadata lookup, inline
scale expansion, kernel, layout, or command cost. It is a necessary bound, not
endpoint TPS.

- **Exactness:** every source code/scale byte is covered; assignments are
  bijective; expanded records and inverse permutations reproduce every source
  tensor hash; unknown or non-finite values fail closed.
- **Representation cost:** expanded records plus U16 permutations add no more
  than 10% to source bytes.
- **Canonicalization signal:** aligned residual compression beats both the
  expanded unmodified and identity-delta controls by at least 10% at the same
  codec level.
- **Physical continuation:** aligned fast compressed bytes are at least 25%
  below the original source bytes and the optimistic acquisition-plus-decode
  bound is at most 47.7 ms.
- **Safety:** at least 20% system memory remains free; swap grows by no more
  than 512 MiB; no throttled page or protected-service loss occurs; all
  temporary buffers release within the declared limits.

If Phase A fails, reject before assignment. If exactness or safety fails,
reject the implementation. If the signal or physical gate fails, reject
arbitrary-neuron permutation as the exact executable-byte mechanism on this
selected route; do not expand to all 256 experts or build a decoder. If both
pass, freeze a separate full-layer/holdout and inline Metal decoder contract
before promotion. A pass does not reopen source-FP8 speculation: PW-0110 and
PW-0112 must be recomputed using the measured transformed bytes and decode
cost.

## Result

Phase A passes exactly. Each 12,672-byte neuron record carries 12,288 source
FP8 codes and 384 bytes of replicated F32 scales. The expanded representation
is 25,952,256 bytes per expert, 780,288 bytes above the 25,171,968-byte source;
including a 4,096-byte U16 inverse permutation yields 3.1161% overhead, below
the frozen 10% stop. A real expert-9 preflight collapsed all 2,048 records and
reproduced every source tensor hash.

The clean implementation at
`a0ef5c4c6b95197d921cae52db5a474637f6550b` then completed all eight
authenticated experts. Every one of the seven 2,048-by-2,048 assignments is a
bijective deterministic mapping over the frozen 384-feature schema. Inverse
permutation and scale-replica collapse reproduce all 48 source tensor hashes.
Peak declared assignment-matrix storage is 100,663,296 bytes.

The expanded streams contain 207,618,048 logical bytes versus 201,375,744
original source bytes:

| Stream | zstd 1 bytes / ratio to source | zstd 19 bytes / ratio to source |
| --- | ---: | ---: |
| Expanded expert-major | 180,730,241 / 89.748% | 179,145,903 / 88.961% |
| Identity reference/XOR | 199,487,324 / 99.062% | 197,253,314 / 97.953% |
| Aligned reference/XOR | 200,691,292 / 99.660% | 194,002,317 / 96.338% |

At the fast setting, alignment is 11.045% larger than the expanded unmodified
control and 0.604% larger than identity-delta. It reduces original source bytes
by only 0.340%, not 25%. Measured decompression is 208.953 ms, so the optimistic
acquisition-plus-decode bound is 266.790 ms rather than 47.7 ms. High-level
analysis improves aligned residuals modestly relative to identity-delta but
still remains 8.29% larger than the unmodified high-level control and is not a
runtime path.

The immutable raw result at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0113/run-001.json` hashes to
`f6cb7d8510d2076b35db074a5c6a0511fff7c047effa0dcbb6fe7a146f7aea6a`.
The clean analyzer at `b90d05feb53048382712196d22932e6d9084cb30`
emitted
`/Users/chad/Models/mimo-prismwing/evidence/PW-0113/analysis-001.json`, hash
`5dfb78f1e32b206050e98754cbcfdfbbf4be2960715954e47465bb882aa51a21`.
The updated throughput model hashes to
`39f3de7d3ebdd774d7c827c19d7bc00aaef8c6acde498798113d0fba4f1669f3`.

Gate 8 passes all 13 boundaries with 78% minimum free memory,
666,550,272-byte peak RSS, 345,950,528-byte maximum physical footprint,
84,297,664-byte released footprint, zero swap growth, zero throttled pages,
and stable protected services. Temporary expanded and codec streams were
removed and never entered Git.

## Decision

Reject arbitrary-neuron permutation with exact per-neuron scale association as
the executable-byte mechanism on this selected real route. It exposes less
shared compressible structure than leaving each expert unmodified, while scale
association consumes most of the tiny gross byte reduction. Together with
PW-0109, both the source-scale-preserving 128-neuron symmetry and the full
2,048-neuron permutation symmetry fail their frozen mechanism and physical
gates.

Do not expand to all experts or build a decoder. Sign symmetry, a common basis,
learned residuals, and modified expert weights remain different mechanisms;
none is implied by this failure. Exact permutation canonicalization no longer
justifies work ahead of a named approximate representation or different
physical embodiment.
