# PW-0113 — Exact fine-grained neuron canonicalization

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: pending
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
