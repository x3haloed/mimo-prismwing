# PW-0317 — Layer-4 three-K4/five-source mixed Metal transaction

- Status: planned
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0316

## Question

Can the layer-generic mixed bundle execute a real layer-4 route with three
qualified target-native K4 experts and five exact source-FP8 fallbacks while
passing the unchanged row-local semantic, Metal parity, repeatability, and
Gate 8 contracts?

## Hypothesis and mechanism

PW-0316 rejects the four-K4 composition only after exact source replay: routed
relative L2 is `0.0109888419` against an exclusive `0.01` gate. A bounded
subset audit predicts that retaining K4 experts `[64,232,31]` and restoring
expert 96 to exact source reduces the same row to `0.0066623394` routed and
`0.0019744170` layer-final relative L2. This changes the active representation
subset, not the threshold.

The resulting `(3,5)` split is already an admitted executor shape. The new
work therefore tests the layer-generic schema-2 bundle path and the complete
target-host Metal transaction rather than adding a new kernel shape.

## Authorities

- checkpoint revision `63651580ca774f8504f676040460aed3e1244ac1` and verified
  installation receipt SHA-256
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
- PW-0116 corpus manifest SHA-256
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
- layer 4, position 1 route IDs `[96,64,232,31,88,245,223,151]` and captured
  route weights;
- PW-0315 canonical summary SHA-256
  `07b3d3793a6750a030eb5b7e12a0add1b603d48758a85e6f45b44504e404d0e8`;
- repeated, qualified PW-0315 run-001 artifacts for K4 experts `[64,232,31]`;
- receipt-bound source-FP8 checkpoint tensors for experts
  `[96,88,245,223,151]`;
- schema-2 loader and PW-0308 hash-bound Metal kernels.

## Protocol and gates

1. Authenticate every authority and reconstruct the complete PW-0116 source
   route and final residual bit exactly before artifact creation.
2. Export each source fallback into an isolated fixture. Replay every recorded
   expert-major placement for each identity, require complete bit equality,
   and select position 1 only afterward.
3. Independently substitute the three saved K4 expert outputs into the source
   schedule. Require selected-row routed and layer-final relative L2 each
   strictly below `0.01`.
4. Build one 16-KiB-aligned schema-2 bundle containing exactly three K4 and five
   source records. Bind layer, route, roles, shapes, offsets, lengths, hashes,
   source fixtures, build spec, and candidate fixture. Refuse overlap,
   substitution, or unknown semantics.
5. Verify the completed bundle with the Rust loader before Metal execution.
6. Execute the mixed route on Metal from the captured MoE input and route
   weights. Require every candidate routed F32 bit to equal the independent
   fixture on the initial run, 20 warmups, and 100 timed samples.
7. Repeat construction, loader verification, and Metal execution from the same
   clean pushed commit. Require identical bundle, manifest, fixture, and output
   hashes plus identical semantic metrics. Timing and host counters may vary.
8. Record cold/warm state, complete wall and GPU distributions, logical and
   physical bytes, batch size one, concurrency one, accepted tokens zero, and
   Gate 8 snapshots through buffer release and final service health.

## Decision rule

- On full pass, promote this schema-2 3/5 transaction as the partial-bank
  integration boundary and reuse it in the resumable bank/runtime path.
- If source replay or row semantics fail, reject the chosen representation
  subset without weakening thresholds.
- If bundle verification or Metal parity fails, reject the generalized runtime
  path even if its numerical error versus source is small.
- Timing is diagnostic only. This zero-token layer fixture cannot establish
  endpoint TPS.

## Claims excluded

- routes or identities outside this frozen row;
- full-bank coverage, a complete decoder, or accepted-token execution;
- hosted, multilingual, modality, long-context, or capability equivalence;
- Prismwing-2, 34.3 TPS, or Prismwing 50 completion.
