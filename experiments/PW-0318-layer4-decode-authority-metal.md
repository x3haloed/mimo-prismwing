# PW-0318 — Layer-4 one-row decode-authority Metal transaction

- Status: planned
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0317

## Question

Does the layer-4 three-K4/five-source transaction reproduce an independently
constructed one-row decode oracle bit-for-bit on M1 Metal while remaining
below the unchanged one-percent distance gates against the frozen PW-0116
expert-major source capture?

## Hypothesis and mechanism

PW-0317 rejects because it uses a prefill-batch output as the bit-exact answer
key for one-row decode execution. Metal differs at exactly two BF16 values but
matches an additive, independently recomputed one-row source diagnostic at all
4,096 columns. The one-row candidate remains close to the batch source:
`0.00666279289` routed and `0.00197441695` layer-final relative L2.

This experiment makes those two authorities explicit rather than weakening or
relabeling PW-0317:

- PW-0116 expert-major output is the frozen source-fidelity comparator;
- independently recomputed one-row source-FP8 plus the qualified K4 outputs is
  the decode implementation answer key.

## Authorities

- checkpoint revision `63651580ca774f8504f676040460aed3e1244ac1` and verified
  installation receipt SHA-256
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
- PW-0116 corpus manifest SHA-256
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
- layer 4, position 1 route IDs `[96,64,232,31,88,245,223,151]` and captured
  route weights;
- PW-0315 summary SHA-256
  `07b3d3793a6750a030eb5b7e12a0add1b603d48758a85e6f45b44504e404d0e8`;
- qualified K4 experts `[64,232,31]` and exact-source experts
  `[96,88,245,223,151]`;
- PW-0317 bundle SHA-256
  `e87a0af2aba57f46b6a2f394d70e530533d04c18aa61650afbc8528a4b8bdc35`
  as a deterministic payload control, not a promoted artifact;
- PW-0317 rejection SHA-256
  `7e9ca112a27de9742bcad371655c6fb0a206a7e6802d2f94d9b8ef6056676080`.

## Protocol and gates

1. Authenticate and exactly reconstruct the complete PW-0116 source route and
   final residual.
2. For each exact-source fallback, replay every expert-major placement exactly
   as the batch-fidelity control. Separately execute position 1 as a one-row
   source operation and bind both output hashes to its fixture.
3. Reconstruct both mixed candidates:
   - batch comparator: three K4 outputs plus batch-captured source outputs;
   - decode oracle: the same K4 outputs plus independently computed one-row
     source outputs, accumulated in native route order and BF16-rounded.
4. Require both candidates' routed and layer-final relative L2 versus PW-0116
   source to remain strictly below `0.01`. The decode oracle becomes the named
   Metal answer key; the batch comparator remains present and immutable.
5. Build and Rust-verify one 16-KiB-aligned schema-2 `(3,5)` bundle. Require its
   binary hash to reproduce the PW-0317 payload control; manifest and fixture
   hashes are experiment-specific.
6. Execute the decode oracle on M1 Metal. Require all 4,096 F32 bits exact for
   the initial run, 20 warmups, and 100 timed samples.
7. Repeat construction, verification, and Metal execution from one clean
   pushed commit. Require identical bundle, manifest, fixture, readback output,
   semantic metrics, and Metal output bits. Timing and safety observations may
   vary.
8. Record cold/warm state, complete-call and GPU distributions, logical and
   physical bytes, batch size one, concurrency one, accepted tokens zero, and
   normative Gate 8 snapshots through buffer release and service health.

## Decision rule

- On full pass, promote the schema-2 one-row decode transaction as the
  partial-bank integration boundary.
- If either source-distance gate fails, reject the representation subset even
  if Metal matches its decode oracle.
- If Metal differs from the one-row oracle, reject the runtime path; the
  PW-0317 batch-shape explanation is then incomplete.
- Timing remains a component diagnostic and cannot establish endpoint TPS.

## Claims excluded

- equivalence of prefill-batch and decode-row intermediate bits;
- routes or identities outside this frozen row;
- full-bank coverage, complete decoder execution, or accepted-token TPS;
- hosted, multilingual, modality, long-context, or capability equivalence;
- Prismwing-2, 34.3 TPS, or Prismwing 50 completion.
