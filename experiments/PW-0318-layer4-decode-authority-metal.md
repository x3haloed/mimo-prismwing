# PW-0318 — Layer-4 one-row decode-authority Metal transaction

- Status: completed
- Disposition: promoted as partial-bank decode integration boundary
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

## Result

Two fresh constructions from clean pushed implementation commit
`7024270173ca8bc9659093118ad99f2ea250f996` pass. Both reproduce these
artifacts byte-for-byte:

- bundle: 164,724,736 bytes, SHA-256
  `e87a0af2aba57f46b6a2f394d70e530533d04c18aa61650afbc8528a4b8bdc35`;
- manifest: SHA-256
  `ca2cd8005c3c8f712fabd0b2fc88183d740bd6613efa065cdd4b25738c4924c3`;
- decode fixture: SHA-256
  `0189a8c15299410537cd43f934c4aefbda1c160e7c9f6920790cabfd812a6706`;
- build specification: SHA-256
  `ecd5717062a0e430cad05ca5c309755edcb84e53a3a9ef2796691827916bac41`;
- Rust loader report: SHA-256
  `06a56ba754f9b1b10696930e05da887186abbffcce12dc702801a7dc417bb171`.

Both numerical authorities pass unchanged. The expert-major batch candidate is
`0.00666233943` routed and `0.00197441695` layer-final relative L2 versus
PW-0116. The independently constructed decode candidate is
`0.00666279289` routed and `0.00197441695` layer-final relative L2. Both are
strictly below `0.01`.

On Apple M1, each initial execution, 20 warmups, and 100 timed samples matches
all 4,096 one-row decode-oracle F32 bits. Run 1 complete-call p90 is
`16.986541` ms and GPU p90 is `10.265500` ms; run 2 complete-call p90 is
`17.098042` ms and GPU p90 is `10.344333` ms. These are one routed-layer
component diagnostics, not distinct-layer or endpoint throughput.

The two Metal reports hash to
`db1becf58c050f50dc88ceb29d20c8ae08e2423f2a9fb61460e8c6321625fa4b`
and
`92a9770896120e19b23cd9ece58ae9804640eedcdf5ee1ec87d5e52cc47c8806`.
The superseding canonical summary, which includes both builder and Metal
safety phases, is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0318/summary-002.json` and
hashes to
`a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f`.
The earlier summary is retained but superseded because it aggregated only the
lower-memory Metal phase.

Gate 8 passes with 68% minimum system-free memory, 830,603,264-byte maximum
peak RSS, 249,269,248-byte maximum physical footprint,
241,257,216-byte maximum release-boundary footprint, zero swap growth or new
throttled pages, and stable protected services. The experiment accepts zero
tokens and makes no performance claim.

## Decision

Promote the schema-2 one-row decode transaction as the partial-bank integration
boundary. Preserve PW-0116 expert-major outputs as source-fidelity comparators
and use named one-row fixtures for decode implementation parity. Reuse the
verified receipt-bound source records, aligned bundle layout, Rust readback,
Metal transaction, and Gate 8 lifecycle in the resumable bank/runtime path.

Do not promote K4 weights generally, infer arbitrary-route coverage, claim a
complete endpoint, or update any measured throughput-model constant. The next
work must increase layer/route coverage and connect this boundary to real
incremental execution before accepted-token timing or external fidelity gates.
