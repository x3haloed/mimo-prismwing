# PW-0316 — Layer-4 four-K4/four-source mixed Metal transaction

- Status: completed
- Disposition: rejected at pre-Metal semantic gate
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0315

## Question

Can the route-specific layer-28 K4/source bundle and Metal transaction be made
layer-generic and execute the qualified four-expert layer-4 bank together with
four exact source-FP8 fallbacks on a real PW-0116 route?

## Hypothesis and mechanism

PW-0315 qualifies layer-4 K4 experts 64, 96, 31, and 232. PW-0116 position 1
selects exactly those four identities plus source experts 88, 245, 223, and
151. A schema-2 bundle can therefore preserve all eight live identities while
testing a 4/4 split absent from the original layer-28-only `(3,5)|(5,3)`
runtime.

This is the smallest vertical slice that joins full-checkpoint target-native
construction to the measured panel-batched Metal executor. It is also the
container/loader boundary required by a resumable larger bank.

## Authorities

- checkpoint revision `63651580ca774f8504f676040460aed3e1244ac1` and verified
  installation receipt SHA-256
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
- PW-0116 corpus manifest SHA-256
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
- layer 4, position 1 route IDs `[96,64,232,31,88,245,223,151]` and its
  captured weights;
- PW-0315 canonical summary SHA-256
  `07b3d3793a6750a030eb5b7e12a0add1b603d48758a85e6f45b44504e404d0e8`;
- K4 run-001 artifacts for experts `[96,64,232,31]`, each independently
  repeated and semantically qualified by PW-0315;
- source-FP8 checkpoint tensors for experts `[88,245,223,151]`;
- PW-0308 hash-bound Metal kernels and target-host safety contract.

## Protocol and gates

1. Authenticate all authorities and reconstruct the PW-0116 source route and
   layer final bit exactly before writing artifacts.
2. Export raw source-FP8 weights/scales for the four fallback identities into
   isolated fixtures. Recompute each complete source expert at position 1 and
   require exact equality with its expert-major PW-0116 answer row.
3. Build one 16-KiB-aligned schema-2 bundle containing the four K4 records,
   four source records, TLUT, payload hashes, dynamic layer identity, and exact
   route authority. Refuse overlap, unknown roles, or identity substitution.
4. Independently reconstruct the mixed route from saved K4 candidate outputs
   and exact source outputs. Require routed relative L2 below `0.01` and the
   layer-final relative L2 below `0.01` for the selected row.
5. Generalize the Rust loader only enough to admit schema 2, arbitrary declared
   layer, and `(4,4)` while retaining schema-1 layer-28 compatibility and every
   existing fail-closed invariant.
6. Execute the mixed route on Metal using the captured MoE input and route
   weights. Require every output F32 bit to match the independently constructed
   schema-2 candidate fixture, including 20 warmups and 100 timed samples.
7. Run twice from one clean pushed commit. Require identical output bits,
   bundle/manifest/fixture hashes, and semantic metrics. Timing and host
   counters may differ.
8. Record cold/warm state, complete wall/GPU distributions, logical and
   physical bytes, batch size one, concurrency one, accepted tokens zero, and
   Gate 8 release evidence.

## Decision rule

- On full pass, promote the schema-2 layer-generic mixed transaction and use it
  as the partial-bank/full-endpoint integration boundary.
- If source replay or bundle readback fails, reject the builder/authority path.
- If Metal differs from the independent candidate, reject the generalized
  executor even if aggregate metrics are close.
- If the row exceeds the unchanged one-percent routed/final gates, retain the
  four-expert bank evidence but reject this mixed transaction as an endpoint
  building block.

## Claims excluded

- arbitrary routes or identities outside this eight-expert row;
- a complete bank, complete decoder, or accepted-token execution;
- hosted, multimodal, long-context, or capability equivalence;
- Prismwing-2, 34.3 TPS, or Prismwing 50 completion.

## Result

The clean `018e18dec27008ae6c84a60f6f45748563fba1ed` run rejected the
candidate before bundle construction or Metal execution. The canonical raw
rejection manifest is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0316/build-003/rejection.json`
and hashes to
`7e5560cf2cdc2abdec8ec1a17af0462f69fa7204f8ba528808ce1f046d0e6ff4`.

The first fail-closed attempt exposed a source-replay prediction error:
executing only position 1 changed Accelerate's GEMM shape relative to the
PW-0116 expert-major capture. Expert 88 then differed by `0.000176921`
relative L2 and `0.015625` maximum absolute error. Replaying each complete
recorded expert batch before selecting position 1 resolves the discrepancy.
Experts 88, 245, 223, and 151 all reproduce every captured output bit. The
builder now regression-tests and enforces that batch-shape invariant.

With exact source fallbacks established, the four qualified K4 identities
produce routed relative L2 `0.0109888419`, above the exclusive `0.01` gate.
The layer-final relative L2 is `0.0027743952`, but both declared boundaries
must pass. No bundle was emitted and the generalized Metal executor was not
run. This is compatible with PW-0315: that record's bank-wide gates permit a
worst routed row of `0.014743290`, whereas PW-0316 deliberately declared a
stricter one-percent row-local integration gate.

Gate 8 passes through the rejection release boundary: 70% minimum system-free
memory, `829,079,552`-byte peak RSS, `194,562,304`-byte final footprint, zero
swap growth or new throttled pages, and stable protected services. The run
accepted zero tokens and changes no throughput-model constant.

## Decision

Kill the four-K4/four-source route transaction under the unchanged row-local
gate. Retain the schema-2 layer-generic loader work and the four-expert bank as
qualified components, but do not treat their union on this row as an endpoint
building block. A bounded subset audit identifies `[64,232,31]` with five
exact-source fallbacks as the next 3/5 experiment; it measures `0.0066623394`
routed and `0.0019744170` layer-final relative L2 on the same row. That
successor requires its own predeclared record and evidence.
