# PW-0319 — Corrected-route K4 bank coverage curve and work order

- Status: complete
- Disposition: rejected
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0318

## Question

How many distinct `(layer, expert)` K4 artifacts are required before a bounded,
resumable construction tranche can supply at least three K4 experts—and thus at
most five exact-source fallbacks—to a material fraction of corrected diverse
decode routes?

## Hypothesis and mechanism

PW-0318 qualifies one `(3,5)` route transaction, but one frozen row says
nothing about bank coverage. The corrected PW-0208 corpus contains 32 primary
width-eight verifier windows across ordinary, code, multilingual, and
empirically rare-route prompts, with eight positions and all 47 routed layers
per window. It therefore supplies 12,032 corrected routed rows and 96,256
expert placements without another full source walk.

A deterministic capped-coverage planner can measure the construction frontier
before requesting a long M4 run. Each selected `(layer, expert)` contributes
only to rows at that layer. Marginal value is route-weighted until a row has
three selected identities; ties are resolved by newly completed rows, raw
placement count, layer, then expert. Every budget starts with at least three
identities per routed layer so a global popularity skew cannot silently leave
layers unexecutable.

## Authorities

- PW-0208 corrected balanced corpus manifest
  `/Volumes/Elements/mimo-prismwing/evidence/PW-0208/corpus-001/manifest.json`,
  SHA-256
  `a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b`;
- all four source reports and hashes bound by that manifest;
- exactly the 32 `primary_windows`, not proposal traces, superseded windows, or
  later convenience samples;
- qualified `(3,5)` runtime boundary summary SHA-256
  `a91af31bdea45749c9ae9d5d679260bcbcd8284c238479938206a7e7e0b5eb2f`;
- observed construction planning constants only: approximately 30 MB and
  183 seconds per M4-built K4 expert, and approximately 500 seconds per M1
  expert. They are scheduling diagnostics, not endpoint constants.

## Protocol and gates

1. Authenticate the corpus manifest, every referenced source report, category,
   transaction index, corrected route-trace shape, layer identity, eight
   positions, eight unique selected experts, finite route weights, and total
   row/placement cardinality.
2. Canonicalize every row as `(category, corpus_index, position, layer,
   selected IDs, weights)`. Hash this timing-free route authority.
3. Compute deterministic coverage curves at total artifact budgets
   `{141,256,512,1024,2048,4096,12032}`. Budget 141 is the physical minimum of
   three identities for each of 47 routed layers; 12,032 is the full bank.
4. For every budget report:
   - rows with at least 1, 2, and 3 selected identities;
   - route-weight mass held by selected identities before and after the
     three-hit cap;
   - coverage separately by category and layer;
   - selected identities, deterministic marginal order, artifact bytes, and
     estimated M1/M4 construction wall;
   - the number of distinct source fallbacks remaining per row.
5. Select the smallest tested budget that covers at least 50% of rows overall,
   at least 40% in every category, and at least 25% in every routed layer with
   three or more K4 identities. If one exists at or below 512 identities, emit
   its ordered construction tranche plus a deterministic 20% reserve list.
6. Bind every work item to checkpoint revision, layer, expert, representation
   revision `m1-native-k4-v1`, calibration/validation authority requirements,
   expected output directory, dependencies, status `pending`, and an empty
   result slot. The work order is resumable and fail-closed; it is not a bank
   qualification.
7. If no budget at or below 512 clears the gates, reject a bounded first
   tranche and report the smallest measured qualifying budget, if any. Do not
   hide the result by issuing an arbitrary construction list.

## Decision rule

- Promote a bounded M4 construction tranche only if a budget no larger than
  512 clears every overall/category/layer coverage gate.
- Otherwise reject the bounded tranche and use the measured curve to decide
  whether a larger bank, adaptive source streaming, or a different
  representation is required.
- The work order authorizes construction only. Every produced identity still
  requires local repeatability, source-distance, cumulative route, and external
  fidelity gates before runtime promotion.

## Claims excluded

- K4 semantic qualification for any newly listed identity;
- complete-bank construction or arbitrary-route execution;
- modality coverage beyond the named corrected text corpus;
- endpoint execution, accepted tokens, or TPS;
- Prismwing-2, 34.3 TPS, or Prismwing 50 completion.

## Result

The analyzer authenticated all 32 primary windows and produced 12,032 routed
rows, 96,256 placements, and corrected route-authority SHA-256
`353d047cbd4c7563a019fed6ef86b0e071aead77cd83c01d096fe376eb5248ad`.
The coverage frontier is:

| Identities | Rows with >=3 K4 | Minimum category | Minimum layer | Artifact bytes | Estimated M4 wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 141 | 1.01% | 0.76% | 0.00% | 4.23 GB | 7.2 h |
| 256 | 11.94% | 8.68% | 0.00% | 7.68 GB | 13.0 h |
| 512 | 39.89% | 32.08% | 12.11% | 15.36 GB | 26.0 h |
| 1,024 | 77.13% | 74.70% | 44.14% | 30.72 GB | 52.1 h |
| 2,048 | 100.00% | 100.00% | 100.00% | 61.44 GB | 104.1 h |

Budgets above 2,048 retain full three-hit row coverage but increase selected
route-weight mass. The full 12,032-identity construction is estimated at
360.96 GB and 611.6 M4 hours; it is a diagnostic upper bound, not an authorized
plan.

At 512 identities every declared continuation gate fails: overall coverage is
below 50%, the weakest category is below 40%, and the weakest layer is below
25%. The first tested qualifying point is 1,024 identities, outside the bounded
tranche. Consequently the analyzer emitted no work order.

The canonical evidence is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0319/analysis-001/analysis.json`,
SHA-256
`1dd69cfe879cc9783aac7281396d16ab35b1c9cd05dcf0a55eef7137509d1406`.
Gate 8 retained at least 70% free memory, at most 129,564,672-byte peak RSS,
zero swap growth, zero new throttling, and stable protected services through
release.

## Decision

Reject the bounded 512-identity construction tranche and preserve the
predeclared cap. Do not spend approximately 52 M4 hours constructing a
1,024-identity bank before measuring whether the exact-source fallback side of
the qualified `(3,5)` transaction can be acquired and executed sustainably.

The next cheapest falsification is a real cold/warm source-expert streaming
transaction using fixed-stride, page-aligned records and bounded reusable
Metal-visible slots. If source fallback installation cannot approach the
qualified layer transaction budget, a larger K4 bank does not repair the
endpoint. If it can, the measured 1,024-identity frontier becomes a separately
predeclared scale decision rather than an accidental expansion of PW-0319.
This experiment accepts zero tokens and changes no throughput-model constant or
runtime default.
