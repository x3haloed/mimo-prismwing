# PW-0333 — Corrected onboard Prismwing-1 feasibility closure

- Status: complete; current authenticated onboard architecture portfolio closed
  below one TPS
- Disposition: scope-decision — evidence-backed onboard Prismwing-1 portfolio
  closed below one TPS; no achieved TPS or universal impossibility claim
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0324, PW-0328, PW-0329, PW-0330, PW-0331,
  PW-0332
- Exactness: analytical synthesis; target and thresholds unchanged

## Question

After repairing target-bonus transaction semantics and replaying the corrected
32-window corpus, does any authenticated architecture in the current onboard
portfolio remain capable of exceeding one sustained accepted token/s on the
16 GiB Apple M1 while retaining the unchanged full-capability, fidelity,
safety, local-inference, and reproducibility constraints?

Companion hardware is inadmissible for this run. This is a fail-closed closure
analysis, not an endpoint measurement and not a theorem against unknown future
algorithms.

## Why a new closure is required

PW-0324 closed the evidence-backed onboard frontier below two TPS. PW-0325
then reopened a narrower one-TPS K4 branch, and PW-0326--PW-0328 replaced its
bonus-free acceptance/routes with corrected causal authority. PW-0329 now
rejects construction continuation even under an impossible-best density-eight,
12-GiB fractional K4 ceiling because required window-tail TPS is below one.
PW-0331 independently shows that byte-neutral rank-one correction can repair
the frozen density-four fidelity row, but that local correctness result cannot
override PW-0329's higher-precedence portfolio tail gate. PW-0330 and PW-0332
cover the remaining named wide-proposer and exact-codec/cache escape branches.

The final one-TPS conclusion therefore needs a new synthesis over corrected
evidence rather than reinterpretation of PW-0324 or any stale PW-0208 route.

## Frozen authorities

The executable analyzer must authenticate and recompute, not transcribe:

1. PW-0324 canonical closure
   `/Users/chad/Models/mimo-prismwing/evidence/PW-0324/analysis-002/analysis.json`,
   SHA-256
   `97d4d20a4c709d42429973e867138495756ce9d52d417f98a7edd40b282ccff3`,
   including its prior portfolio dispositions and explicit unknown-algorithm
   limitation.
2. PW-0328 canonical corrected corpus
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0328/corpus-001/manifest.json`,
   SHA-256
   `36e4f10b6f807f766c87ee7078f5f18ea8fc339dd12e4dbc24f1f4ac6e824403`,
   full `A=232`, observable `A=231`, `sum(U)=142.71808510638297`, all four
   categories, rare-route evidence, byte closure, and Gate 8.
3. PW-0329 canonical corrected K4 analysis
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0329/analysis-001/analysis.json`,
   SHA-256
   `81af4d7b9158fe170503755c38436d5266e41c57a9e67d9c98e142995fdce6f6`.
   Recompute its strongest density-eight, 12-GiB fractional ceiling from
   `514,538,083,176` moved bytes, `A=232`, and favorable bandwidth. Require
   aggregate `1.564789923566762` TPS, fourth-lowest-window
   `0.8827413202181071` TPS, precedence gate two, no work order, and null
   performance claim.
4. PW-0330 cyclic q32 falsifier
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0330/run-001/report.json`,
   SHA-256
   `fbb454f6992ba8e21ade89aff416a494d14625dc126b769f420a861ed6414674`.
   Preserve its conditional scope: the named cyclic schedule fixes `A=4` and
   has a favorable `0.6281149081` TPS storage ceiling if direct-q32 first-chunk
   parity holds; it is not a rejection of arbitrary future proposers.
5. PW-0331 Stage-A analysis
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0331/analysis-004/analysis.json`,
   SHA-256
   `fd5ac314b7e9072f22f773496444678c91f8be0a5165fa24e8df8687906c23c7`.
   Require byte-identical repeated factor hashes, all unchanged sliced gates,
   unseen routed relative L2 `0.008777164859819555`, Stage-A pass, and only
   local Stage-B authorization. Record that PW-0329 gate two prevents further
   portfolio construction; do not relabel this as a K4-bank or endpoint pass.
6. PW-0332 canonical exact top-seven token-cache oracle
   `/Volumes/Elements/mimo-prismwing/evidence/PW-0332/analysis-001/analysis.json`,
   SHA-256
   `e2452a4f2eb9b66ed89097e8e78e5158f7ea53cc00bce8a2ba52c821f61ea085`,
   executed from clean commit
   `d9691ee84bd728093305ed7fa8e403815394bb01`. Authenticate the frozen
   pre-result contract blob
   `073fcb4fd52330acb8ed8d8d645f521ae2ded3b8` and SHA-256
   `e37d1586311989f2e23e1af5737774d1332077e705baaf8d53e96e63e75d90e1`,
   its 381-object fixed census, 480-block exact-codec replay, full PW-0328 q1
   demand SHA-256
   `91fd42fe48033a1b04c1b3d9cdba30a4e6847147064db9946e71c6595bf71db6`,
   and exact capacities `204/230/250`. Recompute the hard absolute-floor
   `49,122` misses, `1,082,237,114,835` encoded bytes,
   `311.8436058584013` favorable storage seconds, aggregate
   `0.7439626647510745` TPS, token p10 `0.5899672933278813`, and
   fourth-lowest-window `0.6962265958830688` TPS. Require every strict
   overall, category, token-tail, category-tail, and window-tail gate false,
   decision `reject_exact_top7_token_cache_oracle`, no analytical survivor,
   no decoder, no runtime default, Gate 8, and null performance claim.
7. The current `TARGET.md`, `RED_LINES.md`, checkpoint revision
   `63651580ca774f8504f676040460aed3e1244ac1`, Apple M1 16 GiB, batch one,
   concurrency one, Gate 8, and explicit companion-hardware exclusion. The
   execution commit must descend from the clean commit freezing the completed
   contract and bind this document's Git blob and SHA-256.

## Closure conditions

Emit `close_corrected_onboard_prismwing1_frontier_below_one_tps` only if all of
the following derive true:

1. No authenticated complete-path result at the repository's unchanged
   correctness scope reaches one sustained accepted TPS. Analytical or
   structural ceilings remain separately labeled.
2. PW-0329's strongest possible current K4 portfolio fails its strict required
   p10 tail gate at or below one. This rejects K4 construction continuation
   even if PW-0331 passes local density-four correctness.
3. PW-0330 rejects the named cyclic-MTP q32 schedule under its exact conditional
   scope, and PW-0324's authenticated proposer-family ledger has no other
   evidence-backed admissible survivor. Do not generalize this to an unknown
   future proposer.
4. PW-0332's absolute zero-escape exact-codec floor fails at least one frozen
   overall, category, token-tail, category-tail, or window-tail one-TPS gate.
   If every strict PW-0332 gate survives, emit `frontier_open` and stop.
5. Every branch that reopened after PW-0324 is therefore rejected, conditional
   below the bound, or blocked on a failed higher-precedence prerequisite; no
   mode has the complete full-capability/fidelity evidence needed for
   promotion.
6. All evidence authorities, formulas, report schemas, claim labels, and Gate
   8 records close exactly. Any missing or stale authority emits
   `frontier_open` or fails closed; absence of evidence is not impossibility.
7. Companion hardware contributes zero storage, memory, compute, bandwidth,
   cost, or performance premise.

This constitutes decisive reproducible closure of the current authenticated
onboard portfolio under the run's fixed constraints. Reopening requires a
genuinely new representation or proposer premise with its own cheap
discriminating correctness and physical gate, not threshold movement or
combination of rejected ceilings.

## Required output and fixtures

The analyzer emits one canonical JSON report with source hashes, exact rational
recomputations, branch dispositions, strongest achieved complete-path result,
analytical ceilings, the PW-0331 prerequisite interaction, companion exclusion,
Gate 8, and explicit limitations. It accepts zero tokens and reports
`performance_claim: null`.

Deterministic fixtures must:

- reject wrong hashes, stale bonus-free routes, dirty Git, overwrite, unsafe
  Gate 8, a non-null performance claim, or any report that relabels ceilings as
  achieved TPS;
- recompute PW-0329 aggregate and fourth-lowest-window TPS from exact fractions
  and enforce gate-two precedence;
- preserve PW-0330's conditional proposer scope and PW-0331's Stage-A-only
  scope;
- force `frontier_open` when every PW-0332 strict gate survives or any reopened
  branch lacks a decisive disposition;
- derive closure only when all conditions pass; and
- preserve the distinction between evidence-backed portfolio closure and a
  theorem about unknown algorithms.

## Claims excluded

- achieved, measured, or complete endpoint TPS at one;
- a universal information-theoretic impossibility theorem;
- rejection of every possible future lossless code or proposer;
- target-faithful K4, an implemented exact codec, or complete full-capability
  promotion;
- any companion-hardware premise.

## Result

The clean analyzer at commit
`78b89aeafe8525584d651c3481dc212bd7a0b48a` authenticates and independently
recomputes all six frozen parent authorities, the corrected PW-0328 q1 demand
stream, the current target and red lines, and the pre-result throughput model.
The canonical report is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0333/analysis-001/analysis.json`,
SHA-256
`8c55fe67ce7bd4b140042f17b7552b6ad7d886f3dcd72c3e95ecb6a4387ff469`.
It closes every authority, formula, schema, claim-label, branch-disposition,
companion-exclusion, and Gate-8 condition and emits
`close_corrected_onboard_prismwing1_frontier_below_one_tps` with no failed
closure conditions.

No authenticated run executes the designated 30-by-512, 60-minute sustained
after-prefill protocol. The strongest clean repeatable post-prefill lower
milestones remain the seven-token multilingual slice at
`0.09587578839539859` TPS and the untouched 32-token ordinary holdout at
`0.0791305231426806` TPS. The latter request's complete-request rate including
prefill is separately `0.04597815174359703` TPS. PW-0203's higher
`0.21984968624124546` value remains a dirty warm, one-block verifier diagnostic
and is neither complete, sustained, nor target qualifying.

The recomputed performance authorities remain analytical and unachieved.
PW-0329's impossible-best K4 placement has `1.564789923566762` aggregate
storage-only TPS but only `0.8827413202181071` at the required fourth-lowest
window. PW-0330's named cyclic-q32 schedule is conditionally capped at
`0.6281149080724167` storage-only TPS, with direct-q32 first-chunk parity still
unproven outside the evidence-backed survivors. PW-0332's absolute zero-escape
exact-codec/cache floor reaches only `0.7439626647510745` aggregate,
`0.5899672933278813` corpus-token p10, and `0.6962265958830688`
fourth-lowest-window storage-only TPS. All required PW-0332 strict gates fail.

The analysis accepted zero tokens (`A=0`, `U=0`), used batch one and
concurrency one on the existing 16 GiB Apple M1, and records a null performance
claim. Its peak resident size was `1,168,261,120` bytes, minimum free memory
was 70%, swap growth and new throttled pages were zero, the release boundary
closed, and all parent and analysis Gate-8 summaries pass. Companion hardware
contributes zero storage, memory, compute, bandwidth, cost, or TPS premise.

## Decision

Reject continuation of the current authenticated onboard Prismwing-1
architecture portfolio: its reopened K4, named cyclic-q32, byte-neutral repair,
and exact-codec/cache branches are respectively rejected, conditionally below
one, or blocked on a failed higher-precedence prerequisite. This is a successful
reproducible analytical closure under the unchanged full-capability, fidelity,
local-inference, and companion-free constraints. It is not achieved endpoint
TPS and not a theorem against unknown algorithms. No endpoint, decoder, K4
bank, PW-0331 Stage B, runtime default, threshold change, or companion branch
is authorized. Reopening requires a genuinely new representation or proposer
premise with independent correctness and physical gates.
