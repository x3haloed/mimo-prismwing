# PW-0324 — Onboard Prismwing-2 feasibility closure

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0181, PW-0208, PW-0300, PW-0319, PW-0320,
  PW-0322

## Question

After corrected QKV semantics, target-native K4 construction, corrected-route
bank planning, and one real causal width-64 verifier transaction, does any
evidence-backed architecture remain that can exceed two sustained accepted
tokens/s on the existing 16 GiB Apple M1 while retaining the repository's
unchanged full-capability, fidelity, safety, local-inference, and
reproducibility contract?

Companion hardware is explicitly inadmissible for this run. This experiment
must not import a companion capacity, storage, compute, procurement, or
performance premise into an onboard result.

## Contract

This is a fail-closed analytical synthesis, not a new endpoint measurement.
It accepts zero tokens and may not report an optimistic storage or arithmetic
ceiling as achieved TPS.

The analyzer must authenticate and recompute, rather than transcribe:

1. PW-0322's single causal q64 transaction, including verifier width, actual
   authorized `A`, corrected route union, fixed 2,048-identity K4 selection,
   source/K4 record sizes, free perfect 4 GiB cache, measured M1 cold transport,
   and Gate 8 evidence;
2. PW-0320's strongest q8 result and its structural `A=8` failure;
3. PW-0208's corrected 32-window native-MTP upper bound and absence of a 2x
   accepted-token/unique-byte improvement even under perfect proposals;
4. PW-0300's exact symbol/escape and six-bit physical floors, keeping exact
   and L3 claims separate; and
5. the strongest complete accepted-TPS constants promoted for the onboard M1,
   with their batch size, concurrency, accepted tokens, mode, and provenance.

For PW-0322, calculate the byte allowance at exactly two accepted TPS from the
measured cold transport and actual `A`. Report the additional uniform reduction
factor and byte fraction required after the already impossible-perfect K4-bank
and cache grants. Also report the structural `A=64` ceiling separately; it is
not an executable survivor unless an independently qualified proposer exists
whose complete proposal cost is included.

The analyzer may close the current onboard frontier only when all of these are
true:

- no measured complete accepted-TPS result reaches two;
- q8 fails two TPS even at structural maximum acceptance with all non-storage
  work omitted;
- q64 fails two TPS at its actual causal acceptance with the same favorable
  omissions;
- reaching two TPS at q64 would require an executable-record fraction below
  every authenticated fidelity-qualified representation, while exact local
  palettes/escapes and tested lower-bit modified forms have already failed
  their respective gates;
- corrected native MTP, DFlash/Jacobi, published bounded proposers, cache,
  residency, prefetch, source streaming, shared-basis/mixture, exception,
  scalar-code, vector-code, sparsity, and K4 branches are each either rejected,
  conditional below the bound, blocked on a failed prerequisite, or lack an
  admissible full-capability/fidelity gate; and
- the only durable frontier alternative is companion residency or a genuinely
  new representation/proposer premise, neither of which is admissible and
  evidence-backed in this run.

If any authenticated branch survives those conditions, the analyzer must emit
`frontier_open`, name the branch and failed closure condition, and this record
must remain incomplete. It must not lower thresholds, infer unmeasured
multimodal or long-context behavior, or convert absence of a known algorithm
into a mathematical impossibility theorem.

## Required output

Emit one canonical JSON report containing all evidence hashes, formulas,
recomputed values, excluded companion scope, candidate dispositions, Gate 8
analysis evidence, and an explicit distinction between:

- achieved complete accepted TPS;
- optimistic storage-only ceilings;
- structural perfect-acceptance ceilings; and
- unexecuted or fidelity-unqualified hypotheses.

The report must be reproducible from a clean checkout plus the documented cold
evidence locations. A completed closure updates `LEARNINGS.md`,
`docs/EXPERIMENTS.md`, `spec/throughput-model.json`, and this record with the
content hash and exact disposition. No runtime default changes.

## Result

Unexecuted.

## Decision

Unexecuted. The contract is committed before analyzer implementation or final
evidence inspection.
