# PW-0324 — Onboard Prismwing-2 feasibility closure

- Status: complete
- Disposition: scope-decision — evidence-backed onboard frontier closed below
  Prismwing-2
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

Clean implementation commit `a44ffd0903b178eb211362aacf570ad9d32fd516`
ran from a detached clean worktree on the target 16 GiB Apple M1. The analyzer
authenticated PW-0320, PW-0322, and PW-0208 raw/canonical evidence and
recomputed their route, bank, cache, acceptance, and storage ledgers.

The strongest corrected complete-path result in the throughput model is the
PW-0216 ordinary 32-token holdout at `0.0459781517` accepted TPS, batch one and
concurrency one. It is a measured lower milestone, not a full-capability
promotion.

The impossible-best q8 configuration—2,048 K4 identities, a free perfect 4 GiB
cache, measured cold M1 transport, and zero compute or common-weight cost—has a
maximum observed storage-only ceiling of `1.070619598` TPS. Replacing every
observed acceptance with structural `A=8` still reaches only `1.223565254` TPS.

The real q64 transaction contains 4,482 layer/expert identities and authorizes
only `A=3`. Under the same favorable bank/cache/zero-work grants, it moves
`91,589,858,640` bytes and reaches only `0.113673556` optimistic storage TPS.
At exactly two TPS, three accepted tokens permit only
`5,205,672,464.516129` bytes per transaction. The already favorable q64
representation would therefore need another `17.594241525x` uniform reduction,
leaving only `5.683677802%` of its current bytes—a `94.316322198%` additional
cut—before charging compute, proposer, attention, common weights, or endpoint
overhead.

Structural `A=64` reaches `2.425035862` storage-only TPS, but it is explicitly
not an executable survivor: no independently qualified proposer supplies that
acceptance, its proposal cost is omitted, and the actual causal authority is
`A=3`. PW-0208 independently shows that corrected native MTP can improve its
expert-byte objective by at most `1.051643192x` even with perfect proposals,
below its `2x` gate.

PW-0300's original remote JSON hashes remain recorded but the files were not
available on this host. The analyzer therefore reran the same mechanisms from
the receipt-authenticated local checkpoint over 480 deterministic FP8 blocks.
No block fits an exact six- or seven-bit palette; median entropy is
`6.518677887` bits/weight; and the idealized exact top-seven-exponent escape
form has an `88.562011719%` minimum and `89.013671875%` median byte ratio.
The six-bit subset representation still occupies at least `75.012207031%` of
source bytes and remains fidelity-unqualified. These results independently
reproduce PW-0300's causal rejection while preserving the missing-original
limitation.

All nine predeclared closure conditions pass. Analysis Gate 8 records at least
83% free memory, `138,215,424` bytes peak RSS, zero swap growth, zero new
throttled pages, stable protected services, and an explicit release boundary.
The canonical report is:

`/Users/chad/Models/mimo-prismwing/evidence/PW-0324/analysis-002/analysis.json`

SHA-256:
`97d4d20a4c709d42429973e867138495756ce9d52d417f98a7edd40b282ccff3`.

The earlier analysis-001 artifact hashes to
`e2b2d2426c52fb268ed5d60518844ec3642b17006d7cc7c34b36867e987bcc67`
and is preserved but superseded: it emitted the correct boolean value for the
full-capability/fidelity non-promotion condition as a literal rather than
deriving it from the strongest measured status and portfolio. Analysis-002
adds that fail-closed derivation and reproduces the same numerical conclusion.

## Decision

Close the current onboard Prismwing-2 frontier. No measured complete path
reaches two TPS; q8 fails even at structural maximum acceptance; q64 fails at
its actual causal acceptance by `17.59x` in bytes before all omitted work; and
every authenticated onboard representation, proposer, cache/residency,
prefetch, streaming, mixture/exception, sparsity, and K4 branch is rejected,
conditional below the bound, blocked on a failed prerequisite, or lacks the
unchanged full-capability/fidelity gates.

This is decisive, reproducible closure of the current evidence-backed
architecture portfolio, not a theorem against unknown future algorithms. The
only durable reopening premises are companion residency or a genuinely new
representation/proposer that first passes a cheap discriminating fidelity and
physical gate. Companion hardware is inadmissible for this run, so neither is
an active continuation. PW-0324 accepts zero tokens, reports no achieved TPS,
changes no runtime default, and does not rewrite the primary Prismwing-50
target.
