# PW-0326 — Full-match target-bonus transaction repair

- Status: complete
- Disposition: correctness repair promoted; causal q8 pilot authorized
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0203, PW-0204, PW-0208, PW-0216, PW-0325

## Question

Does repairing the fully converged Jacobi commit to emit and carry the target
bonus restore the repository's frozen transaction semantics without changing
the target-authorized token sequence or the mismatch path?

## Prediction error

PW-0204 froze the transaction before the repeated endpoint: on full
convergence, emit the verified proposal suffix, carry the final target
posterior as the next anchor, and retain every proposal-input K/V row. The
acceptance helper already returns `q` in this case. The production commit
instead emits only `proposal[1..]`, retains `q-1` rows, and carries
`proposal.last()`. Its unit test encodes that contradictory behavior.

This wastes one already-computed target decision and shifts every later causal
proposal/route boundary. It does not authorize adding one to old `A` values or
reusing PW-0208 routes: those reports remain immutable evidence for the named
bonus-free implementation.

## Exact transaction contract

For equal proposal/posterior widths `q >= 2`, proposal row zero is the target
anchor committed by the preceding decision.

- On first mismatch at posterior index `i`, preserve existing behavior: emit
  the matching proposal suffix through `i`, then the correcting
  `posterior[i]`; retain `i+1` proposal-input rows; and carry the correction.
- On full match, require `posterior[0..q-1] == proposal[1..q]`; emit
  `proposal[1..q]` followed by `posterior[q-1]`; retain all `q` proposal-input
  rows; and carry `posterior[q-1]` as the next anchor.
- Never commit a draft token before target verification. The target remains the
  sole output authority, and the repair remains target-faithful L2 standard
  speculation rather than a modified model mode.

## Correctness ladder

1. Replace the contradictory convergence fixture before changing production
   logic. It must expect `[42,43,44,45]`, four retained rows, and anchor 45 for
   proposal `[41,42,43,44]` and posterior `[42,43,44,45]`.
2. Preserve mismatch, `q=2`, invalid-width, K/V rollback, final-output clipping,
   and native-MTP retained-history fixtures. Add an explicit two-transaction
   fixture showing that the bonus is emitted once and becomes the next input,
   not emitted twice.
3. Change only the shared commit authority used by the wide verifier and
   repeated endpoint. Record a named transaction semantic in new reports so
   legacy bonus-free evidence cannot be silently mixed with repaired evidence.
4. Run the complete Rust library suite and affected Python corpus/audit tests.
5. Emit a small canonical zero-model-token analysis artifact that authenticates
   the clean implementation commit, old/new fixture vectors, PW-0204's frozen
   contract, relevant source hashes, and Gate 8 state. It must report no TPS.

## Continuation gates

Promote the transaction repair only if every deterministic gate passes and the
mismatch result is byte-for-byte unchanged. Then predeclare and regenerate a
causal corrected q8 route/acceptance panel, starting with one real transaction
per text category. Continue to the full 32-window corpus only if the pilot
confirms bonus carry and stable target authority.

Recompute the Prismwing-1 bank/cache envelope from the regenerated corpus. Do
not construct any missing K4 identity unless that new envelope still requires
dense K4 and separately reauthorizes the six-of-eight falsifier.

## Claims excluded

- achieved or projected accepted TPS;
- `A+1` projections from old reports or stale-route economics;
- proposer improvement, K4 fidelity, bank/cache construction, or a runtime
  performance default;
- Prismwing-2, Prismwing-50, or a target rewrite.

## Result

Clean implementation commit `accb1e96e223f253db4b2772da18ccf087cb7454`
changes the shared commit authority to `target_bonus_full_match_v1`. Proposal
`[41,42,43,44]` with target posterior `[42,43,44,45]` now emits
`[42,43,44,45]`, retains all four proposal-input rows, and carries 45. The
legacy `[42,43,44]`/three-row/anchor-44 vector is recorded as superseded.
The mismatch control remains exactly `[13,15,13,15,481]`, five retained rows,
and anchor 481.

Two consecutive full matches emit `[42,43,44,45,46,47,48,49]`, demonstrating
that the first target bonus is neither skipped nor duplicated. Dedicated Rust
fixtures also cover q=2, terminal output clipping, cache-position arithmetic,
and native-MTP hidden/input pairing. Future generation reports carry an
explicit target-bonus semantic. The legacy PW-0208 auditor remains unchanged;
a separate fail-closed repaired-semantic auditor admits `A=q`, distinguishes
verifier retention from terminal clipping, and validates the exact bonus.

The canonical analyzer ran all 120 Rust library tests and all 15 affected
Python audit/corpus tests with zero failures. Gate 8 recorded 79% free memory,
zero swap growth, zero throttling, stable protected services, and a release
boundary. The report accepts zero model tokens, moves zero model bytes, and
makes no performance claim.

Canonical report:

`/Users/chad/Models/mimo-prismwing/evidence/PW-0326/analysis-002/analysis.json`

SHA-256:
`57a7c2ee1044b6b3fd3710ecf27fd013c78c7a6533e3d9636adaaa5765e3becb`.

Analysis-001 hashes to
`ed674a3041c7487c36e1c607b11febaa8f251657c8775580e794c78dd8bf80fe`
and is preserved but superseded. Its core result was correct, but it lacked a
repaired-semantic corpus auditor, omitted the legacy vector, did not prove the
named Rust/Python fixtures ran, and retained one contradictory serialization
fixture.

## Decision

Promote the target-bonus transaction repair as correctness, not performance.
Authorize PW-0327's four-category, one-transaction causal q8 pilot. Preserve
PW-0204 through PW-0325 as named bonus-free historical evidence and forbid
`A+1` projections or stale-route economics. No endpoint TPS or runtime
performance default changes.
