# PW-0327 — Target-bonus q8 causal route pilot

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0204, PW-0208, PW-0325, PW-0326

## Question

Does the promoted `target_bonus_full_match_v1` transaction behave correctly on
one real source-target q8 verification transaction in each of the ordinary,
code, multilingual, and rare-route text categories, and does that evidence
justify regenerating the complete 32-window route/economics corpus?

## Contract

Use the existing receipt-authenticated MiMo-V2.5 checkpoint, Apple M1, source
weights/routes, corrected QKV layout, and target self-proposer. This is a
target-faithful correctness/route pilot, not a fast proposer or endpoint run.

Run `arbitrary-text-route-trace` in a fresh process for each frozen PW-0208
prompt with a two-token diagnostic output bound. The prefill supplies the first
observable target anchor; the runtime must still execute exactly one complete
q8 proposal/verifier transaction, retain its full verifier-authorized vector
in the report, and clip only the observable output/cache prefix to the one
remaining requested token.

Prompt SHA-256 authorities:

- ordinary: `d15e7fad81828b710303ce5e9dc5fd9c2104450108eb627167e6bc2080b9ee5d`;
- code: `ad2940784d5028baa1dfab4585cb3a5a7fbffa22ca224f455fabc851549daefa`;
- multilingual: `6ece2dd3189d6b482f3356d344db6228e428db60a7530283eedc39be77d1beca`;
- rare-route: `5a71638364fff89af264dd3acea1ce31ef92128c3922cc8fb64826e793643373`.

For each report, authenticate the clean commit, model/checkpoint/kernel and
prompt identities, `target_bonus_full_match_v1` semantic, Apple M1, batch one,
concurrency one, cold process state, one transaction, exact eight-row verifier
trace at all 48 layers, full physical/logical byte ledgers, and Gate 8.

## Correctness and continuation gates

- Recompute the commit from proposal/posterior tokens. On convergence require
  the exact seven-token proposal suffix plus final target bonus, eight verifier
  rows, and the bonus as next-anchor authority. On mismatch require the
  unchanged first-correction rule.
- Require the single observable clipped token to be the prefix of the full
  verifier-authorized vector; require one retained output row while preserving
  the larger verifier-retention count in evidence.
- Recompute `U`, unique `(layer, expert)` identities, source bytes, `A`, and
  `A/U` from exact route rows. Here `A` means the full verifier-authorized
  transaction, not the intentionally clipped observable count.
- Require target-authorized token identity and all safety gates; make no TPS
  claim from a one-transaction diagnostic.

Authorize the complete four-category 32-window causal recapture if all reports
pass semantic/authority gates and at least three categories exercise the
full-match bonus branch. Otherwise preserve the exact mismatch and decide from
the observed `A/U`; never edit or project old reports.

## Claims excluded

- sustained or complete endpoint TPS;
- native-MTP proposal quality or latency;
- stale-route `A+1`, K4 fidelity, bank/cache construction, or a runtime
  performance default;
- multimodal/full-capability promotion, Prismwing-2, or Prismwing-50.

## Result

Unexecuted.

## Decision

Unexecuted. Commit this contract before launching any target-model process.
