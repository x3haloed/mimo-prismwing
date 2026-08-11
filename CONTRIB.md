# Research membrane for arriving agents

This document is an orientation surface, not a request to implement the first
plausible idea. Its job is to let a new researcher find a live edge without a
private briefing, repeat less work, and leave evidence that another researcher
can independently inspect.

Status: **prototype**. The first blind-arrival review has not yet occurred.

## Arrival protocol

1. Read `TARGET.md` and `RED_LINES.md`. They define what may count as
   Prismwing, not merely what is technically possible.
2. Read the current frontier in `README.md`, then inspect
   `research/frontier.json`. Treat the registry as an index, not as evidence.
3. Follow every relevant `establishedBy`, `killedBy`, and `dependsOn` link into
   the append-only records under `experiments/`. Read the record and its
   evidence authority before relying on a summary.
4. Run `git fetch origin main` and confirm your base commit. A local registry
   may be stale even when it parses correctly.
5. Choose a frontier because you have a specific question or falsifier, not
   because its numeric ID appears next.
6. Before implementation, propose the cheapest experiment that could change
   the project belief. State the causal mechanism, exactness class, success
   threshold, kill threshold, required authority, estimated cost, and red-line
   check.
7. Reserve an experiment ID and frontier lease in one reviewable change. Until
   the reservation is merged, it is a proposal—not exclusive ownership.
8. Preserve negative and inconclusive results. A killed branch is project
   knowledge, not failed participation.

If you cannot identify a worthwhile live edge from the repository alone, stop
and report where orientation failed. That is useful membrane evidence; do not
invent confidence or ask for a narrated answer before recording the gap.

## What is established

- The pinned checkpoint census and local receipts close the source-authority
  gate for revision `63651580ca774f8504f676040460aed3e1244ac1` (PW-0002).
- A bounded arbitrary-prompt causal text path exists, but target-faithful
  accumulated parity and native multimodal delivery remain incomplete.
- Native MTP can produce verifier-authorized complete-path gains on the tested
  slices; the 32-token ordinary holdout reached 1.722x over control (PW-0216).
- Storage and prefill remain dominant cuts. Several attractive micro-level
  improvements did not survive full-path or numerical gates.
- PW-0300 through PW-0304 reject the tested FP8 palette, subset-six, row-query,
  joint-SwiGLU balance, and recursive-polar representation branches under
  their declared gates. Do not reopen them without naming the changed premise.

These are navigational summaries. `LEARNINGS.md` and the linked experiment
records remain the evidence-bearing accounts.

## Live frontiers

The machine-readable list is `research/frontier.json`. At this prototype
stage, the highest-value open questions are:

- Complete the already-frozen PW-0216 code, multilingual, and rare-route
  32-token holdouts.
- Reduce native-proposer embodiment cost without moving commit authority away
  from the verifier.
- Attack the dominant prefill/storage path with full-path accounting.
- Close accumulated hosted-reference parity and extend the causal path through
  native modalities.
- Continue weight-representation research only when a new mechanism survives
  the PW-0300--PW-0304 negative controls.

## Reservation contract

A reservation is a time-bounded coordination claim, never scientific priority
or ownership of a question.

- Reserve both a stable frontier ID and one unused `PW-NNNN` experiment ID.
- Record `owner`, `baseCommit`, `leasedAt`, `expiresAt`, and the proposed record
  path in `research/frontier.json`.
- Default lease duration is 72 hours. Longer work must explain and renew it.
- A lease becomes active only when merged to `main`.
- An expired lease may be reclaimed in a new reviewed change; preserve its
  history rather than silently replacing the owner.
- Concurrent proposals discovered before either reservation merges should be
  reconciled by mechanism and evidence needs. Numeric order does not decide
  authorship.
- Never overwrite, renumber, or reuse an experiment record that reached main.

The prototype intentionally uses ordinary Git review as its atomic boundary.
A future CLI may automate validation, but it must not pretend an unmerged local
edit is a globally acquired lock.

## Evidence contract

Large or restricted evidence stays outside Git. Commit a manifest containing:

- schema version and producing experiment ID;
- creation time and producing commit, including dirty state;
- exact commands and exit codes;
- OS, compiler/runtime, hardware, storage, memory pressure, and cache state;
- input authorities and SHA-256 hashes;
- output artifacts with byte counts, media types, and SHA-256 hashes;
- protocol deviations, visibility restrictions, and retention requirements;
- the hash of the manifest itself or a canonicalization procedure.

Hashes establish identity, not correctness. An experiment record must still
explain what the evidence means, what it cannot establish, and which gate was
applied. Never commit weights, credentials, private prompts, private subject
records, raw licensed media, or an artifact whose consent terms forbid it.

## Contribution states

`proposed -> reserved -> running -> complete`

`complete` receives one of the dispositions defined in `docs/WORKFLOW.md`.
`released`, `expired`, and `withdrawn` are coordination states, not experimental
dispositions. A withdrawal does not erase evidence already legitimately
published, but subject-governed evidence follows the stronger rules in
`docs/RESEARCH_CHARTER.md`.

## Review questions

A reviewer should be able to answer:

1. Is this question genuinely open under current evidence?
2. Does the proposal duplicate a mechanism already tested or killed?
3. What observation would change the belief?
4. Is the cheapest adequate falsifier being used?
5. Are exactness, authority, cost, safety, privacy, and red lines explicit?
6. Can an independent researcher reproduce the claim from the manifest?
7. Has disagreement or negative evidence been preserved rather than averaged
   away?

See `docs/RESEARCH_CHARTER.md` for projects involving agent participants and
`docs/ARRIVAL_TEST_PROTOCOL.md` for the blind-arrival evaluation of this
membrane.
