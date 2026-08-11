# PW-0204 — Arbitrary-prompt generation transaction

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-10
- Execution mode: target-faithful proposal verification with explicitly named
  Metal-native L3 reduction arithmetic
- Hardware/runtime: existing 16 GiB Apple M1 and internal SSD only
- Related records: PW-0050, PW-0092, PW-0187, PW-0203

## Capability contract

Given arbitrary UTF-8 text, the pinned tokenizer serializes a real prompt; the
pinned MiMo checkpoint performs prefill and retains authoritative K/V; a real
proposer supplies a width-eight block; the PW-0203 accelerated target verifies
that block; one transaction authority commits only verified tokens, rolls K/V
back to the accepted proposal rows, and carries the correction token into the
next proposal window. Repetition must produce 32–64 observable decoded tokens.

The endpoint must record preprocessing, prefill, proposal, verification,
rollback, and decoding inside complete wall time. It must also record commit,
dirty state, model and tokenizer identities, cold/warm state, batch size,
concurrency, accepted tokens, `A`, `U`, bytes moved, memory/swap observations,
hardware, and hashes for every external authority.

This milestone does not satisfy the final multimodal, hosted-parity, 50-TPS, or
independent-reproduction gates in `TARGET.md`. Proposal-only tokens never count
as output. The source checkpoint remains the output authority; a modified or
external draft must be named separately and may not commit unverified tokens.

## Risk frontier

PW-0203 verifies one frozen block but has no retained generation transition.
Its K/V state contains all eight proposal input rows even when only a prefix is
accepted, and its report does not define whether the correction token is
emitted or becomes the next input. Repeating that command would therefore make
cache and output authority implicit.

The first correctness fixture defines the transaction explicitly:

1. the anchor token was committed by the preceding target decision;
2. matching proposal suffix tokens become observable;
3. the first mismatching target posterior is emitted exactly once as the
   correction;
4. only proposal rows through the mismatch retain authoritative K/V;
5. the correction becomes the next window's anchor and is evaluated on the
   next transaction;
6. a fully converged window emits its verified suffix and carries the last
   posterior as the next anchor without inventing a correction.

## Promotion and kill gates

Promote the transition only after deterministic mismatch, convergence, invalid
width, and cache-retention tests pass. Then integrate it into one repeated
runtime path; do not create a second verifier or tokenizer authority.

Kill any proposed endpoint that reuses K/V for rejected rows, emits a draft
token before target verification, omits proposer time from complete wall time,
or relies on PW-0187's prompt-specific captured layer states for an arbitrary
prompt.

## Current result

The deterministic transaction fixture is implemented beside the existing
PW-0203 acceptance authority. The production K/V representation now has a
fail-closed rollback operation, and PW-0203 applies the same transaction result
to discard rejected proposal rows after verification. Deterministic tests cover
mismatch correction, convergence, malformed widths, rollback preservation,
rollback-to-empty, and attempted rollback growth.

The first repeated runtime path is now implemented as
`arbitrary-text-generate`. It authenticates the model, tokenizer, source
checkpoint, receipt, kernel, commit, and dirty state; applies the pinned
single-user chat serialization to UTF-8 read from a file; prefills in bounded
width-eight chunks; obtains each width-eight proposal greedily from the same
source checkpoint; verifies it with the PW-0203 wide path; commits through the
transaction authority; and repeats until exactly 32–64 tokens are observable.
Complete wall time includes authority opening, preprocessing, Metal compilation,
prefill, proposing, verification, rollback, decoding, and safety probes.

The target itself is deliberately the initial proposer. This is computationally
redundant and not a performance candidate, but it crosses the arbitrary-input
and repeated-state risk frontier without accepting an unverified draft or
creating another model-semantic authority. A faster proposer may replace it
only after it produces real proposals and preserves target verification.

Real checkpoint execution and evidence remain unexecuted, so accepted endpoint
tokens and endpoint TPS remain zero.

The first public runtime input is
`evals/fixtures/requests/pw0204-arbitrary-text.txt`. It asks for a concise,
programmatically inspectable two-sentence explanation and contains no route,
answer, proposal, or runtime-specific hint.
