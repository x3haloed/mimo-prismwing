# PW-0204 — Arbitrary-prompt generation transaction

- Status: complete
- Disposition: rejected
- Date: 2026-08-10
- Execution mode: source-weight-and-route authority with explicitly modified
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

The first real launch failed closed before prefill because it was initially
pointed at a relocated external checkpoint copy whose device and inode identity
did not match the recovered PW-0049 receipt. The authoritative internal APFS
installation was then recovered; its file identity matches that receipt apart
from explicitly observed device drift. The next launch crossed checkpoint open
and failed at an unnamed page-rounded tensor interval. The runtime now attaches
the tensor name to every no-copy projection-binding failure before any fallback
or layout repair is considered.

The localized tensor is
`model.layers.9.mlp.experts.159.up_proj.weight`: the arbitrary prompt selects a
real expert absent from PW-0203's frozen route, and its source tensor ends at
shard EOF, so no page-rounded mapping can exist without extending beyond the
file. The already-proven QKV fallback copies exactly the tensor's immutable
source bytes into a bounded Metal buffer. That fallback is now the single
projection-region authority and applies equally to routed expert weights;
page-coverable tensors remain no-copy, scales remain no-copy, and no tensor
content or arithmetic changes.

After that repair, the real prompt crossed the routed layer that had failed and
reached the last partial prefill chunk. It then failed closed because the
PW-0203 spine wrappers admitted exactly eight rows even though the ordinary FP8
projection kernels were already specialized for widths one through eight.
The Metal runtime now owns one bounded-row rule: one through eight real rows are
copied unchanged into an eight-row buffer, unused rows are exactly zero, kernels
execute the same per-row arithmetic, and only real rows can route, scatter,
round, update K/V, or become observable. A deterministic fixture covers row
identity, zero padding, malformed shapes, empty input, and widths above eight.

That run crossed partial prefill but remained silent for more than 16 minutes,
making proposer/verifier acceptance and remaining duration unobservable. It was
stopped without a completion claim. The endpoint now writes and synchronizes a
create-new JSONL progress artifact after prefill and every complete transaction,
prints the same bounded progress to stderr, and binds the final report to the
progress-file SHA-256. Interrupted runs remain inspectable and cannot be
silently overwritten.

The first public runtime input is
`evals/fixtures/requests/pw0204-arbitrary-text.txt`. It asks for a concise,
programmatically inspectable two-sentence explanation and contains no route,
answer, proposal, or runtime-specific hint.

## Executed result

Run 001 completed the entire causal path from the arbitrary UTF-8 fixture
through chat serialization, tokenization, six real prefill chunks, six proposal
transactions, target verification, cache rollback, commitment, and decoding.
It emitted exactly 32 verifier-accepted tokens. The report hashes to
`7a6674f5946a195cc58732c4b9acae322a3b6e4dacc802833dab58c86d85b266`;
its synchronized progress log hashes to
`78732c76be24c76e4dcf8d3cc0c7789a7ebf10b599f8bf7aae2f061e40691119`.
The immutable evidence is outside Git under `PW-0204/run-001`, in accordance
with the evidence-size policy.

The run took 1,152,431.038 ms end to end: 431.488 ms preprocessing,
148,569.266 ms prefill, 873,636.069 ms proposal, and 129,401.598 ms
verification. It recorded 1,062,128,594,176 logical source bytes,
1,063,299,391,488 process disk bytes read, and 3,985,113,088 bytes peak RSS on
the 16 GiB Apple M1. Batch size and concurrency were one. The transaction
retention/emission counts were `8/7`, `8/7`, `4/4`, `1/1`, `8/7`, and `8/7`;
the first, second, fifth, and sixth proposal blocks converged.

The observable output was not coherent. Token IDs
`[264,264,264,15,15,15,15,15,12,264,13,13,11,481,481,481,11,481,481,15,11,11,11,481,481,481,481,481,481,481,481,481]`
decode as ` a a a00000- a.., - - -, - -0,,, - - - - - - - - -`. This fails
the milestone's explicit coherent-output requirement even though the causal,
commitment, and resource-accounting path executed successfully.

## Decision

Reject this source-authority modified-L3 generation branch as evidence of a
usable text endpoint. It proves the repeated transaction mechanism and exposes
its physical cost, but it does not prove the claimed user-facing capability.
Do not report its token rate as accepted TPS and do not promote it for Hacker
News. PW-0092 already shows that the much slower source-checkpoint CPU path
also begins with locally authoritative but hosted-divergent output (` a.` for
the fixed Hello prompt), so replacing Metal with that path is not a justified
coherence repair. The next experiment must localize a source/runtime semantic
discrepancy or establish that the pinned checkpoint itself has a different
behavioral contract; it may not tune against a desired sentence or weaken any
gate. No throughput-model constant changes: this rejected run establishes no
accepted performance claim.
