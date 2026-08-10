# PW-0156 — 8K prefill route-coverage storage bound

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0154 analysis
  `1b57250d45f1b24e32f43e93a653fc3d00fa061e37cd0df1c6f0fdff551535f2`;
  8K fixture
  `3b5bc4e8f41fed2a13867bc96ea8236d1630bf994eee5608a8366f1f846a79d5`
- Hardware/runtime: Apple M1 shared 16 GiB; internal SSD; source-FP8
  target-faithful route-only reference; evidence generation only
- Related records: PW-0112, PW-0128, PW-0151, PW-0154, PW-0155; E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0151 grants two P100s an impossible 12.260-second matrix-only floor for an
8K prefill. PW-0154 reduces decode misses with a prompt-trained HBM cache, but
does not measure how many distinct source experts an 8K prefill must acquire.
Before purchasing or writing CUDA, determine whether even four ideal
3.5-GB/s lanes can supply the monotonic distinct expert set within the
15-second TTFT gate.

The whole 8K walk is unnecessary if a causal prefix becomes decisive. Later
tokens cannot remove an already selected `(layer, expert)` record. Execute
frozen prefixes in the sequence 512, 1,024, 2,048, 4,096, and 8,000, stopping
as soon as the optimistic full-request storage bound is crossed.

## Exactness and red-line check

This is target-faithful L1 route and byte accounting over the unchanged
source-FP8 base. The route-only runner still computes every preceding layer
and MoE output required to make each route authoritative; it omits final logits
only after their absence cannot affect routing. The frozen corpus is exactly
8,000 local tokenizer tokens reconstructed from TARGET, RED_LINES, the
validation protocol, and LEARNINGS at commit
`aca9a6044cd348244028850dbb798178695d6bd8`. It is a text performance slice,
not hosted-equivalence or capability evidence.

## Contract

1. Hash-pin the schema-5 fixture, its exact 8,000 token IDs, decoded UTF-8,
   source commit/blobs, tokenizer, config, checkpoint verification, and model
   revision. Fail closed on any identity or token round-trip change.
2. Reuse the source-exact route-only semantics already exercised by PW-0112.
   For each requested causal prefix, execute all 48 layers, preserve all 47
   top-eight routed selections and weights, and reject malformed, duplicate,
   out-of-range, tied-authority, or non-finite routes.
3. Count each distinct `(layer, expert)` once at 25,171,968 source bytes. The
   count is monotonic across longer prefixes; never extrapolate an unobserved
   route or treat logical reuse as physical acquisition evidence.
4. Grant the candidate every favorable assumption: four independent lanes at
   3.5 decimal GB/s each for all 15 seconds; perfect overlap with compute and
   all other work; and PW-0154's 660 complete HBM slots filled by an offline
   oracle before the request. This grants 210,000,000,000 streamed bytes, or
   at most 8,342 complete records after residency.
5. The first decisive count is therefore 9,003 distinct records. Kill the
   two-P100/source-FP8/four-lane 8K TTFT branch as soon as any causal prefix
   reaches at least 9,003. Continue to the next frozen prefix only when it has
   not crossed. A survivor at 8,000 positions remains only a necessary bound.
6. Do not add PW-0151's 12.260-second matrix floor to storage: the primary
   rejection deliberately permits impossible perfect overlap. Do not call
   the resulting bound measured storage or endpoint TTFT.
7. Apply Gate 8 at authority open, every layer boundary, serialization,
   checkpoint release, and final service health. Preserve any safe stop or
   top-k authority failure without interpreting incomplete coverage.
8. Report zero accepted tokens and no endpoint TPS. This experiment can reject
   a hardware/runtime branch; it cannot promote one.

## Promotion and kill rule

Reject the exact cached four-lane source-FP8 prefill embodiment if observed
distinct coverage reaches 9,003 at any frozen prefix. That is decisive even
before dense loading, attention/KV traffic, filesystem overhead, CUDA work,
or the already tight compute floor. If it does not cross at 8K, retain only
the storage-capacity prerequisite and proceed to measured hardware/runtime
work; no purchase follows automatically.

## Pre-result verified-install repair

The first 512-position invocation at `70237f6` failed before checkpoint mapping
and emitted no manifest. The installed `model_mtp.safetensors` retained the
receipt's exact size, inode, nanosecond mtime, and SHA-256
`a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143`,
but macOS reported device `16777231` instead of the captured `16777233`.
Python's verified-install validator already treats device-number changes as
observable mount drift while rejecting size/inode/mtime changes; the Rust
endpoint incorrectly rejected the device field too.

Repair the Rust path to match the existing receipt contract and add every
drifting shard name to the endpoint ledger. Continue to reject changed size,
inode, mtime, receipt status, or malformed recorded hash. This is a
correctness/provenance repair, not permission to skip initial payload hashing
or to trust a copied file without a verified receipt.

## Result

Pending execution from a clean implementation commit.
