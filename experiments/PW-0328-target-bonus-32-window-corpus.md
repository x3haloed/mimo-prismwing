# PW-0328 — Target-bonus 32-window causal corpus

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-27
- Owner: Codex
- Parent experiments: PW-0208, PW-0325, PW-0326, PW-0327

## Question

What are the real chronological q8 acceptance, route-union, and byte values
after the repaired full-match target bonus across eight windows in each frozen
ordinary, code, multilingual, and rare-route category, and can those windows
form a complete native-MTP history authority for the next Prismwing-1
falsifier?

## Prediction being resolved

PW-0208 and PW-0325 are exact for their historical bonus-free transaction
boundary, but the target bonus changes the next generated anchor and therefore
can change every later proposal, route, `A`, and `U`. PW-0327 proves the repair
works in one transaction per category and passes its predeclared continuation
gate. It does not authorize projecting `A+1` or reusing later legacy routes.

The fixed 64-token capture may now contain only eight transactions when every
q8 proposal converges. The old corpus builder skipped transaction zero because
it bound the proposal anchor only to a preceding verifier window. That premise
is superseded for a complete-history corpus: transaction zero has the exact
preceding target-hidden authority in the already authenticated prefill
artifact. Its target hidden history is every serialized-prompt row, its target
input IDs are the serialized prompt IDs, and its MTP layer-zero input IDs are
the prompt IDs shifted left with the first target anchor appended.

## Frozen inputs and execution

Use the receipt-authenticated internal MiMo-V2.5 checkpoint, `spec/model.lock.json`,
`kernels/block_fp8_gemv.metal`, Apple M1, batch one, concurrency one, q8 target
self-proposal, and the four PW-0208 prompt files and hashes frozen in PW-0327.
Companion hardware is inadmissible.

Run one clean cold process at a time from the committed contract/runtime:

```text
target/release/prismwing native-mtp-window-capture <checkpoint> \
  spec/model.lock.json <checkpoint-receipt> kernels/block_fp8_gemv.metal \
  <frozen-prompt> <category> verification-layer47-hidden.f32 report.json \
  <clean-capture-commit>
```

Preserve raw reports, progress logs, and hidden payloads outside Git. Reuse the
PW-0208 prefill artifacts only after authenticating their report/payload hashes,
prompt tokens, first anchor, model/checkpoint/kernel identities, shape, finite
payload, safety evidence, and exact agreement with the new generation prefill:

| Category | Prefill report SHA-256 | Target-hidden SHA-256 |
| --- | --- | --- |
| ordinary | `11a02fd9d653c6351ed22d03f7d39efb80ee8d6009fc9a3d22d41fd2f42d1ddb` | `5df877426383c5750a09c0d54e9d992d3d3f99e9f0c15ee5eaece5312659240c` |
| code | `a75aab62fa434f73d8f0053919fc9c3eab68c71e96a690cfed6f8871306b35ae` | `616ac368c4893517083fef39e58ecc41b85001cdac7ddedf9db66d3ea249b938` |
| multilingual | `b8c68eac9834c24ea09ffa65e7f3f5ef2ef5c015209c862419f4471480e846d2` | `bc8d7a03be5860a99ba1398a6c6697c63a94551d0d3b33b3545791c4b10a3468` |
| rare-route | `385425155ab48a965169d860ff56c09e8967325e536b72dfd3b5e8e164c83773` | `d50c34d1766c1cbf1a2fb1c42c96338f7a96b3740091851460f223bf4b11005c` |

## Correctness and evidence gates

For every new source report, require the repaired semantic, clean capture
commit, exact model/checkpoint/tokenizer/kernel/prompt identities, Apple M1,
cold state, requested and accepted token count 64, requested-maximum stop,
batch one, concurrency one, at least eight chronological q8 transactions,
complete route and proposal traces, progress-hash closure, hidden-artifact
hash/shape closure, positive ordered logical/physical byte ledgers, and every
normative Gate 8 memory/service condition.

Recompute each transaction commit from proposal and posterior tokens. A full
match authorizes the seven-token proposal suffix plus target bonus and eight
verifier-retained rows. A mismatch follows the unchanged first-correction rule.
Distinguish full verifier-authorized `A` from terminal output clipping; the
last selected window may expose only a prefix to satisfy the 64-token bound.

Select transactions zero through seven in each category. Recompute `U`, exact
layer-qualified identity lists, source expert bytes, proposal/verification
walls, full `A`, observable emitted tokens, and all byte ledgers. For native-MTP
history:

- transaction zero binds to the final row and full history of the authenticated
  prefill target-hidden artifact;
- later transactions bind to the final retained row of the preceding verifier
  transaction;
- every history segment, target input ID, shifted MTP input ID, and anchor must
  close exactly.

Add deterministic fixtures for transaction-zero prefill binding, later-window
binding, full-match target bonus, mismatch correction, terminal clipping, and
eight-transaction fixed-64 selection. Preserve the legacy PW-0208 builder and
manifest unchanged under their original names.

## Continuation and kill gates

Publish a new, distinctly named 32-window manifest only if all four sources and
all 32 primary windows pass. Then authorize a fresh Prismwing-1 K4/cache
envelope computation from these exact `A`, `U`, identities, and routes.

Kill the regenerated-envelope branch if any category cannot supply eight valid
chronological windows, if transaction-zero history does not close against the
prefill authority, or if any semantic, byte, hidden, or safety gate fails. Do
not substitute a stale legacy window or projected bonus.

## Claims excluded

- sustained or complete endpoint TPS;
- native-MTP proposal latency, accelerated proposal execution, or general
  runtime promotion;
- K4 fidelity, construction, cache residency, common-weight cost, or achieved
  Prismwing-1 throughput;
- multimodal/full-capability promotion or any companion-hardware branch.

## Result

Unexecuted.

## Decision

Unexecuted. Commit this contract before launching the first full capture.
