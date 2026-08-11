# PW-0211 — Native MTP proposal-latency lower milestone

- Status: executing
- Disposition: pending
- Date: 2026-08-11
- Execution mode: L2 target-distribution-preserving draft under exact ordinary verification
- Related records: PW-0103, PW-0205, PW-0206, PW-0207, PW-0208

## Hypothesis and mechanism

PW-0208 proves native MTP cannot deliver its predeclared 2x accepted-token per
unique-expert-byte gain, even with perfect proposals. It does not bound proposal
latency. The current corrected q=8 endpoint spends roughly 137--157 seconds per
window on seven same-model one-row proposal steps and roughly 32--41 seconds on
one verifier pass.

Hypothesis: the pinned three dense native-MTP layers can replace those seven
same-model proposal steps cheaply enough that exact verifier-authorized output
improves complete transaction TPS by any repeatable positive amount. This is a
lower milestone, not PW-0208 promotion and not a 50-TPS claim.

## Contract

Authenticate the exact MiMo MTP payload and pinned SGLang source revision. For
each proposal, reconstruct the complete target layer-47 history from the
hash-bound prefill and retained verifier segments. MiMoV2 layer zero consumes
tokens shifted by one; layers one and two reuse the same target hidden history
while rotating input IDs and appending the preceding draft. Begin with q=4,
the trained three-layer native chain. Any q=8 cycling or tree schedule is a
separately named modified scheduler.

Record proposal tokens, per-layer and complete proposal wall, target posterior,
accepted tokens, rollback, exact target routes/bytes, resident state, hardware,
and commit. CPU reference execution is correctness and feasibility evidence,
not an accepted TPS result. Runtime promotion requires ordinary verifier-only
commit and exact observable tokens.

## Cheap falsifier and gates

First reproduce PW-0206's corrected first native proposal with a last-row
implementation over complete history. Then run one chronological window per
PW-0208 category. Continue to all 32 only if the reference semantics are exact
and at least one window shows acceptance worth accelerating.

Build a measured candidate model using native proposal wall plus a real q=4
verifier wall under the same residency state. Preserve any repeatable positive
complete-path TPS gain. Kill only when an optimistic latency bound, measured
acceptance, or implemented complete path shows that no tested schedule can beat
the corrected Jacobi control; missing 2x or 50 TPS alone does not kill this
lower-milestone branch.

## Decision

Executing; the correctness/pilot gate passes and a real q=4 verifier timing is
authorized. PW-0208's complete-history manifest SHA-256 is
`a9bb6bd26bf048a2144133cc0a96023a8af112eae58122b666915149f2993a7b`.
The four prefill source reports hash to `11a02fd9d653c6351ed22d03f7d39efb80ee8d6009fc9a3d22d41fd2f42d1ddb`,
`a75aab62fa434f73d8f0053919fc9c3eab68c71e96a690cfed6f8871306b35ae`,
`b8c68eac9834c24ea09ffa65e7f3f5ef2ef5c015209c862419f4471480e846d2`,
and `385425155ab48a965169d860ff56c09e8967325e536b72dfd3b5e8e164c83773`.

Clean commit `ba09e9b2a02285a7c94eef288d00f9870558b6e2` adds a
complete-history last-row reference. It preserves the full history row count
through the FP8 MLP GEMMs because the readable authority's reduction topology
is row-count dependent. The known PW-0206 validation emits token `0` and all
152,576 logits are bit-identical to the earlier full-row oracle. Its report
hashes to
`395e61eb628c1b9ec3c892d285f5b3d0bc0749b6e5e7bc782cb5671dd299645f`.

The first chronological window in each category then produces:

| Category | Native q4 block | Target posterior prefix | Native `A` | Control `A` | CPU reference complete ms | Control proposal ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ordinary | `[11941,7949,7324,8628]` | `[7949,7324,8628,2041]` | 4 | 7 | 10,444.325 | 143,313.654 |
| code | `[3084,2268,1097,3286]` | `[2268,1097,3286,3255]` | 4 | 7 | 8,922.493 | 142,806.753 |
| multilingual | `[102533,101920,99607,101079]` | `[33108,81812,101920,99607]` | 1 | 7 | 8,699.134 | 138,834.677 |
| rare-route | `[549,17588,11,308]` | `[17588,11,308,488]` | 4 | 7 | 9,015.061 | 157,260.542 |

The four report hashes are respectively
`511aa5b4d1353074f193e4c5488d001c3ce1cb0d3b9342f91454a7a9c35a33bc`,
`3573e1e25472d19523c7ed0e1b617dc0032136aec5cd300ef7c6fdeb86f247fa`,
`124a8064963d79eadbb303ed2baf9811da6357978557a08abb88cb6b2fb93f2a`,
and `a82be5904a6ff67d9a86eba164b3a5153470025e22c1b73d3399b7f5c2503bce`.
Three of four pilots perfectly accept the trained three-token draft; the
multilingual pilot accepts only the anchor. This is enough to pass the frozen
continuation gate, but it is not a corpus mean and the CPU wall is diagnostic,
not endpoint TPS.

Do not run all 32 proposal references yet. Replay the exact preceding control
transaction where required, then measure one ordinary q=4 target verifier with
the frozen native block and the same checkpoint, cache, cold-state, Metal, and
residency accounting. Only that real verifier wall can estimate a complete
candidate transaction. If the enclosing model remains positive, integrate the
native proposer and broaden acceptance measurement; otherwise preserve the
three positive category results and close only the tested runtime schedule.
