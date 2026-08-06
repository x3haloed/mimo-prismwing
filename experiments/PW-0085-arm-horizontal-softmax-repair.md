# PW-0085 — ARM horizontal softmax reduction repair

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0084 oracle
  `94c7411a5879f4ade7a700a4309d3a2b48354cc67409701e39003391cadde736`;
  PW-0084 comparison
  `e1309ffe9bec70866181ebd6333212d9964ef82aedbd6fe604473019ec3e1a8f`;
  pinned PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`
- Hardware/runtime: Apple M1 shared 16 GiB host, PyTorch 2.13.0 CPU oracle,
  production Rust trace
- Related records: PW-0057, PW-0065, PW-0066, PW-0083, PW-0084

## Hypothesis and contract

PW-0084's exact 23-value centered-score row discriminates the ARM
`vaddvq_f32` horizontal reduction that PW-0066's corpus did not. The current
Rust model reduces adjacent pairs `(lane0 + lane1) + (lane2 + lane3)` and
produces denominator bits `0x40c255d5`; PyTorch's F32 probabilities imply
`0x40c255d4`. The ARM intrinsic's pairwise low/high reduction
`(lane0 + lane2) + (lane1 + lane3)` produces the implied denominator.

Freeze the hash-bound PW-0084 row with PyTorch exponential, F32 probability,
and BF16 probability payloads. The fixture must independently prove whether
exponential evaluation differs, and must distinguish the two horizontal sums.
Change only the four-lane horizontal reduction if the fixture confirms the
hypothesis. Preserve every prior softmax fixture and the complete test suite.

Then replay production layer 29 from exact layer 28 against the frozen oracle.
All 21 captures, selected experts/order, and route weights must meet their
existing gates. Retain normative Gate 8, batch 1, concurrency 1, accepted
tokens 0, buffer release, allocator relief, and complete wall time. This is a
correctness experiment and cannot count as TPS or alter any threshold.

## Result

Unexecuted.

## Decision

Unexecuted.
