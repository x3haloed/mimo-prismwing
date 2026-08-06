# PW-0085 — ARM horizontal softmax reduction repair

- Status: complete
- Disposition: correctness-repair
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

The hash-bound 23-value fixture proves every SLEEF exponential bit already
matched PyTorch. It discriminates only the horizontal reduction: adjacent
pairs produce denominator `0x40c255d5`, while ARM low/high pairs produce
`0x40c255d4` and all 23 PyTorch F32 and BF16 probability payloads exactly.
The fixture hash is
`3fdd3c5f49d5922329621e8eb1df03dff08b4124f66cad9735cb6ee172be6a71`.

Changing the final four-lane reduction to `(lane0 + lane2) + (lane1 + lane3)`
passes all 37 Rust tests, 42 Python tests, strict Clippy, deterministic fixture
regeneration, and every prior softmax fixture. The repaired production
layer-29 replay makes all 21 captures bit-exact, preserves exact expert
selection/order, and holds route-weight serialization error to
`1.9330596900957175e-8`.

The Rust replay completed in 465.415 seconds, peaked at 746,192,896 bytes RSS,
returned to a 144,471,296-byte footprint, retained at least 83% free memory,
grew no swap, observed no throttling, and kept every protected service
healthy. Evidence hashes:

- Rust manifest:
  `e850fbc09014f89dcfe4dbccd0bb9ced40b5f03511770dbe3870aca25f2871ce`
- Comparison:
  `716fa337cde3e90de10342f46afafd802d5b78f5b73a2e82e7c90ef9462da5b3`

## Decision

Promote the ARM low/high horizontal reduction as a correctness repair. It
removes PW-0084's first actual difference and restores exact layer-29 state
without weakening any gate. The accumulated exact frontier is ready for a
full-prefix replay beyond layer 29. No throughput or hosted threshold changes.
