# PW-0088 — Generic BF16 attention value GEMM reduction

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0087 oracle
  `dcf92f0c37e825766984f524b2338701adf28dd528ffafd374d59e6f20673fc1`;
  PW-0087 comparison
  `7f455767cbe8b065a87185b6e75a09a51a80cd702ccfb786b3086f5451a5f5de`;
  pinned PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`
- Hardware/runtime: Apple M1 shared 16 GiB host, PyTorch 2.13.0 CPU oracle,
  production Rust trace
- Related records: PW-0075, PW-0076, PW-0086, PW-0087

## Hypothesis and contract

PW-0087's exact width-25 pair discriminates the generic BF16 GEMM reduction
used by PyTorch's probability-vector-by-value-matrix operation from the
specialized contiguous BF16 dot used by Rust. Generic four-part reduction
produces raw F32 `0x3dc37fff` and oracle BF16 `0x3dc3`; the specialized path
produces tie `0x3dc38000` and BF16 `0x3dc4`.

Freeze the hash-bound probability/value pair with the actual PyTorch matrix
result and both source-modeled raw reductions. Preserve PW-0076, explicitly
showing that its pair does not discriminate generic from specialized
topologies. Change only attention value-by-matrix accumulation to generic
four-part reduction if both fixtures and all existing semantics pass.

Replay production layer 34 from exact layer 33 against the frozen oracle. All
21 captures, selected experts/order, and route weights must meet their
existing gates. Retain normative Gate 8, batch 1, concurrency 1, accepted
tokens 0, buffer release, allocator relief, and complete wall time. This is a
correctness experiment and cannot count as TPS or alter any threshold.

## Result

The hash-bound layer-34 fixture distinguishes the real operator paths. Generic
four-part GEMM produces raw F32 `0x3dc37fff` and the PyTorch matrix result
BF16 `0x3dc3`; the specialized contiguous dot produces `0x3dc38000` and BF16
`0x3dc4`. Its evidence hash is
`ae1a917b498362cf54d9bb6ab44746a722cac235d03393320716ca92e31e06a9`.

PW-0076 is preserved and now explicitly proves both generic and specialized
topologies produce its raw F32 `0xbcaa8002`, so it could not select between
them. Using generic four-part reduction only for attention value-by-matrix
accumulation passes all 37 Rust tests, 42 Python tests, strict Clippy, and
deterministic fixture regeneration.

The repaired production layer-34 replay makes all 21 captures bit-exact,
preserves exact expert selection/order, and holds route-weight serialization
error to `2.8776550253795108e-8`. It completed in 542.954 seconds, peaked at
737,640,448 bytes RSS, returned to a 154,783,936-byte footprint, retained 83%
free memory, reduced swap use, observed no throttling, and kept every protected
service healthy. Evidence hashes:

- Rust manifest:
  `e965ccf091b60a0c794ccc524a7e4bfb63097ba4d3d4208374677c08bfac0bf1`
- Comparison:
  `967a7f9d0ee0c0b004c8b1b365b68cd1ff2c4cca2c280d93318de4950d1274aa`

## Decision

Promote generic four-part BF16 GEMM reduction for attention value-by-matrix
accumulation. It removes PW-0087's first actual difference and restores exact
layer-34 state without weakening any gate. Preserve specialized contiguous
dots for attention scores. The accumulated frontier is ready for one frozen
full-prefix replay; no throughput or hosted threshold changes.
