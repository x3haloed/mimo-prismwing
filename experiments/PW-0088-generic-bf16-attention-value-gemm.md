# PW-0088 — Generic BF16 attention value GEMM reduction

- Status: in progress
- Disposition: unexecuted
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

Unexecuted.

## Decision

Unexecuted.
