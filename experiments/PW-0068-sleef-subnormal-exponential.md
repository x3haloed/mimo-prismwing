# PW-0068 — SLEEF subnormal exponential scaling

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean implementation
  `ea88339104221bd3d115e4cf818b7cd50805b5f1`
- Reference hashes: PyTorch
  `cf30153c4c131c8164ee7798e5022d810682e2cb`; PW-0067 failure
  `60a39dbd4437502ba930bc88bcdb51554c657fc89439111c9a8fadc672bf9470`
- Related records: PW-0063, PW-0066, PW-0067

## Hypothesis and contract

The scalar SLEEF U10 port incorrectly constructs `2^q` as one normal F32,
while pinned SLEEF `vldexp2` multiplies by `2^(q>>1)` and
`2^(q-(q>>1))`. The split keeps both factors normal and produces the required
subnormal final exponential without panic.

Freeze exact installed-PyTorch F32 exponential bits across the normal,
subnormal, underflow, and overflow boundaries, plus exact sigmoid bits for
the corresponding signed logits. Implement the source operation order with
checked exponent construction; reject non-finite input at the caller and
retain SLEEF's `< -104` zero and `> 100` infinity rules.

Pass only if the boundary fixture, router fixture, softmax fixture, all tests,
strict Clippy, and release build pass. Repeat PW-0067 from a clean commit under
the unchanged shared-host safety contract. A panic or weakened threshold
kills the repair. This is correctness-only and cannot make a TPS claim.

## Result

The source-exact two-stage scaling passes exact installed-PyTorch bits across
normal, minimum-subnormal, underflow, overflow, and saturated sigmoid cases.
All 32 Rust tests, 42 Python tests, strict Clippy, and the release build pass.

The repeated full-prefix walk completed without panic in 779.516 seconds.
Embedding and layers 0–6 are bit-exact against the frozen oracle. Layers 1–6
also have exact selected-expert order and all 1,296 route-weight F32 bits.
Layer 7 is the first failing boundary: 12 of 110,592 BF16 values differ,
relative L2 is `4.075792958108558e-6`, maximum absolute error is `0.0625`, and
equality is 99.9891%. Expert sets and order remain exact; eight route-weight
F32 values differ by at most `1.6983320045432793e-6`, above the `5e-7` gate.

The run peaked at 3,777,511,424 bytes during the LM head, ended at
2,959,760,320 bytes, retained at least 78% system-free memory, grew no swap,
observed no throttling, and retained every protected service. Rust manifest
hash:
`c12c3cc1197d4f30a43d0e5e780ebe3b9235f25bffa0242597cfa87c7f7dac7d`.
Comparison hash:
`bc380e725d358594d6f73b8ec4e2b87371017eb4e1b7af47d2071ce985363799`.

## Decision

Promote the split SLEEF exponent scaling as a correctness repair. It removes
the PW-0067 panic and advances the exact accumulated frontier through layer 6
without weakening any gate. Localize layer 7 from the exact layer-6 state
before another full walk or arithmetic change. No throughput or hosted
threshold changes.
