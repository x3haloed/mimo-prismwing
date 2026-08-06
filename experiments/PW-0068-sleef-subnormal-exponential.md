# PW-0068 — SLEEF subnormal exponential scaling

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
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

Unexecuted.

## Decision

Unexecuted.
