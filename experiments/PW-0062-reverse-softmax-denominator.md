# PW-0062 — PyTorch-compatible softmax denominator order

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0061 comparison
  `848965d901ba9d540204446c7658985110628df5eee88469015732463018e103`
- Related records: PW-0057, PW-0061

## Hypothesis and contract

Rust's forward sequential F32 softmax denominator differs from PyTorch CPU on
one exact layer-2 row. Reverse F32 accumulation matches the oracle there and
does not alter any captured layer-0/layer-1 result.

Add the exact 19-value failing score row and PyTorch BF16 probability payload
to the correctness fixture. Reverse only BF16-attention denominator
accumulation; retain centered scores, vForce exponentials, division, BF16 cast,
and the distinct F32 scalar path. All existing fixtures must pass.

Success additionally requires a repeated PW-0061 trace with bit-exact layer-2
probabilities through final residual, exact routes, and route-weight error at
most `5e-7`. Retain all shared-host stops. This is a correctness repair and
cannot alter hosted acceptance or make a throughput claim.

## Result

The exact 19-value PW-0061 row was added to the fixture and passes with reverse
F32 denominator accumulation. All 29 Rust and 42 Python tests, strict Clippy,
and the release build pass.

PW-0061 Rust run 002 makes attention probabilities, output, projection,
post-attention state, MoE input, router logits, and every expert tensor
bit-exact. Route expert sets remain exact. The next difference is router
sigmoid: scores differ by one F32 ULP maximum, route weights by
`2.2180175779373812e-8`, routed output by `0.001953125`, and nine final BF16
values differ with maximum `0.015625`. Rust run 002 manifest hashes to
`c4e3a1d52ddfe757e11e9d266ad494f21d64d2fc9cca5e9bd5f6d40332c3a435`;
comparison 002 hashes to
`dc0aee3b534f8ccc6ea37c1e8cf47215a9c848571dc5dc400abffbcaafa88a09`.

## Decision

Promote reverse denominator accumulation as a correctness repair. It removes
the first PW-0061 mismatch without perturbing earlier real corpora. Do not
relax the final-layer gate: open the independently gated vector-sigmoid repair
for the remaining one-ULP score difference.
