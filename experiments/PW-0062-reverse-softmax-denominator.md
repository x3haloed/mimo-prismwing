# PW-0062 — PyTorch-compatible softmax denominator order

- Status: in progress
- Disposition: unexecuted
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

Unexecuted.

## Decision

Unexecuted.
