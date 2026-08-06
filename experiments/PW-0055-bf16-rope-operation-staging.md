# PW-0055 — BF16 text-RoPE operation staging

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract and fixture generator precede runtime change
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; pinned
  `modeling_mimo_v2.py` SHA-256
  `a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad`
- Hardware, OS, compiler, storage, memory pressure: M1 shared 16 GiB host and
  PW-0050 safety contract; PyTorch 2.13.0 is the independent fixture authority
- Related records: PW-0024, PW-0049, PW-0052, PW-0054

## Hypothesis and mechanism

PW-0054 returns final RoPE results to BF16 but still combines each rotated pair
as one F32 expression. The pinned text `apply_rotary_pos_emb` instead executes
`(q * cos) + (rotate_half(q) * sin)` on BF16 tensors. Each multiplication and
the addition therefore has a BF16 result boundary. Repeated over Q/K in all 48
layers, the omitted operation staging may materially alter attention and the
hosted distribution.

## Contract

Before runtime change, generate seeded production-head-width BF16 Q/K values
for positions 0, 1, 7, and 27 under both theta 10,000 and 10,000,000. Compute
the pinned `freqs`, duplicated cos/sin, half rotation, two multiplications, and
addition in PyTorch BF16. Preserve input and output BF16 payloads. Rust must
match every output payload bit and preserve the unrotated 128-value tail.
The eight-case fixture hashes to
`d06d3ab22488a0a17c98ca28081781075cec41558593d96bd258539e19a23ee9`.

Change only text RoPE arithmetic staging. Retain PW-0054 boundaries, dynamic
FP8 activations, positions, theta selection, partial dimension 64, Q/K layout,
weights, routes, and all acceptance limits.

Success requires exact fixture parity, every existing gate, a safe raw walk,
and material movement toward the frozen hosted distribution. Kill as the
primary mismatch if chosen-token errors do not improve materially or move
oppositely without top-k agreement. Preserve an exact source repair even if it
is independently insufficient. Do not tune staging or theta against `Hello!`.

## Baseline

Clean `6bd74ebccddb0b314338adceabccdbaf04a04282` produces `.3` on the
frozen chat prefix, with hosted-token logprob errors 12.8016 and 7.8176 nats
and 0/2 top-1 agreement.

## Result

Unexecuted.

## Decision

Unexecuted.
