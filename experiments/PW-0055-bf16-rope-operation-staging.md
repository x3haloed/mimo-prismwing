# PW-0055 — BF16 text-RoPE operation staging

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean `45cf257bd3fd11afbc9c44f3e2b93049aa047e9c`
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

All eight PyTorch RoPE cases match every BF16 payload bit and preserve the
unrotated tail. All 28 Rust tests, 35 Python tests, strict Clippy, and the
release build pass.

Raw run 001 completed safely in 290.812 seconds and retained `[122046,13]`
(`瀛.`), but its second-token routes changed and its top logits moved by as
much as 0.0625 versus PW-0054. Its report hashes to
`350e9fb298356d8a149617018809d9be593a779f993846e4ee97f70368a7ae5f`.

Frozen chat run 001 completed in 935.040 seconds and changed the local output
to `[264,13]` (` a.`). Hosted token 9707 error improved from 12.8016 to
12.5231 nats, while hosted token 0 error worsened from 7.8176 to 9.5774 nats.
Top-1 agreement remained 0/2. The chat report is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0055/chat-001/endpoint.json`,
SHA-256 `ddd72e5110112ecbc77aa446381cf243dc5d33c78f85e1b3aa5f93976584682a`.

The chat walk moved 84,281,691,904 logical source bytes and measured
85,559,504,896 process disk-read bytes. Minimum system free memory was 82%,
peak process residency was 4,327,292,928 bytes, swap did not grow, no new
throttled pages appeared, and every protected service remained alive.

## Decision

Keep exact BF16 text-RoPE staging as a correctness repair. Reject it as the
primary hosted mismatch: movement was mixed and the second position regressed
substantially. Move down the correctness ladder to an independent real
layer-0 PyTorch trace with BF16 and dynamic-FP8 semantics before another full
walk. That layer-local oracle must identify the first divergent intermediate;
do not infer a new whole-model repair from output text alone.
