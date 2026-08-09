# PW-0150 — Exported DFlash mask-embedding control

- Status: ready to execute
- Disposition: unexecuted
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  DFlash revision `1f58446181abcaa01030fdbde835fbd38ae9a2b1`; PW-0102 artifact
  manifest `e67b0106aa2c26a091f1fef0661a4ccc408389f2bc5d1bab9ed42e46a6e898c6`
- Hardware/runtime: Apple M1 shared 16 GiB; verified SSD base checkpoint;
  pinned DFlash draft and exported mask artifact on `/Volumes/Elements`
- Related records: PW-0009, PW-0102, PW-0103, PW-0110 through PW-0112

## Question and changed premise

PW-0102 executed the DFlash draft against the pinned base target and rejected
its first width-eight block after the first suffix token mismatched. That run
used row 151675 of the pinned base embedding for all seven masked draft
positions. The DFlash artifact also ships a separately authenticated
`mask_embedding.pt`, but PW-0102 explicitly did not use it.

The two values are not interchangeable. The base row has F32 norm
`0.0000207325`; the exported BF16 vector has F32 norm `1.4794452`, relative L2
distance `71358.67`, and cosine similarity `-0.0310`. Repetition in PW-0102's
proposal is therefore consistent with near-zero mask noise, not decisive
evidence about the trained mask embodiment.

Run the cheapest one-variable control: preserve the pinned base anchor token,
target hidden states, draft weights, full-head SGLang RoPE adaptation, base LM
head, positions, caches, and greedy sampling, but replace only the seven masked
position embeddings with the shipped authenticated mask vector.

## Exactness and red-line check

This remains L2 target-faithful speculation. The exported mask is draft-only;
no target weight, route, arithmetic, token, or acceptance rule changes. No
draft token can be committed without exact target verification. The experiment
does not substitute the incompatible bundled DFlash target.

## Contract and gates

1. Authenticate the base checkpoint receipt, PW-0091 frozen states, DFlash
   artifact manifest, draft payload, mask artifact, and pinned SGLang source.
2. Add a deterministic fixture proving that the base anchor remains unchanged,
   the exported vector occupies exactly the seven mask positions, and malformed
   mask tensors fail closed.
3. Run one cold draft proposal and record complete output hashes, proposal IDs,
   physical I/O, timing, and normative Gate-8 snapshots. This is diagnostic,
   not accepted TPS.
4. Compare the first suffix proposal with PW-0102's already-verified pinned-base
   next token `13`. If it differs, reject this correction without another full
   target walk because greedy acceptance is necessarily zero beyond the anchor.
5. If it equals `13`, authorize one new source-target width-eight walk; later
   target posteriors cannot be reused because causal inputs changed. Promote
   only if measured `A/U` improves and then passes the physical threshold of the
   named companion/store branch.

Gate 8 stops below 20% free memory, above 8 GiB process current/peak memory,
above 512 MiB swap growth, on a new throttled page or protected-service loss,
or above 4 GiB after declared release. Preserve any failed run rather than
silently retrying it.

## Result

The first invocation stopped before input authentication because its command
named a nonexistent receipt path. Preserve
`/Users/chad/Models/mimo-prismwing/evidence/PW-0150/draft-001/failure.json`,
SHA-256 `7c9f7ca03b8c6ea98b61f071e9b5f0b7713ed05a33aeadd4af5b7933912bb6bc`.
Its sole safety snapshot retained 67% free memory, used no new swap or
throttled pages, and retained every protected service. No checkpoint or draft
payload was opened. The authenticated receipt is the already-established
PW-0049 artifact at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0049/checkpoint-verification.json`,
whose SHA-256 is the required
`9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`.
Authorize `draft-002` with that corrected path; this operational amendment
changes no model, mask, proposal, or gate semantics.

## Decision

Pending execution. This control can reverse only PW-0102's judgment about the
exported-mask embodiment. It cannot establish broad acceptance, a wide proposer,
companion feasibility, endpoint TPS, or Prismwing completion by itself.
