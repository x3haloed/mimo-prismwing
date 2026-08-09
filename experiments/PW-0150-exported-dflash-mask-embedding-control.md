# PW-0150 — Exported DFlash mask-embedding control

- Status: completed
- Disposition: rejected
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Commit and dirty state: `draft-003` runtime `4d8c34d`, clean; analysis
  `db4a283b536586993796317368130107a5f6a94c`, clean
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

`draft-002` completed draft and LM-head computation but stopped at manifest
serialization because Transformers 5.14 returned `mismatched_keys` as an empty
Python set, whereas the existing evidence code assumed the earlier list form.
Its explicit failure record hashes to
`89974d17b07487963d7d3b40940f7747369e59bd25c58f536ca3933241f8981e`.
Quarantine its two partial captures; they hash to
`8cabe5649d6aecaf65b66f86cd9852f0aa0779c6c4609c261b7da1c507c3b8f4`
and `9a97853dee932e215c801075bbb2bd0abc95ebb276aeea7fd14c226c176492f7`
but have no valid manifest and are not results. Normalize only the loader's
diagnostic container at the JSON boundary, add a regression fixture for list,
tuple, and set forms, and authorize `draft-003`. The model outputs are not read
from or compared with the quarantined files.

The authorized post-failure `draft-003` run passed. Its immutable raw manifest
is `/Users/chad/Models/mimo-prismwing/evidence/PW-0150/draft-003/manifest.json`
and hashes to
`0582f905d8d6531e0c7d4e9a50def819a6d337a62c5e9b0cac351caa9435f882`.
The shipped mask changes the complete proposal from PW-0102's
`[264, 1773, 102092, 102092, 102092, 1773, 1773, 1773]` to
`[264, 11, 11, 11, 11, 11, 11, 11]`. Its final-hidden and complete-logit
captures hash to `8cabe564...b8f4` and `9a97853d...2f7` respectively.

The changed proposal still fails at the first required suffix. The already
authenticated target token is `13`, while the exported-mask draft chooses
token `11` at logit `8.75`. Token `13` is draft rank four at logit `6.53125`,
a `2.21875` gap. Therefore matching suffix length is zero and formal `A=1`
counts only the target anchor. Every width-eight top-eight route union has
`U>=1`, so even without another target walk this block has the decisive bound
`A/U<=1` and cannot pass the strict routed-byte leverage gate `A/U>1`.

The post-failure run was cache-influenced and is not a cold latency claim. It
read 1,095,921,664 physical bytes, spent 27,582.396 ms in the draft and
1,002.224 ms in the base LM head, and completed in 29,918.319 ms. Gate 8 passed
all 15 snapshots with 62% minimum free memory, 4,110,647,296-byte peak RSS,
299,475,008-byte maximum physical footprint, 289,049,088-byte final footprint,
zero swap growth or new throttled pages, and stable services.

Independent analysis authenticates the raw proposal, both captures, and the
PW-0102 target posterior. Its manifest is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0150/analysis-001/manifest.json`
and hashes to
`72051c021ae1d93989508b0423ab1b0811072c24799b8e986d4543b4a513f04e`.

## Decision

Reject the supplied DFlash-8 draft with its exported mask against this frozen
pinned-base trace. The shipped mask is the correct causal variable to retain in
any future draft audit, and PW-0102's near-zero-mask proposal should not be
treated as representative of that embodiment. Nevertheless, the corrected
draft still misses the first token and cannot provide routed-byte leverage, so
do not spend another full target walk or optimize this proposal. A base-trained
proposer or materially wider route-coherent lattice remains a distinct branch.
No endpoint TPS or throughput-model constant changes.
