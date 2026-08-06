# PW-0103 — Native MTP first causal proposal

- Status: complete
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: `8a99f2f2e3f171e16e4c54893ecb73392cc1e200`,
  clean executable
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0091 manifest
  `87466b59480a5a5b4256c490f1dfe670fe09f28d21d169085ab13bb1b4b7ab59`;
  pinned SGLang MTP semantics `2fc557254b3aaf539e80266e52a6d1e1f8da9980`
- Hardware/runtime: Apple M1 shared 16 GiB host; verified SSD base and MTP
  checkpoint payloads
- Related records: PW-0026 through PW-0030, PW-0091, PW-0095, PW-0102

## Hypothesis and causal mechanism

PW-0102 rejects a DFlash draft trained with a different bundled target: its
first suffix proposal 1773 misses the base target's token 13. The pinned base
checkpoint itself ships three trained MTP layers. Before constructing another
width-four target verifier, test the smallest causal prerequisite: pair the
PW-0091 target's final unnormalized hidden sequence with the one-token-shifted
base embeddings ending in target anchor 264, run native MTP layer zero under
the pinned SGLang transition, and ask whether its first greedy proposal is the
PW-0095 target token 13.

This experiment does not estimate acceptance from tensor names or from the old
synthetic MTP fixture. It executes the learned `enorm`, `hnorm`, `eh_proj`,
dense SWA decoder, `final_layernorm`, and shared base LM head as one real causal
path.

## Exactness and safety contract

This is target-faithful greedy L2 draft work: the MTP approximation may differ,
but no draft token is committed without a later exact target verifier. Follow
the pinned SGLang MiMo-V2 MTP source, including the shifted token/target-hidden
pairing, 64 rotary dimensions inside 192-wide Q/K heads, 128-wide values scaled
by 0.707, sliding-window attention with learned sinks, BF16 boundaries, and
base embedding/LM-head reuse. Fail closed on source-lock, MTP-file hash/tensor
inventory, PW-0091 hidden/logit, shape, cache, or non-finite mismatches.

Use `tools/host_safety.py` at input authentication, each materialized stage,
attention, every MLP projection, LM head, evidence publication, and release.
The 8 GiB in-flight and 4 GiB post-release ceilings, 20% free-memory floor,
512 MiB swap-growth limit, zero new throttled pages, and protected-service
health are normative stops. Preserve a failed artifact.

## Gates

First validate the complete 48-tensor MTP inventory and source SHA-256, then
execute only MTP layer zero. Capture widened-BF16 states after fusion,
attention, MLP, final norm, and the full F32 logits. Record token rank/margin,
cold/warm state, logical/physical bytes, wall time, and all safety boundaries.

- Pass the causal prerequisite only if the greedy proposal is target token 13.
- If it passes, run one clean determinism repeat, then open a separate chained
  three-layer proposal/width-four `A/U` experiment.
- If token 13 is not top one, reject native MTP for this trace without another
  full target walk. Token 13's rank may inform a separately bounded tree draft,
  but is not accepted-token evidence.

No performance default or accepted TPS can be promoted here. A passing first
proposal proves only that the trained draft is aligned at one position.

## Result

The first attempt exposed a standalone harness import error before attention
arithmetic. It is preserved at
`/Volumes/Elements/mimo-prismwing/evidence/PW-0103/mtp-001/failure.json`, SHA-256
`493ad376f023e3bb9b5b82dc5e2f2ec35767d1f229e3e314aeef759239fd8f2a`.
The next commit fixed the import and broadened exception capture; this attempt
is tooling evidence only.

The clean `mtp-002` execution completed and rejected the causal prerequisite.
Its immutable manifest hashes to
`65404539dc1b0f0e5b8cf0a0962b1b65fcd5e5fdcfe15ae2f1fd5ebdd49992a7`.
MTP layer zero proposes token 100730, while the independently established base
target requires token 13. Token 13 ranks 175th in the complete 152,576-token
MTP logit vector, with logit 7.84375 versus the top logit 12.0625. The full
logit capture hashes to
`ecc64fe2775a708554abde06688cd1b1684e0801ff98f2831bbbe9f3e3a4bc0b`.
This is not a near-tie or a top-k-one scheduling artifact.

The source boundary is resolved rather than inferred. Pinned SGLang requests
the target hidden state before final norm, rotates each prefill input sequence
left, appends the target next token, and feeds that pair into the selected MTP
layer. PW-0103 does exactly that with PW-0091 layer-47 states and
`prompt[1:] + [264]`. The MTP payload's complete SHA-256 and all 48 tensor
names, dtypes, and shapes pass before execution.

The warm-cache run completed in 3,195.041 ms: 85.284 ms fusion, 333.716 ms
attention, 794.424 ms MLP, and 581.115 ms LM head, with safety/evidence overhead
outside those components. It read 122,880 process-disk bytes and moved no
unrecorded model representation. It retained at least 76% system-free memory,
peaked at 4,157,685,760 bytes RSS and 184,372,288 bytes physical footprint,
grew neither swap nor throttled pages, retained every protected service, and
released to 154,553,152 bytes physical footprint.

## Decision

Reject native MTP for this pinned trace without a chained three-layer scheduler
or another full target walk. Its first required draft token is wrong and the
correct token lies outside a practical top-k-one proposal. Preserve the MTP
oracle and source lock; a future source-runtime parity result that changes this
first logit vector would constitute new semantic evidence and may open a new
experiment, but downstream speculation cannot repair a failed first proposal.
