# PW-0103 — Native MTP first causal proposal

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: to be recorded by the executable manifest
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

Unexecuted.

## Decision

Execute the one-layer causal prerequisite before any multi-layer MTP scheduler
or second target walk.
