# PW-0315 — Layer-4 four-expert target-native K4 bank

- Status: planned
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0314

## Question

Can the qualified layer-4 expert-64 control be expanded to the four most-used
layer-4 identities in the frozen PW-0116 route while preserving deterministic
construction, identity-local semantics, cumulative routed quality, and Gate 8?

## Hypothesis and mechanism

PW-0314 proves that the receipt-authenticated full checkpoint can construct one
new-layer `m1-native-k4-v1` expert. PW-0313 proves that qualification is
identity-local. The next smallest useful bank therefore keeps expert 64 as an
immutable control and adds experts 96, 31, and 232, the next three most-used
identities in layer 4. Their frozen placement counts are 174, 168, and 166;
expert 64 has 181.

The experiment constructs all four identities with the unchanged recipe,
requires the expert-64 candidate and packed hashes to remain equal to PW-0314,
gates each identity separately, and then substitutes all four candidates into
the same expert-major source schedule before measuring cumulative route and
layer-final error.

## Authorities

- checkpoint revision `63651580ca774f8504f676040460aed3e1244ac1`;
- installed-checkpoint receipt SHA-256
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
- PW-0116 corpus manifest SHA-256
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
- layer 4 experts `[64, 96, 31, 232]`, in descending frozen route frequency;
- receipt-bound shards `model_pp0_ep2_shard0.safetensors`,
  `model_pp0_ep3_shard0.safetensors`, `model_pp0_ep0_shard0.safetensors`, and
  `model_pp0_ep7_shard0.safetensors`;
- PW-0314 run-001 expert-64 projection candidate, packed-trellis, and manifest
  hashes as the immutable control;
- PW-0311 QTIP, calibration-atlas, recipe, seed, codebook, TLUT, and Gate 8
  authorities unchanged.

## Protocol and gates

1. Preflight the receipt, index, all 24 requested tensor mappings, and all four
   installed shard identities without rescanning receipt-proven shard payloads.
2. Reconstruct the source routed and final layer outputs bit exactly.
3. For each identity, recompute its source output at every frozen placement and
   require exact equality with the corresponding expert-major capture rows.
4. Construct gate/up/down K4 projections sequentially, independently decode
   every serialized projection, and require relative L2 at most `2e-5`.
5. Require expert 64's candidate, packed-trellis, and manifest hashes to match
   PW-0314 run 001 exactly.
6. For every identity, record complete-expert aggregate and worst-row error,
   then substitute only that identity and require route and layer-final relative
   L2 below `0.01` on overall, train, validation, and pilot holdout, with every
   maximum row below `0.05`.
7. Substitute all four candidates together and apply the same cumulative route
   and layer-final gates on every slice.
8. Run twice in fresh processes from one clean pushed commit. Require identical
   deterministic trees, projection hashes, semantic arrays, and metrics. Only
   timing and host counters may differ.
9. Record wall time, RSS, physical footprint, release footprint, memory-free
   floor, swap/throttle growth, protected services, hardware, software, commit,
   batch size one, concurrency one, and zero accepted tokens.

## Decision rule

- Qualify only identities that pass their own semantic gates in both runs.
- Authorize a larger bank only if all four identity-local gates, the cumulative
  bank gates, repeatability, the expert-64 control, and Gate 8 pass.
- If an added identity fails, preserve the passing subset and reject that
  identity; do not average it away inside cumulative metrics.
- If cumulative quality fails despite identity-local passes, stop geometric
  bank expansion and investigate error interaction before constructing more
  experts.

## Claims excluded

- source-exact or L1 weights;
- identities outside the frozen four-expert layer-4 set;
- other layers, a complete bank, or arbitrary routes;
- hosted-reference, multimodal, long-context, or capability equivalence;
- ordinary endpoint execution or accepted-token TPS;
- Prismwing-2, 34.3 TPS, or Prismwing 50 completion.
