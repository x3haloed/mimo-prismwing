# PW-0315 — Layer-4 four-expert target-native K4 bank

- Status: complete
- Disposition: conditional
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

## Execution and evidence

The constructor and independent analyzer were committed and pushed at clean
commit `9dd77d195dbc3706adeb8f94dc159b6802a7ff03`. Each identity was constructed
twice in a fresh process, sequentially on the target M1:

| Expert | Run-001 report SHA-256 | Run-002 report SHA-256 | Seconds (001 / 002) |
| ---: | --- | --- | ---: |
| 64 | `93e3ba1de9ffa4ec8d234b9317844dfa5a27ac629e8cc4db99224690988caa7d` | `95c611be59b729a420f7b261cd3dcb59dfeeae1e2066c58f82135d870e2b5866` | 500.952 / 500.007 |
| 96 | `eb033a9a60f304a4c54069484247b31801a203c70142417afe3055161b940609` | `2d07ff6bcd4fa6feafaf3d3c0d0951a0cba81b3d45b283400533e9865a30e6c2` | 499.893 / 499.766 |
| 31 | `89d2f43c62f7e45ee6502b8736095d72dbd34cd04474e6adaae63aa066b2679e` | `787d641385574bd14ed733aa60958e641b0a8e352aae53f0b675c79f42af3883` | 499.732 / 501.851 |
| 232 | `22ec36d0cbe2661b21fffd204543afbb4161bfe9cb3a3fb6300dc5656a5ff2d8` | `c257acd56f8557a667cd397a69c4fdc83956a46c18243c58f64540b7c90cad36` | 502.041 / 499.609 |

The canonical independent summary is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0315/summary.json`, SHA-256
`07b3d3793a6750a030eb5b7e12a0add1b603d48758a85e6f45b44504e404d0e8`.
Two invalid-commit preflight reports are preserved beside the accepted runs;
both failed before opening model payloads and are excluded from the canonical
run set.

## Results

Every source expert replay is bit exact. Both runs for every identity have the
same 34-file deterministic tree, stable projection fields, candidate output,
and semantic report. Expert 64 also reproduces every frozen PW-0314 candidate,
packed-trellis, and projection-manifest hash.

Identity-local results are:

| Expert | Placements | Expert relative L2 | Expert max row | Routed relative L2 | Routed max row | Final relative L2 | Final max row |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 181 | 0.006314151 | 0.018526057 | 0.000945201 | 0.005299529 | 0.000984831 | 0.002801227 |
| 96 | 174 | 0.021996005 | 0.041301663 | 0.003000297 | 0.013957299 | 0.003030594 | 0.008494531 |
| 31 | 168 | 0.005345228 | 0.023241695 | 0.000881166 | 0.003941390 | 0.000910878 | 0.002363015 |
| 232 | 166 | 0.004453180 | 0.021292464 | 0.000786904 | 0.004082866 | 0.000821485 | 0.002069160 |

Expert 96 demonstrates why the declared gate is route-weighted layer behavior,
not an invented post-hoc raw-expert threshold: its raw expert error is the
largest, but all route and final gates still pass on every frozen partition.

With all four candidates substituted together, routed relative L2 is
`0.003130533` overall, `0.003131999` train, `0.002940495` validation, and
`0.001091375` pilot holdout. The maximum routed row is `0.014743290`. Layer-final
relative L2 is `0.003160456` overall, `0.003178841` train, `0.001690742`
validation, and `0.000730365` pilot holdout; its maximum row is `0.009032566`.
All aggregate values remain below `0.01` and all rows below `0.05`.

Gate 8 passes all eight runs. Minimum free memory is 61%, maximum process
footprint is 1,734,661,312 bytes, maximum peak RSS is 1,554,481,152 bytes, and
maximum release footprint is 403,771,200 bytes. Swap growth and new throttling
are zero.

## Decision

Conditionally qualify the four-identity layer-4 `m1-native-k4-v1` bank under the
frozen PW-0116 trace and authorize a bounded bank expansion. The cumulative
result does not show harmful error interaction at width four; its overall error
is close to the largest identity-local contribution rather than their scalar
sum.

Do not generalize to the other 252 layer-4 identities, other layers, or a full
endpoint. Continue identity-local gates and cumulative prefix measurements.
These runs accept zero tokens, so they change no throughput-model constant,
runtime default, or Prismwing tier claim.
