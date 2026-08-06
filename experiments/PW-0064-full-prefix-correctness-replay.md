# PW-0064 — Full-prefix correctness replay

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean
  `cd6883ebc60a05ee0d5f835fea3e1f59b1a06795`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0063 comparison
  `8aaeeeb9c9fb5698867dc7da68932a7d3b75679d7023f4e0cfb6fdad996ba832`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch 2.13.0 CPU oracle
- Related records: PW-0050, PW-0060 through PW-0063

## Hypothesis and contract

PW-0063 removes every observed layer-2 difference without changing the frozen
oracle. One production Rust replay against PW-0060 oracle run 002 is now the
cheapest experiment that can determine whether accumulated parity advances
beyond layer 2 and localize the next causal boundary.

Execute the same frozen 27-token chat prefill through embedding and all 48
layers. Capture the same embedding, layer-final BF16 states, final norm, F32
logits, routes, and route weights. Bind the checkpoint verification, revision,
fixture, tensor index, dynamic-FP8 policy, BF16 staging, hashes, and clean
implementation commit. Do not regenerate or modify the oracle.

Require each BF16 boundary to remain within relative L2 `5e-4`, maximum
absolute error `2e-2`, and equality 99%; preserve final-state relative L2
`4e-5` and maximum absolute error `3e-6`. Expert sets must be exact and
route-weight maximum error at most `5e-7`. If all layer states clear, compare
final norm, logits, and frozen hosted-token logits before making any new
semantic change. Otherwise name the first failing layer and stop speculative
whole-model repairs until that layer is locally explained.

The shared-host safety contract is an execution gate. Sample phase-level
process footprint/peak RSS, system-free percentage, swap growth, throttled
pages, allocator pressure relief, and the start-resident protected services.
Fail closed below 20% system-free memory, above 8 GiB current or peak process
footprint, above 4 GiB post-phase footprint, above 512 MiB swap growth, on any
new throttled page, or if ChatGPT, WindowServer, nxnode, or syncthing
disappears. Release decoded matrices, mapped-file pages, and allocator
transients at matrix/expert boundaries and require the resident footprint to
return below the post-phase cap after every layer. A stop is a preserved
safety result, not permission to retry with weaker limits.

Record cache state as warm/uncontrolled after the bounded PW-0063 replay,
batch 1, concurrency 1, accepted tokens 0, complete wall time, logical and
actual bytes, hardware, and commit. This is a correctness diagnostic only: it
cannot change hosted thresholds, count as accepted TPS, or promote a
performance default.

## Result

Rust run 001 completed all 48 layers and advanced the exact accumulated
frontier. Embedding and layers 0–3 are bit-exact against the frozen oracle.
Layers 1–3 also have identical selected-expert order and all 648 route-weight
F32 bits; the comparison's approximately `7e-9` values are decimal JSON
spelling only.

Layer 4 is the first failing boundary. Its final state has relative L2
`7.339645954888441e-6`, maximum absolute error `0.0625`, and equality
99.1093%. Expert sets and their output order remain exact, while 16 of 216
route-weight F32 values differ with maximum absolute error
`5.9301374015809094e-5`, above the `5e-7` gate. Layer 5 then amplifies the
difference to maximum absolute error `2.0`; later layers are downstream and do
not identify the layer-4 substage.

The warm/uncontrolled correctness walk completed in 776.790 seconds. Before
the LM head, the recorded peak was 716,701,696 bytes and repeated cleanup
returned phase footprint as low as roughly 152 MB. The full-vocabulary LM
head peaked at 3,944,726,528 bytes and ended at 2,686,933,120 bytes, below the
8 GiB peak and 4 GiB post-phase stops. System-free memory stayed at 81%, swap
growth and new throttled pages were zero, and every protected service remained
present.

Rust manifest hash:
`0f1ed1832f2343cc501ecb6293626ff3ed46e5a8a79808b095965f677a8d7643`.
Comparison hash:
`cd057b3eb6ecb7c7075599d432595b3f3dbdd6d246c3816822437bede55d13b0`.

## Decision

Promote layers 2 and 3 into the exact accumulated frontier and localize the
next discrepancy to layer 4. Do not rerun the full prefix or alter late-layer
arithmetic. Open a bounded layer-4 substage trace from the exact layer-3 final
state under the same shared-host stops. No throughput constant or hosted
threshold changes.
