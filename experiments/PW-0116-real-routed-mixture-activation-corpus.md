# PW-0116 — Real routed-mixture activation pilot corpus

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0112 route manifest
  `584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e`;
  PW-0115 feasibility analysis
  `41cc9b745561a09073902ba65354889d6b87e7d8716aea4db85940cbafc9c67a`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD
  checkpoint; Rust source-derived teacher-forced endpoint
- Related records: PW-0045, PW-0112, PW-0114, PW-0115; prospective E5

## Question and causal mechanism

PW-0115 proves that three all-projection shared-basis shapes can satisfy the
necessary physical envelope, but parameter counts cannot predict fidelity.
Create the smallest real target-routed activation corpus that can cheaply
falsify those shapes before training a full bank or writing a Metal kernel.

Reuse the exact PW-0112 224-position teacher-forced sequence: 87 prompt
positions plus the first 137 tokens of the frozen pinned-OpenRouter suffix. The
new execution must reproduce input-token hash
`ec757454956b42c085e5402ded86975176b987deba3d9b5a94c739fa49e459ad`
and complete 48-layer route-payload hash
`d6024840a97fd180aad17c39fef944da9a28db56bdc4de3301962b36c81923eb`.

At routed layers 4, 24, and 46, capture the causal source-derived mapping from
the layer's real MoE input and actual route to its expert outputs, weighted
mixture residual, and final residual. These are early, middle, and late pilot
layers. Layer 46 retains a following routed layer for later route-stability
tests.

## Construction and capture contract

Extend the existing single Rust route-trace authority; do not create a second
model scheduler or recompute fixture-supplied routes. Add a targeted capture
sink used only at layers `{4,24,46}`. For each target layer write immutable
F32 little-endian payloads plus shape, dtype, byte length, and SHA-256 for:

- `moe_input`: `[224, 4096]` real post-attention normalized activations;
- `expert_down`: `[1792, 4096]`, containing all `224 * 8` source expert output
  rows in the recorded expert-major schedule;
- `routed_output`: `[224, 4096]` after exact source route weighting and BF16
  boundary semantics;
- `post_attention`: `[224, 4096]`; and
- `final`: `[224, 4096]` after the routed residual addition.

Record the expert-major schedule as exact expert IDs plus global position lists,
and preserve the ordinary per-position selected IDs and route weights. The
schedule must be a bijection over all 1,792 `(position, route-slot)` placements:
every position appears exactly eight times, every captured expert-down row has
one placement, and weighted reconstruction from `expert_down`, schedule, and
route weights must reproduce `routed_output` bit-for-bit. Adding
`post_attention` and applying the source BF16 boundary must reproduce `final`
bit-for-bit. Add deterministic tiny fixtures for both reconstructions before
the full walk.

Write large payloads only under the external evidence root. The Git record
contains schema, analyzer, hashes, and results, not activations. Fail closed on
unknown layer, shape, dtype, schedule, token, route, tensor, non-finite value,
hash, existing output, or evidence schema.

## Frozen pilot partitions and coverage

Before any factor fitting, assign global positions deterministically:

- train: positions `0..111`;
- validation: positions `112..167`;
- untouched pilot holdout: positions `168..223`.

These contiguous partitions prevent later model selection from moving examples
between sets, but they remain one correlated English trace. Report per-layer
expert access counts, distinct experts, experts with one or two placements,
top-quartile frequency experts, and each partition's coverage. Never drop rare
experts or positions from the corpus or later error calculation.

The corpus is a cheap pilot only. It does not satisfy PW-0045's eventual
representative common/rare, multilingual, long-context, and native-modality
requirement. A positive pilot authorizes broader corpus acquisition; it cannot
promote a factorization or be reused as the final untouched evaluation set.

## Gates and bounded execution

Run one clean process. It passes only if:

1. checkpoint, fixture, input tokens, all 48 layer routes, and source ledgers
   reproduce PW-0112 exactly;
2. all five captures at all three target layers pass shape, finiteness, byte,
   and content-hash validation;
3. expert schedule reconstruction reproduces every routed output and final
   residual bit-for-bit;
4. the frozen partition and rare/common coverage report accounts for all 224
   positions and 1,792 placements per layer; and
5. normative Gate 8 passes at process start, checkpoint open, every layer,
   each target capture write, checkpoint release, and final service-health
   boundary.

Gate 8 stops below 20% system-free memory, above 8 GiB current or peak process
memory, above 4 GiB after declared release, above 512 MiB swap growth, on new
throttled pages, or when a protected service disappears. Report cold/warm
state, batch one, concurrency one, accepted tokens zero, `A=0`, logical and
actual bytes, wall time, hardware, OS, compiler, and commit. Timing is capture
cost, not endpoint TPS.

A passing corpus authorizes only the frozen PW-0115 rank/basis audit and matched
per-expert controls. Do not train a full bank, build a runtime artifact, or
claim representation quality from corpus construction.

## Result

Unexecuted.

## Decision

Unexecuted.
