# PW-0016 — Real heterogeneous MoE block

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `9f727ca`; dirty selected-expert extractor,
  block benchmark, and fixture
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; selected range identities in
  external manifest SHA-256
  `62b15f1c6aabbbacb9a5c730af30b8f78ded0516666024dd3dbfefbe74549f22`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2; warm application buffers;
  peak MLX allocation 137,101,312 bytes
- Related records: PW-0003, PW-0015

## Hypothesis and mechanism

Measure one complete routed MoE block using the actual router and every actual
heterogeneous expert it selects. This adds noaux_tc routing, variable expert
position batches, normalized weights, and summation to PW-0015's complete
single-expert path.

## Contract

Source router semantics and L3 affine-INT4 experts. The deterministic
eight-position input is fixed before acquisition. Fetch only the selected
expert tensor ranges from the pinned revision, retaining source shard, offsets,
payload hashes, and the limitation that the locked remote whole-file hashes are
not yet locally verified.

The MLX router's selected expert sets must equal the source-derived PyTorch
sets. The timed block must recompute router scores and route weights, execute
each scheduled expert only for its selected positions, weight outputs, and sum
them. A fixture-specialized dispatch schedule is allowed for this experiment
but must be disclosed. Two measured runs and a fresh fixture-verified run are
required. Relative L2 error above 10% keeps the candidate conditional.

## Baseline and candidate

Input row `r`, column `c` is
`f16(sin((c + 19*r) / 17) * 0.01)`. Source noaux_tc routing computes f32 router
logits, sigmoid scores, adds the learned correction bias for choice, selects
top eight, gathers the uncorrected sigmoid scores, and normalizes them.

The resulting expert sets are committed in the fixture. Across eight positions
the union is `{3, 8, 63, 98, 141, 152, 182, 185, 208}`: nine unique experts,
or `U=9/8=1.125`. All 54 expert tensors were losslessly materialized from their
indexed shards. The timed candidate holds their affine-INT4 representations
warm.

## Isolated attribution

| Repeat | MoE block median ms | p10 ms | p90 ms |
| --- | ---: | ---: | ---: |
| 1 | 9.9059 | 8.8670 | 10.6675 |
| 2 | 9.7564 | 8.6550 | 10.6188 |

Mean median is 9.8311 ms per layer. Router plus nine installed experts occupy
124,519,424 executable bytes. The measured path processes the exact 64 routed
expert positions: seven experts receive eight positions, one receives five,
and one receives three.

Repeating this measured layer cost over 47 routed layers gives 17.31
routed-only accepted TPS at `A=8`. This assumes every layer behaves like the
fixture's layer 43 and excludes attention, dense layer zero, logits, draft,
storage misses, sampling, and rollback; it is not endpoint TPS.

## End-to-end result

This is a complete MoE component, not a complete transformer layer or model.
The router decision causally determines the specialized schedule used by the
block. The general runtime still needs dynamic schedule construction rather
than a fixture-static expert/position map.

## Correctness result

MLX route sets match the source-derived implementation for every position.
The committed fixture checks all 64 selected IDs plus representative source
and candidate output values. A fresh run passed it.

The weighted affine-INT4 MoE output differs from source FP8 by 17.02% relative
L2 with cosine 0.98547. This is synthetic-input component evidence, but it
crosses the fixed 10% caution threshold and strengthens the case against
promoting the current INT4 representation without real-activation/logit gates.

Evidence SHA-256:

- router extraction: `48913ac6e04638a9eca91ce0be77d1eba928bc7c0b463920e6fe584978c9526b`
- selected experts extraction: `62b15f1c6aabbbacb9a5c730af30b8f78ded0516666024dd3dbfefbe74549f22`
- repeat 1: `5ddf38423ef0a0b91a6299c72a03381b08e94b70a357293d08c3460fe2368ed1`
- repeat 2: `f5356a8563972496916539957cd7048d6532469ca1ab6febf9bb3b793258e237`
- fixture-verified run: `bb918cd935ef9b92ec8a4b8009bc002a23f13e266dd0e48c5a95eda060e519f1`
- committed fixture: `85ad8cdca9361bd2ae40dba737edaef21b5825cd40aae2706e3d4ffeb1f19aaf`

External evidence root: `/Volumes/Elements/mimo-prismwing/evidence/PW-0016`.

## Decision

Promote selected-range acquisition and the real MoE-block benchmark as
research substrate. Retain affine INT4 as conditional L3. The observed
`U=1.125` is promising for DFlash-8 but one synthetic layer cannot establish a
route-union distribution. Next acquire source-derived real layer inputs or,
while whole-model execution is unavailable, sample routers across layers and
inputs without treating that substitute as target behavior.
