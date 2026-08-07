# PW-0114 — Repair-free Metal-native distribution probe

- Status: complete
- Disposition: conditional
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract
  `4238b4d523ebcc36e0760f971b2f84664eeffe56`; implementation
  `a25236b7d5c0a65200934954665024b40b7890fb`; clean analysis
  `8fd4ac82177959acc7429b3463615a1c1ab8344c`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0095 cached source oracle
  `75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8`
- Hardware/runtime: Apple M1 shared 16 GiB host; verified internal-SSD
  checkpoint; bounded Rust/Metal incremental endpoint
- Related records: PW-0095, PW-0099, PW-0100, PW-0101, PW-0111

## Question and changed premise

PW-0100 rejected the projection-at-a-time Metal endpoint because its layer-final
and logit tensors failed source-derived component gates and its complete token
took 75.7 seconds. It nevertheless preserved the source routes and greedy token.
PW-0111 subsequently proved that a repair-free, one-barrier Metal-native layer
can reproduce the repaired control's routed and final-residual bytes on the
authenticated layer-4 row while reducing the warm transaction wall by 2.702x.
Neither result answers whether source-framework BF16-identical intermediate
states are stronger than the project's declared behavioral target.

This experiment isolates that numerical premise. Re-run the frozen PW-0100
incremental token twice: once with the existing value-derived sparse repair as
the control and once with every sparse repair disabled as the explicitly named
`metal-native-l3` candidate. Preserve the tokenizer, checkpoint bytes, retained
K/V state, attention, router, selected experts, route weights, dynamic FP8,
BF16 boundaries, residuals, LM head, and greedy sampler. The only candidate
change is whether the Metal projection result is sparsely replaced by the
source-topology correction path.

This does not reopen PW-0100's performance architecture. Projection-at-a-time
buffer installation and synchronization are a bounded vehicle for measuring
accumulated numerical behavior; they are not the proposed final runtime.

## Exactness and reporting contract

The candidate is an explicit **L3 bounded approximation** named
`metal-native-l3`. It is not target-faithful, L0, L1, or L2, even if this probe
passes. The source-derived PW-0095 logits remain the local numerical control.
The unavailable whole-model official-framework comparison and the known hosted
serving divergence remain explicit evidence limits.

Add a diagnostic-only endpoint that always keeps acceptance false and never
commits its sampled token into authoritative generation state. A completed
diagnostic may write its immutable report even when the older strict
intermediate gates fail, provided causal accounting, checkpoint integrity,
finite-value checks, cache lengths, routes, output shape, and Gate 8 pass. The
ordinary PW-0100 endpoint must retain its existing fail-closed behavior.

Each report records:

- every layer's expert IDs, route-weight error, layer-final relative L2,
  maximum absolute error, and BF16 identity against PW-0095;
- sparse-repair counts by projection, which must be nonzero for the control and
  exactly zero for `metal-native-l3`;
- final RMSNorm and full-logit relative L2, maximum absolute error, and BF16
  identity;
- source and candidate argmax, source-chosen-token logprob error, source-top-20
  token overlap, and projected Jensen-Shannon divergence over the source top 20
  plus `OTHER` using natural logarithms;
- complete wall, routed-layer wall, logical and installed bytes, dispatches,
  waits, releases, batch one, concurrency one, accepted tokens zero, `A=0`,
  per-layer `U`, cold/warm state, hardware, OS, compiler, and commit; and
- raw artifact hashes plus phase-level RSS, memory pressure, swap, release, and
  protected-service health.

Add deterministic tests for the 21-bucket projection, chosen-token logprob
error, top-20 overlap, diagnostic/non-acceptance labeling, and repair-disable
accounting before the full walk.

## Gates and bounded execution

Run exactly one clean control and one clean candidate process. These are
correctness probes, not performance comparisons; order and cache state are
recorded but no TPS promotion may use their timing. The candidate numerical
premise passes this first rung only when all of the following hold:

1. the source checkpoint, frozen fixture, cached oracle, routes, route weights,
   cache lengths, tensor shapes, and all causal/accounting ledgers validate;
2. `metal-native-l3` performs zero sparse repairs and has no hidden CPU
   projection fallback;
3. the candidate argmax equals the source argmax, source-chosen-token absolute
   logprob error is at most `0.08` nats, projected JSD is at most `0.01` nats,
   and at least 18 of the source top-20 token IDs remain in the candidate top 20;
4. the candidate is deterministic within the single completed process and all
   values used for the final distribution are finite; and
5. normative Gate 8 passes at process start, checkpoint open, Metal compile,
   every prefill and incremental layer, every routed-layer release boundary,
   LM-head completion, diagnostic emission, and final release.

Gate 8 stops below 20% system-free memory, above 8 GiB current or peak process
memory, above 4 GiB after a declared release, above 512 MiB swap growth, on any
new throttled page, or when a protected service disappears. Preserve a stopped
run as failed evidence. Do not run a third full process to rescue an unfavorable
result.

A pass authorizes a separately contracted multi-position, multi-slice L3
teacher-forced evaluation after an executable-byte mechanism exists. It does
not promote the arithmetic, build the full Metal-ready bank, rerun wide
speculation, claim hosted parity, or weaken any threshold in `TARGET.md`.

Kill the repair-free current Metal numerical premise if the candidate fails any
distribution gate. A failure does not kill all reordered arithmetic or the
PW-0045 learned routed-mixture compiler; it prevents either from citing this
specific reduction topology as already quality-qualified.

## Result

Both clean diagnostic processes completed. The sparse-repaired control applied
`[129, 170, 250]` gate/up/down repairs and decoded 1,736,704 repair weight
bytes. The `metal-native-l3` candidate applied `[0, 0, 0]` repairs and decoded
zero repair bytes. Both executed 376 experts and 1,128 projection dispatches,
installed 9,464,659,968 authenticated source bytes, remained diagnostic-only,
reported `accepted_tokens=0` and `A=0`, and did not commit their output token.

The repair-free candidate passes every predeclared distribution gate on the
frozen source-derived position:

| Metric | Control | Repair-free candidate | Candidate gate |
| --- | ---: | ---: | ---: |
| Source/candidate argmax | 13 / 13 | 13 / 13 | equal |
| Source-chosen absolute logprob error | 0.028671 nats | 0.024239 nats | <= 0.08 |
| Source-top-20 overlap | 19 | 19 | >= 18 |
| Projected JSD | 0.000679 nats | 0.000578 nats | <= 0.01 |

This does not restore component identity. The candidate first fails the frozen
layer-final gate at layer 4, reaches 3.1774% final-layer relative L2, differs
from source routes at 20 of 48 layers, and reaches 0.503707 maximum route-weight
error. It also differs from the repaired control's selected experts at layers
43, 44, and 46 and from its route weights at layers 36--47. Those divergences
are retained rather than hidden by the favorable final distribution.

The control and candidate incremental walls were 75,782.644 and 75,834.419 ms;
their 778,799.204 and 777,590.137 ms CPU prefills dominate complete process
time. These are diagnostic-only measurements of the deliberately rejected
projection-at-a-time vehicle and are not accepted TPS or a timing comparison.

Gate 8 passed both processes. Minimum system-free memory was 77%; maximum peak
RSS was 4,341,989,376 bytes for the control and 4,430,184,448 bytes for the
candidate; post-release footprints were 3,203,014,784 and 3,178,832,960 bytes;
swap growth and new throttled pages were zero; protected-service PIDs remained
stable.

Raw control evidence hashes to
`24622adf564d840880ea44163edbb3c98905d4914b2bc4153cb21151fc58281e`;
raw candidate evidence hashes to
`16aaaded5cb082e5672f1a18b132fa52665375117dadb07c0c628fcc76b3b43f`;
clean analysis hashes to
`14866caa426287a61d9ed91a441ff6937465da4e542114ba29b773726b332fa6`.
The updated throughput model hashes to
`021b8688d3cea29da310d3360ff03ad2d261112801956660ca98a566fb9b86ac`.

## Decision

Retain repair-free Metal-native arithmetic as a conditional L3 numerical
premise for a future executable-byte-reduced or resident representation. The
result supersedes the assumption that source-framework BF16-identical
intermediates or sparse topology repair are prerequisites for a useful final
distribution on this position.

Do not promote the arithmetic, build the approximately 303 GB exact bank, or
claim local/hosted near-equivalence. One source-derived text position cannot
predict multi-position, modality, long-context, capability, or hosted behavior,
and substantial accumulated route divergence remains. Do not run a broader
numerical walk in isolation; the next L3 test must pair this premise with a
representation that first changes the PW-0108 executable-byte bound, then use
held-out routed activations and the unchanged target gates.
