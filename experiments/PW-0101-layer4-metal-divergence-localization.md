# PW-0101 — Layer-4 Metal divergence localization

- Status: complete
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: oracle `347463c`; Metal diagnostic `1635e2c`;
  counterfactual `429437f`; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0095 oracle manifest
  `75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8`;
  PW-0100 failure
  `7e76c0bcabb445ded01f547ce56f096f2a6c9474a1fccee74761293cfd29df74`
- Hardware/runtime: Apple M1 shared 16 GiB; partial independent PyTorch cached
  oracle plus generic bounded Rust/Metal routed-row diagnostic
- Related records: PW-0095, PW-0098, PW-0099, PW-0100

## Hypothesis and mechanism

PW-0100 first fails at layer 4 despite exact expert IDs and route weights. The
fixed four-F32-ULP PW-0099 predicate likely misses another Metal reduction near
a BF16 midpoint; if that value changes a subsequent dynamic-FP8 group maximum,
one local rounding decision can again amplify into the observed layer-state
failure. Localize this without another 48-layer walk.

Run the independent PyTorch cache authority only through prefill layers 0--4
and incremental layers 0--4. Capture layer 4's post-attention normalized MoE
input, unsorted routes and weights, every selected expert's gate, up, SwiGLU,
and down BF16 boundaries, routed output, and final residual. The candidate
must consume that real MoE input, independently recompute routing from the
verified checkpoint, and execute the existing bounded Metal path directly from
checkpoint tensor views while preserving pre-round values and repair decisions.

## Gates

Fail closed on checkpoint, verification, input, route, tensor, dtype, shape,
scale, oracle schema/hash, non-finite, output, commit, or create-new mismatch.
Add a layer-generic fixture before execution; no layer, expert, row, oracle
value, or expected hash may enter the repair predicate.

Routes must match exactly and weights within `3e-8`. Report BF16 equality,
maximum error, relative L2, pre-round bits, midpoint distance, dynamic-FP8
group identity, and sparse-repair selection at gate, up, SwiGLU, down, routed
output, and final residual boundaries. The diagnostic passes only if it names
the first causal divergence and demonstrates whether correcting that value
removes its downstream fan-out. A merely correlated mismatch is inconclusive.

The partial oracle must finish in at most 180 seconds; the candidate is batch
one, concurrency one, accepted tokens zero, `A=0`, and `U=8`. Timing is
diagnostic only. Enforce Gate 8 at checkpoint open, every partial layer, Metal
compile, every expert release, routed output, and final release. Stop below 20%
free memory, above 8 GiB current/peak or 4 GiB post-release, above 512 MiB swap
growth, on any throttled page, or on protected-service loss.

A localized missed midpoint authorizes a separately contracted, value-derived
repair-bound experiment with discovery/holdout separation. It does not
authorize widening the existing threshold, rerunning PW-0100, changing
correctness gates, or promoting the Metal endpoint. If the first divergence is
not a missed midpoint, kill that explanation and follow the measured boundary.

## Result

The bounded independent PyTorch oracle completed prefill and cached incremental
execution through layer 4 in 51,582.555 ms, well inside the 180-second gate.
It froze five layer caches plus every layer-4 expert boundary under
`/Users/chad/Models/mimo-prismwing/evidence/PW-0101/reference-001`; its manifest
hashes to
`9c96d85e45832abdccd3be2325db993749579a904469d1862c8f3437cafab86d`.
Routes are `[232,31,64,96,9,88,245,130]`. The oracle retained 79% free memory,
peaked at 1,316,192,256 bytes, and recorded no swap growth, throttling, or
protected-service loss.

The generic verified-checkpoint Metal diagnostic recomputed those routes and
weights exactly from the frozen source MoE input. It reproduced PW-0100's
failure without upstream candidate drift: routed output has `0.00172562`
relative L2, `1.0` maximum error, and 96.8994% BF16 identity; adding the exact
post-attention residual yields exactly PW-0100's layer-final `0.00163510`,
`1.0`, and 97.8760%. The 1,892.291 ms diagnostic includes full protected-host
checks after each expert and is not a performance result. It installed
201,375,744 source bytes, dispatched and released 24 projection buffers,
selected six gate, four up, and three down repairs, retained 79% free memory,
peaked at 205,783,040 bytes, returned to 121,376,768 bytes, and caused no swap
growth or throttling. Its report hashes to
`b2021bb4d37383a62693565da7f39a0e313a721419a49fb3b64881bfd91893bf`.

The decisive mismatch is expert 245 gate row 1798. Its Metal pre-round value is
exactly BF16 midpoint `0x40808000`, so the four-ULP predicate correctly selects
the row. Nevertheless the sparse repair returns `0x40800000` while the full
source projection returns `0x40810000`. Sparse correction uses a one-row
Accelerate SGEMM; the authoritative full 2,048-row projection uses a different
reduction topology. The defect is therefore correction shape, not uncertainty
selection or a threshold miss.

The source-exact counterfactual changes only that gate value. The wrong neighbor
creates one SwiGLU mismatch and then 233/4,096 down-output mismatches, 94.3115%
BF16 identity, `4.0` maximum error, and `0.00127801` relative L2. Restoring the
single source neighbor makes all 4,096 down values bit-exact. This closes the
causal fan-out rather than merely correlating it. The counterfactual hashes to
`1489517ceb9e704279a3c6f908bf4e38c0e1c8ef33515ccbaca4d3231b676781`
and also passes its safety checks.
The updated throughput model hashes to
`702f4c0b39c399e86b06810a0629d5f307d4e87b7fe7a04edfa6435d0990fc35`.

## Decision

Reject the hypothesis that PW-0100 needs a wider midpoint threshold. The
threshold found the decisive row; the sparse one-row reduction was not a
source-authoritative correction. Do not widen the four-ULP predicate and do
not rerun the complete token.

Any subsequent repair must first reproduce the full-projection reduction
topology for selected rows or replace it with another independently proven
source-authoritative boundary decision. It must retain value-derived selection
and repeat discovery/holdout gates before endpoint use. Executing all 2,048
rows merely to repair one is a correctness control, not yet a physically fit
solution. Independently, PW-0100's 75.7-second embodiment failure remains and
cannot be repaired by numerical work alone.
