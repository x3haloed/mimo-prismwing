# PW-0101 — Layer-4 Metal divergence localization

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes implementation and execution
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

Unexecuted.

## Decision

Unexecuted.
