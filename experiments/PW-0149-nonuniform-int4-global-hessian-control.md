# PW-0149 — Nonuniform INT4 global-Hessian three-expert control

- Status: completed
- Disposition: rejected
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0138 raw
  `37fa27ce90d0dc46b4b9308ed708c99405eb7ad3d924b859489716b9771bde49`;
  PW-0148 raw
  `48d1c28cc589e55002ce5a4b836d62ef172d3ed77106c100b2ad49d708fd1257`
- Hardware/runtime: Apple M1 shared 16 GiB; verified internal-SSD checkpoint;
  deterministic NumPy one-dimensional Lloyd codebooks, global-Hessian
  assignment oracle, and dense-F16 execution oracle
- Exactness: explicitly modified L3 nonuniform four-bit weight quantization;
  source routes, targets, and validation partitions unchanged
- Related records: PW-0129, PW-0137 through PW-0142, PW-0147, PW-0148

## Question and changed assumption

PW-0147 and PW-0148 show diminishing gains from adding levels to one affine
min/max grid. Test whether the shared failure is level geometry rather than raw
code count: keep 16 codes, but let every 128-weight row-group use 16
deterministic nonuniform F16 centroids.

Initialize centroids from the midpoints of 16 equal-population sorted bins,
then run exactly eight deterministic one-dimensional Lloyd iterations. Empty
clusters retain their prior centroid; ties choose the lowest code. Sort and
stage final centroids through F16 before deriving round-to-nearest codes. No
seed, initialization, iteration count, or validation-visible search is allowed.

Apply PW-0138's unchanged full-Hessian assignment with 0.1% damping,
activation order, original group lookup, and 128-column blocked propagation.
At every ordered column, choose the nearest fixed centroid for that row and
original source group. The runtime form is a four-bit code plus an F16
codebook lookup; dense reconstructed F16 matrices are an oracle only.

## Physical ledger

The three code matrices contain 25,165,824 weights and require 12,582,912
packed bytes. Their 196,608 row-groups each store 16 F16 centroids, requiring
6,291,456 bytes. Total payload is 18,874,368 bytes/expert (`0.749817` of
source), or 227,096,395,776 bytes (211.5 GiB) for all 47×256 routed experts
before container padding. This is smaller than PW-0148 six-bit affine and
leaves 44.5 GiB of a prospective 256 GiB companion for other state. Codebook
lookup changes the kernel but adds no matrix MACs. This is eligibility, not a
bank, hardware, or TPS claim.

## Frozen control and partitions

Use layer 4/expert 96, layer 24/expert 22, and layer 46/expert 28. Calibrate
only routed positions below 112, score `112..167`, and keep `168..223` sealed.
Reproduce source outputs and the exact PW-0138 four-bit controls. Bind the
immutable PW-0148 six-bit validation results. Add fixtures for deterministic
centroids and ties, empty clusters, F16 staging, reconstruction, code domain,
byte arithmetic, cross-block propagation, partition isolation, and corrupt
authority rejection.

## Continuation gate

Authorize a separately frozen all-validation-expert nonuniform audit only if:

1. every expert reaches validation relative L2 at most 2%;
2. every maximum validation row is at most 5%;
3. every candidate improves train output over its nonuniform round-to-nearest
   control;
4. every candidate improves its immutable PW-0148 six-bit validation result;
5. four-bit controls reproduce, all codes remain in `[0,15]`, source and
   partitions remain authoritative, and holdout stays sealed; and
6. packed bytes remain exactly 18,874,368 per expert (`<=75%` of source) with
   zero additional runtime matrix MACs.

Failure rejects this per-row-group 16-centroid/global-Hessian form on the three
representative experts. A pass authorizes only an all-validation-expert audit,
not holdout, bank construction, kernel work, hardware purchase, accumulated
model, or endpoint.

Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.
Apply normative Gate 8 before and after every projection and expert release.

## Result

All three candidates fail. Validation relative L2 is `0.033762` at layer
4/expert 96, `0.060601` at layer 24/expert 22, and `0.060640` at layer
46/expert 28. Maximum-row error is `0.068806`, `0.081569`, and `0.077481`.
Every result is worse than its immutable six-bit affine control and misses both
the 2% aggregate and 5% row gates.

Global-Hessian assignment does improve every nonuniform round-to-nearest train
control, reducing train relative L2 from `0.094429/0.179407/0.157468` to
`0.005443/0.041036/0.032874`. Those final train errors again converge near the
four-to-six-bit floor while validation remains poor. This strengthens the
inference that visible-train compensation in a fixed scalar grid does not
generalize; changing scalar level spacing is not sufficient.

Reject this deterministic per-row-group 16-centroid/global-Hessian form. Do
not build its 227,096,395,776-byte bank or lookup kernel. The result does not
reject vector quantization, learned executable programs, or a companion
hardware embodiment that preserves the source representation.

Gate 8 passes across 24 snapshots: minimum free memory is 65%, maximum process
peak RSS is 1,856,372,736 bytes, maximum physical and release-boundary
footprint are 362,843,968 bytes, swap and throttled-page growth are zero, and
protected services remain stable. Raw evidence hashes to
`f8860f648cc6596d5c6a35eca7b2236270676aa421d0890c1d2b02236dffd54a`;
independent analysis hashes to
`eeb5576f1d20f81cfa0c6326622fa649262ac5fcf13ca09625e59b7512044f18`.

No holdout, all-expert audit, runtime, hardware purchase, endpoint, accepted
tokens, TPS claim, or throughput-model constant follows from PW-0149.
