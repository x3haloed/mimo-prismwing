# PW-0028 — Real learned WHT KV precision sweep

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `04fae15`; implementation dirty
- Checkpoint/processor/reference hashes: same locked MTP source/tensors and
  deterministic activations as PW-0026
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2; source read-only
- Related records: PW-0026, PW-0027

## Hypothesis and mechanism

Jointly increasing K and V precision should reduce PW-0026's learned sublayer
error. A WHT plus per-128-vector symmetric affine quantizer at 4, 5, 6, and 8
bits provides a controlled bit-depth sweep distinct from Turbo4's Lloyd-Max
codebook.

## Contract

Use PW-0026's exact learned fixture and source baseline. For each bit depth:

1. normalize no semantics away: apply WHT to the actual padded K or V vector,
   choose one finite symmetric scale from its maximum magnitude, round to the
   signed `[-qmax,qmax]` range, and reconstruct in the rotated domain;
2. change K and V together; preserve RoPE, value scale, learned sinks, GQA,
   softmax, inverse WHT, and learned output projection;
3. produce deterministic finite attention and 4,096-wide projected hashes plus
   relative L2 versus source;
4. charge `2 + ceil(128*bits/8)` bytes per vector block, two K blocks and one V
   block per head, nine global layers at 1,048,576 tokens, and 39 eight-KV-head
   SWA layers at 128 tokens;
5. report Turbo4 alongside the affine sweep but do not treat the two 4-bit
   quantizers as equivalent. Promote only the next candidate for accelerated
   implementation; one deterministic learned fixture cannot establish model
   fidelity.

No performance or endpoint TPS claim is in scope.

## Baseline and candidate

Baseline and Turbo4 are PW-0026. Candidates are WHT-affine 4/5/6/8 applied to
both K and V.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0028`.

## Isolated attribution

The full locked MTP source SHA and tensor metadata pass the same fail-closed
checks as PW-0026. Its sampled QKV scalar-oracle maximum absolute error remains
`1.91264e-6`. A second complete generation was byte-identical to the first.

Each actual 192-wide K was RoPE-transformed, padded to 256, and represented as
two independent 128-value blocks. Each scaled 128-wide V used one block. The
scale itself was rounded to FP16 before reconstruction, codes used ties-to-even
rounding, and all attention and projected outputs were finite.

| Format | Bytes / 128 | Max-context cache | Attention rel. L2 | Projected rel. L2 |
|---|---:|---:|---:|---:|
| Turbo4 | 68 | 7.179 GiB | 18.5848% | 19.4277% |
| affine4 | 66 | 6.968 GiB | 21.4865% | 20.9765% |
| affine5 | 82 | 8.658 GiB | 8.6216% | 10.0036% |
| affine6 | 98 | 10.347 GiB | 3.7891% | 4.3359% |
| affine8 | 130 | 13.725 GiB | 0.9841% | 1.0576% |

The max-context byte totals include nine four-KV-head global layers at
1,048,576 tokens plus 39 eight-KV-head SWA layers at 128 tokens. Affine8 is
`14,737,582,080` bytes: 39.0625% below exact FP16, but 1.912 times Turbo4.
These are storage-model values, not measured traffic or residency.

## End-to-end result

Out of scope; no performance or endpoint TPS claim is permitted.

## Correctness result

The deterministic sweep passes the contract. Affine4 does not improve on
Turbo4 despite its slightly smaller block. Error decreases monotonically from
affine4 through affine8. Candidate output hashes are preserved in
`precision-sweep.json`; the raw evidence manifest is
`b7ac285778c96c07ab61177feceeece4f0dc34f9c5ada5879586f14320853e41`.

## Decision

Promote joint WHT-affine8 K/V as the next accelerated attention candidate. It
is the first tested cache representation to put this learned projected
sublayer near the approximately 1% error regime already selected for the MoE
research substrate, while still reducing the exact maximum-context footprint
from 22.524 GiB FP16 to 13.725 GiB.

Do not promote it for target fidelity. This is one context-17 deterministic
MTP fixture, not base-layer, accumulated-state, logit, hosted-reference, or
endpoint evidence. Affine5/6 remain recorded tradeoff branches; affine4 is
rejected as the default fidelity branch. The next experiment must implement
affine8 in the shared-KV Metal schedule and measure correctness and full
attention-core cost before whole-layer integration.
