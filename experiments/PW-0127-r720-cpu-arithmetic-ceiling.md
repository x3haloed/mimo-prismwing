# PW-0127 — Under-$500 R720 CPU arithmetic ceiling

- Status: completed
- Disposition: negative for Prismwing 50; 34.3 remains measurement-only
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: implementation
  `0c1542f627336ee710aea066907cb26b6b57b666`; clean tree at execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0105/PW-0112 source traffic authorities
- Hardware candidate class: Dell PowerEdge R720, two Xeon E5-2680 v2 CPUs,
  512 GiB RDIMM; analytical pre-purchase bound only
- Exactness: target-faithful source matrix arithmetic lower bound; omits work
  and therefore favors the candidate
- Related records: PW-0048, PW-0105, PW-0110 through PW-0112, PW-0126; E7

## Question and changed premise

The reduced $500 cap still admits historical used-market examples of a
complete 512-GiB dual-socket server: one R720 with two E5-2680 v2 CPUs and no
disk sold for `$303.75`. Capacity is therefore not enough to dismiss PW-0048.
Before asking for borrowed-node access or building an x86 stage, determine
whether that CPU-only candidate class has even an impossible-perfect arithmetic
path to Prismwing 50.

Intel specifies ten cores, 3.60-GHz maximum single-core turbo, AVX, 115-W TDP,
and 59.7 GB/s maximum memory bandwidth per E5-2680 v2. Grant both sockets the
single-core maximum turbo on all twenty cores continuously and grant every core
the Ivy Bridge AVX peak of 16 F32 operations/cycle. This deliberately
overstates sustainable compute. The resulting ceiling is 1.152 TFLOP/s.

## Frozen authenticated arithmetic

Read only verified checkpoint tensor metadata and count mandatory matrix MACs
for one ordinary incremental target token:

1. all 48 QKV projections and all 48 output projections;
2. layer 0's dense gate, up, and down projections;
3. all 47 routed-layer router matrices;
4. eight selected source experts' gate, up, and down projections across all 47
   routed layers; and
5. the final LM head.

Each matrix element contributes one multiply and one add: two F32-equivalent
operations per MAC. Fail closed on any unknown shape, count, dtype, layer, or
revision. Separately authenticate PW-0105's `9,464,659,968` selected expert
bytes per ordinary token and compare it with the impossible dual-socket
`119.4 GB/s` memory-bandwidth sum.

Exclude embedding lookup, RMSNorm, RoPE, attention score/value work, softmax,
KV traffic, nonlinearities, BF16/FP8 conversion, scale reads, route selection,
threading, NUMA exchange, network, sampling, and every utilization loss. The
bound also grants all-core maximum turbo and perfect simultaneous add/multiply
issue. These omissions make a failure decisive and a pass merely inconclusive.

Add fixtures for shape-derived MAC accounting, two-operations-per-MAC
conversion, peak-FLOP algebra, TPS ceilings, required peak utilization, and
invalid candidate specifications.

## Gates and interpretation

1. Checkpoint matrix identities and dimensions reproduce the expected 48-layer,
   47-routed-layer, top-eight architecture and every count closes exactly.
2. The CPU-only candidate class is rejected for Prismwing 50 if mandatory
   matrix arithmetic alone exceeds its impossible 1.152-TFLOP/s peak at 50
   TPS, or equivalently if its ideal ceiling is below 50 TPS.
3. Report the same ceiling and required peak utilization for 10, 12.5, 34.3,
   and 50 TPS. Do not convert an analytical pass at a lower rate into a
   performance claim.
4. Report the ordinary-token expert-only bandwidth ceiling; if it supplies
   less than 25% headroom over 10 TPS, require measured wide/expert-major
   batching for any borrowed-node experiment.
5. Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS claim.

A rejection kills only the target-faithful CPU-only dual-E5-2680-v2/R720 class
for Prismwing 50. It does not reject the valuable 34.3-TPS horizon, a GPU or
newer CPU found within the complete $500 BOM, a modified low-bit model, or
PW-0048 generally. If 34.3 consumes at least 80% of impossible peak, retain it
only as a measurement question: no purchase based on the roofline. A
production-shaped borrowed-node stage remains mandatory before any purchase.

## Result

The authenticated checkpoint contains a mandatory `14,820,573,184` matrix MACs
per ordinary target token before any omitted work:

| Category | MACs/token |
| --- | ---: |
| Selected experts | `9,462,349,824` |
| Attention projections | `4,482,662,400` |
| LM head | `624,951,296` |
| Dense layer-0 MLP | `201,326,592` |
| Routers | `49,283,072` |

At two operations per MAC, the lower bound is `29,641,146,368` operations per
token. The deliberately impossible dual-CPU peak is 1.152 TFLOP/s, yielding a
hard ceiling of `38.8649 TPS`. Ten and 12.5 TPS consume 25.73% and 32.16% of
that peak. The valuable 34.3-TPS horizon consumes 88.25%, while 50 TPS requires
128.65% and is arithmetically impossible before FP8 decoding, attention-score
work, KV traffic, NUMA, or any other omitted cost.

The independent ordinary-token bandwidth bound is `12.6154 TPS` for selected
expert bytes alone at an impossible dual-socket 119.4 GB/s. This barely clears
PW-0048's 12.5-TPS pre-purchase threshold before dense weights and therefore
cannot support purchase; a real stage would need measured wide/expert-major
reuse.

Gate 8 passes at 79% minimum free memory, 179,292,288-byte maximum physical
footprint, zero swap growth or new throttled pages, and stable services. Raw
evidence hashes to
`6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140`;
independent analysis hashes to
`5a44e66114b51e2b241acb26fcb2c58280fc2a823314b273c0761d58c27ff113`.

## Decision

Reject the target-faithful CPU-only dual-E5-2680-v2/R720 class for Prismwing
50. Retain 34.3 TPS only as a borrowed-node measurement question: it needs
88.25% of an impossible peak before unavoidable work and is not a credible
purchase inference. This does not reject a complete under-$500 GPU/newer-CPU
BOM, modified low-bit execution, or PW-0048 generally. No endpoint TPS or
throughput-model measured constant changes.
