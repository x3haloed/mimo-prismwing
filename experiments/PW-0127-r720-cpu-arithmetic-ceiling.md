# PW-0127 — Under-$500 R720 CPU arithmetic ceiling

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
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

Unexecuted.

## Decision

Unexecuted.
