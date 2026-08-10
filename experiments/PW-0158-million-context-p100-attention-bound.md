# PW-0158 — million-context two-P100 attention bound

- Status: completed
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  TARGET
  `91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950`;
  PW-0151 analysis
  `d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1`;
  official P100 product brief
  `bda27f98b088ab9ff54e374048e18093374c510c781efcad1e9325b301df4662`
- Hardware candidate: PW-0151's two P100 PCIe 16-GB cards in the owned
  H11SSL-i/EPYC host; analytical pre-purchase ceiling only
- Related records: PW-0020, PW-0028, PW-0151, PW-0154 through PW-0157; E7
- Implementation commit and dirty state:
  `6981410c9c31327639e514708c41587cb4a6e62e`, clean

## Question and changed premise

PW-0154 deliberately derives its 660-slot exact HBM expert cache at an 8,000-
token context, which is the primary decode benchmark. The same declared runtime
must also satisfy TARGET's full-capability one-million-token smoke case: no
truncation or OOM, time to first generated token at most 30 minutes, and at
least one accepted TPS thereafter. Neither PW-0154's cache arithmetic nor
PW-0156's 8K route coverage tests the mandatory global-attention work at that
capability boundary.

Before any purchase or CUDA implementation, bind the pinned nine-global/
39-sliding attention schedule to the fastest advertised arithmetic rate of the
two surviving P100s. The causal question is narrower than endpoint performance:
can ordinary native dense attention fit inside the complete 30-minute prefill
envelope even when every projection, routed expert, softmax, transfer, storage
operation, synchronization, and protocol cost is free?

## Shared construction and compression-depth contract

Capability invariant: preserve all one million input positions, all nine
global-attention layers, all 64 query heads, the published K/V dimensions,
causal visibility, and the 128-position window in all 39 SWA layers. Do not
drop positions, heads, dimensions, layers, or attention history.

Authorized embodiment boundary: this experiment may change layout, tiling,
fusion, scheduling, precision used for a favorable hardware ceiling, and the
mapping across the two P100s. It may not replace dense global attention with a
sparse, linear, retrieval, recurrent, summarized, or learned approximation and
still call the result target-faithful. Such a mechanism is a separately named
L3/L4 branch and remains governed by every distributional, capability, and
long-context gate.

Project constraints remain separate: the cards, PSU, storage, cooling, and
adapters must ultimately fit the complete `$500` and 1,000-W limits. This
experiment evaluates physical arithmetic and HBM fitness before those project
ledgers; a physical failure is decisive for this candidate even if its cards
are inexpensive.

Central truth: each ordinary dense global-attention query must form every
causally visible QK score and apply its probability to V. A layout or fused
kernel can remove materialization and traffic, but cannot omit those products
without changing the attention mechanism.

## Contract

1. Authenticate TARGET, the pinned config, PW-0151, and the official P100
   product brief by SHA-256. Fail closed on any source identity, model revision,
   layer schedule, head count, dimension, sliding window, card count, advertised
   peak, or latency-gate change.
2. Freeze exactly `N=1,000,000` input positions, the minimum literal size of
   TARGET's one-million-token smoke case. Count causal pairs as
   `N(N+1)/2` in every global layer. For every query head and pair, charge one
   multiply and one add for the 192-wide QK dot and one multiply and one add for
   the 128-wide weighted-V accumulation: `2*(192+128)=640` FLOPs.
3. Derive the SWA pair count rather than approximating it. At `N>=128`, it is
   `1+...+127 + 128*(N-127)` per layer. Charge the same 64 query heads and 640
   FLOPs per visible pair. Report global and SWA work separately.
4. Grant both cards their combined advertised FP16 peak continuously across
   the entire attention operation. FP16 is a favorable ceiling, not a fidelity
   promotion: source BF16 and PW-0151's direct-FP32 control cannot execute more
   arithmetic than this grant. Permit perfect dual-card scaling and zero work
   for softmax, masking, RoPE, sinks, reductions, QKV/output projections,
   routed or dense MLPs, cache append, and all system overhead.
5. Compare mandatory attention FLOPs with the peak FLOPs executable in
   TARGET's 1,800-second limit. Report minimum attention-only wall, required
   effective TFLOPS, the factor over combined P100 peak, and remaining or
   exceeded wall budget. These are roofline bounds, not measured hardware
   performance or endpoint timing.
6. Recompute exact BF16 KV bytes for one million positions. First reserve every
   non-routed source tensor from PW-0154 plus its three layer arenas; report
   whether any aggregate HBM remains. Then grant the candidate free streaming
   of all non-routed tensors and report the maximum complete exact expert slots
   left after KV and arenas. This generous alternative cannot repair an
   arithmetic failure.
7. Apply Gate 8 to the analysis process. Report zero accepted tokens and no
   endpoint TPS. Do not convert an advertised peak, calculated floor, or HBM
   capacity into an achieved prefill claim.

## Promotion and kill rule

Reject the two-P100 ordinary-dense-attention embodiment for the required 1M
capability slice if its mandatory attention arithmetic exceeds 1,800 seconds
at the combined advertised FP16 peak. This rejection survives every ordinary
kernel, tiling, fusion, queue, and storage improvement because those costs are
already granted as zero.

A failure does not prove that every admissible Prismwing embodiment is
impossible. Reopening the capability slice requires a separately frozen
mechanism that changes the quadratic attention premise or a different complete
hardware candidate, and it must preserve the long-context behavior and fit the
same cost/power envelope. Do not use this target-faithful ceiling to claim that
an untested approximation fails quality.

## Result

The authoritative `analysis-001` manifest hashes to
`3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315`.
It authenticates TARGET, the pinned config, PW-0151's owned-host analysis, the
official P100 product brief, and the clean implementation commit.

At exactly one million positions, one global layer has `500,000,500,000`
causal query/key pairs. Across nine global layers, 64 query heads, and a
favorable charge of only 640 FLOPs per head/pair for the 192-wide QK dot and
128-wide weighted-V accumulation, global attention requires
`184,320,184,320,000,000` FLOPs. The 39 exact 128-position sliding layers add
`204,459,336,007,680`, for `184,524,643,656,007,680` mandatory attention
FLOPs total.

Granting both cards their combined advertised 37.4-TFLOPS FP16 peak without
interruption gives an attention-only floor of `4,933.8140` seconds, or
`82.2302` minutes. The complete 30-minute gate would require `102.5137`
effective TFLOPS, `2.7410x` the combined advertised FP16 peak. At the
source-control combined FP32 peak the same lower bound is `9,920.6798`
seconds. These floors make all projections, experts, softmax, transfers,
storage, synchronization, and protocol work free; no ordinary kernel or
scheduling improvement can recover the missing `3,133.8140` seconds.

The independent capacity ledger also tightens PW-0154. Exact BF16 KV at one
million positions is `23,065,559,040` bytes. Reserving PW-0154's non-routed
source tensors and three layer arenas produces a `38,221,107,536`-byte total,
`6,221,107,536` bytes beyond two P100s' aggregate decimal HBM. Even granting
free streaming of every non-routed tensor leaves only `6,593,447,936` bytes,
or 261 complete source experts and a `23,564,288`-byte tail.

Gate 8 passes across five snapshots with 62% minimum free memory,
32,997,376-byte peak RSS, 19,629,440-byte maximum physical footprint, zero
swap growth, zero new throttled pages, and stable protected services. The
report records zero accepted tokens, no performance claim, and no endpoint
TPS. No measured throughput-model constant changes.

## Decision

Reject the two-P100 ordinary-dense-attention embodiment for TARGET's mandatory
one-million-token capability slice. This supersedes PW-0151/PW-0154's
conditional retention when those components are proposed as a complete
target-faithful two-P100 system; their 8K decode, cache, storage, and physical
results remain valid component evidence. Do not purchase or implement that
complete configuration.

This is not a project-wide impossibility proof. Reopen the capability slice
only with a separately named sparse, linear, retrieval, recurrent, summarized,
or learned L3/L4 attention mechanism that passes all long-context quality and
capability gates, or with a different complete hardware candidate inside the
same cost and power limits.
