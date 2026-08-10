# PW-0165 — affordable RDNA4 complete-system envelope

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0127, PW-0151, PW-0158, and PW-0161 authorities to be authenticated at
  execution; official RX 9060 XT page
  `9013b9e7dfd1e4ecc805e2756df2838b60f3ee0d69a44d25ff8059671194f4ba`;
  official RDNA4 ISA
  `96dc97df3468a4e63a13095e2540ba13aaa75cf4635a29516b59760695e25e0c`;
  official launch release
  `03df4b873908c7e15ef80644888bfa4f1a49999628eda9c4260e34c6c2cdb977`;
  dated retailer transcription
  `79065195a1e523514aa377a91dad8f514db72a504e533b595303699d3148f718`
- Hardware candidate: one Radeon RX 9060 XT 16-GB card in the owned EPYC host;
  analytical pre-purchase envelope only
- Related records: PW-0127, PW-0151, PW-0158, PW-0159, PW-0163, PW-0164; E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0164 permanently rejects the strongest NVIDIA Blackwell tier officially
launched below `$500`, but a current new RX 9060 XT 16-GB listing remains below
the cap and AMD advertises 103-TFLOPS dense half-precision Matrix performance.
Close that distinct consumer counterexample before inferring that the current
sub-`$500` market lacks an ordinary-dense 1M survivor.

The same AMD page advertises 205 TFLOPS with structured sparsity. The RDNA4 ISA
defines dense BF16/F32-accumulate WMMA and a distinct sparse form whose premise
is two zeros in every four elements. Do not substitute the sparse number for
the unchanged dense Prismwing weights.

## Shared construction and compression-depth contract

Capability invariant: preserve all source weights, one million positions, all
nine global and 39 sliding attention layers, native modality paths, and every
TARGET gate. Ordinary dense attention and weights remain the control.

Authorized embodiment boundary: grant the full official 103-TFLOPS dense half
Matrix rate to source-oriented BF16/F32 accumulation, even though achieved HIP
support and utilization are unproven. Treat the same rate as favorable L3 FP16.
Record 205-TFLOPS structured sparsity only as an inadmissible diagnostic.

## Contract

1. Authenticate TARGET, config, prior arithmetic/host authorities, AMD's
   product page, RDNA4 ISA, launch release, and dated retailer transcription.
2. Add two operations per mandatory MAC at one million positions to exact
   ordinary-attention work. Grant 103 TFLOPS plus the EPYC's impossible peak.
3. Reject ordinary-dense RX 9060 XT permanently if that lower bound exceeds
   1,800 seconds. Do not grant 205 TFLOPS without exact source 2:4 sparsity.
4. Recompute exact BF16 1M KV and common/arena capacity against 16 decimal GB.
5. Bind 160-W TBP, 450-W minimum PSU, one 8-pin connector, and the installed
   PSU's 732-W combined +12-V label without calling nameplates installation.
6. Preserve AMD's `$349` launch SEP and the active new `$449.99` free-shipping
   transcription. Unknown tax and installation mean no complete BOM.
7. Apply Gate 8; report zero accepted tokens and no endpoint TPS.

## Promotion and kill rule

Permanently reject RX 9060 XT for ordinary-dense 1M execution if the 103-
TFLOPS lower bound fails. This does not reject explicitly modified 2:4 sparse
weights, FP8, changed attention, RX 9070-class cards, or distributed systems.

## Result

Pending clean implementation and execution.
