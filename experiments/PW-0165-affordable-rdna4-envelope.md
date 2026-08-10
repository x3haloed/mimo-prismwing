# PW-0165 — affordable RDNA4 complete-system envelope

- Status: completed
- Disposition: rejected
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
- Implementation commit and dirty state:
  `28776eac37264eb1a1366bd4a9fe0feccef12da8`, clean

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

The authoritative `analysis-001` manifest hashes to
`7ce474e66fca10bb87b3a5c016f689792119539b87c54438245473e153999d58`.
It authenticates TARGET, the pinned config and arithmetic authorities, the
owned host, AMD's product page, RDNA4 ISA and launch release, the dated active
retailer transcription, and the clean implementation commit.

RX 9060 XT does not rescue ordinary dense one-million-position execution.
Granting AMD's entire 103-TFLOPS dense half-precision Matrix rate to source-
oriented BF16/F32 accumulation and the EPYC's impossible 0.7424-TFLOPS peak
concurrently still needs `2,064.3998` seconds. This exceeds the complete gate
by `264.3998` seconds before softmax, routing, source-FP8 decode, memory
traffic, storage, dispatch, protocol, or any other operation. The same 103-
TFLOPS rate is also the favorable dense FP16 L3 ceiling, so numerical loosening
alone does not repair ordinary dense 1M execution on this card.

AMD's 205-TFLOPS structured-sparse rate would reach an idealized
`1,040.9414` seconds and leave `759.0586` seconds, but it is not admissible for
the unchanged source weights. The authenticated ISA defines sparse WMMA around
the distinct premise that two elements of every four are zero. PW-0165 does
not authorize pruning or changed weights, so the sparse row is preserved as a
modified-representation clue rather than promoted as source performance.

Exact BF16 1M KV alone exceeds 16 decimal GB by `7,065,559,040` bytes; KV,
three arenas, and common weights exceed it by `22,221,107,536`. The 160-W GPU
plus 170-W CPU leaves 402 W under the PSU's 732-W +12-V label and uses one
8-pin input, but this favorable nameplate margin is not fit, cable, thermal,
or measured-load proof.

AMD's official 16-GB SEP was `$349`. The dated retailer row records a new,
in-stock card at `$449.99` with free shipping before unknown tax, leaving only
`$50.01` before any other required installation part. No complete delivered
BOM is proven, and the permanent ordinary-dense arithmetic rejection makes
purchase irrelevant for that mode.

Gate 8 passes across five snapshots with 49% minimum free memory,
40,108,032-byte peak RSS, 20,252,288-byte maximum physical footprint, zero
swap growth or new throttled pages, an explicit release boundary, and stable
services. One failed invocation published no manifest and corrected an
operator commit typo. Accepted tokens and endpoint TPS remain zero.

## Decision

Permanently reject RX 9060 XT for ordinary dense one-million-position
Prismwing. Preserve an explicitly modified 2:4 sparse representation, FP8,
changed attention, RX 9070-class cards, and distributed systems as separate
candidates. Authorize no purchase or HIP runtime.
