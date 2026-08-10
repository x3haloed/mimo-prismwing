# PW-0159 — 12-GB Ampere complete-system envelope

- Status: completed
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0127 arithmetic and PW-0157 route authorities to be authenticated by
  SHA-256 at execution; official NVIDIA architecture/specification captures
  and dated market observations to be frozen after this contract
- Hardware candidate: one used GeForce RTX 3080 12 GB in the owned
  H11SSL-i/EPYC host plus the minimum independently readable NVMe set; analytical
  pre-purchase envelope only
- Related records: PW-0020 through PW-0029, PW-0112, PW-0127, PW-0151,
  PW-0154 through PW-0158; E7
- Implementation commit and dirty state:
  `863594b8a9e296f4fafbe6f68f15eecec0c1c743`, clean

## Question and changed premise

PW-0158 rejects ordinary dense one-million-token attention on two inexpensive
P100s, but it does not reject every accelerator below the project cap. A
12-GB RTX 3080 is the narrow current-market counterexample worth testing: its
Ampere Tensor Cores support BF16, its favorable dense FP16-with-FP16-accumulate
roofline is approximately 122.6 TFLOPS, and recent used prices approach the
residual budget. It has much less memory than two P100s, but one card removes
cross-device sharding and leaves materially more electrical margin.

Test the complete causal envelope before requesting a purchase. It is not
enough for the card to clear PW-0158's attention-only floor: charge all
mandatory matrices during the one-million-token prefill, exact 8K routed
weight installation, the minimum required storage hardware, and delivered
component cost. Preserve the distinction between an arithmetic survivor, a
captured purchasable BOM, and a measured runtime.

## Shared construction and compression-depth contract

Capability invariant: preserve all one million positions, the pinned nine
global and 39 sliding attention layers, all source heads/dimensions, all source
experts, native modality paths, and every TARGET gate. No sparsity or
summarization is granted in the primary arithmetic screen.

Authorized embodiment boundary: use dense Ampere BF16 Tensor Core execution
with FP32 accumulation as the source-oriented control. Report dense FP16 with
FP16 accumulation separately as a favorable L3 numerical ceiling. Neither
nameplate promotes the reduction topology or establishes target fidelity. The 8K route/storage
screen keeps exact source-FP8 expert records and may preload only complete
records into HBM. A compressed KV mode is reported separately; it cannot be
used to claim source-exact 1M residency or fidelity.

Project constraints: the M1 remains the user-facing host; the owned EPYC host
costs zero; every newly acquired card, storage device, adapter, cable, and
cooling part must fit `$500` delivered including tax and shipping; complete
measured wall power must remain at most 1,000 W and the installed PSU's tighter
electrical limits must be respected.

## Contract

1. Authenticate TARGET, the pinned config, PW-0127's mandatory
   `14,820,573,184` MAC/token ledger, PW-0158's attention arithmetic, and the
   completed PW-0157 route manifests by SHA-256. Fail closed on revision,
   attention schedule, geometry, route semantics, expert size, commit, or
   source-identity drift.
2. Freeze exactly one million input positions. Charge two operations per
   PW-0127 MAC for every position and add PW-0158's exact ordinary attention
   work. Derive both dense rates from the official 8,960-core, 1.71-GHz card
   geometry and GA10x per-SM Tensor throughput: about 61.3 TFLOPS for BF16 with
   FP32 accumulation and 122.6 TFLOPS for FP16 with FP16 accumulation. Grant
   perfect occupancy and zero cost for softmax, RoPE, cache traffic, routing,
   storage, dispatch, protocol, and every other operation. Report both floors
   and their remaining 1,800-second budgets. Kill the source-oriented control
   if its sum exceeds the gate; keep the FP16 mode visibly L3 and unqualified.
3. For the 8K TTFT slice, give exact source-expert installation the faster L3
   FP16 arithmetic ceiling, which is most favorable to the storage candidate.
   Grant the card all 12 decimal GB of
   HBM, free streaming of every non-routed tensor, exact BF16 8K KV, and
   PW-0128's three maximum layer arenas. Convert the remainder to complete
   25,171,968-byte expert slots.
4. Give offline residency perfect foresight: subtract every available HBM
   expert slot from PW-0157's observed distinct `(layer, expert)` records even
   if the selected records would not all be useful. Stream every remaining
   distinct record exactly once. This is a lower bound; eviction, repeats,
   filesystem work, and cache installation are free.
5. Test one through four independent storage lanes at 3.5 GB/s each. The
   minimum lane count is the first whose serial storage plus arithmetic floor
   is at most 15 seconds. Nameplate lanes are not measured bandwidth.
6. Freeze a dated market ledger for an active, functional 12-GB RTX 3080 plus
   that minimum number of capacity-sufficient NVMe devices and the cheapest
   topology-compatible adapters. Reject parts-only, board-issue, zero-feedback,
   ambiguous-memory, and already-sold cards as purchase authority. Include
   shipping and mark unknown tax as incomplete rather than zero.
7. Bind official card geometry, 350-W board power, connector requirements,
   the owned board slots, and the photographed NEX750B label. No purchase is
   authorized until chassis clearance, original-compatible dedicated VGA
   cables, rail assignment, cooling, and staged power/temperature checks are
   proven.
8. Report exact BF16 1M KV residency separately from PW-0020/PW-0029's named
   compressed-KV modes. A compressed mode remains L3 and must pass accumulated
   local-logit, hosted distributional, capability, and long-context gates.
9. Apply Gate 8 to the analyzer. Record zero accepted tokens and no endpoint
   TPS. Do not present Tensor, HBM, PCIe, storage, or market nameplates as
   achieved performance.

## Promotion and kill rule

Reject the target-faithful/source-oriented RTX 3080 12-GB arithmetic control if
its BF16-with-FP32-accumulate 1M floor exceeds 1,800 seconds. Treat the faster
FP16-with-FP16-accumulate result only as an L3 branch requiring every fidelity
gate. Independently reject the captured procurement branch if the minimum
storage-lane BOM plus a functional active card exceeds `$500` before or after
any known delivery charge; an already-sold bargain only defines a reopening
price and is not an available BOM.

Passing both checks retains only a pre-purchase candidate. Runtime work still
requires a frozen CUDA toolchain, deterministic BF16 correctness fixtures,
measured NVMe concurrency, exact or quality-qualified 1M KV, complete physical
installation, and end-to-end fidelity/performance evidence. A captured cost
failure is not a permanent market-wide impossibility proof; state the maximum
delivered card price that would reopen it.

## Result

The corrected authoritative `analysis-002` manifest hashes to
`945079702501f990e2cdd40a326b09fad0f2bb71b3f9615c8114c0bbd71590c2`.
It authenticates TARGET, the config, PW-0127, PW-0155, PW-0157's exact
4,096-position lower bound, PW-0158, the throughput model, two official NVIDIA
captures, the dated market transcription, and the clean implementation commit.

The official geometry gives 70 SMs. At 1.71 GHz and 512 dense FP16 FMA per SM
clock, dense FP16 with FP16 accumulation is `122.5728` TFLOPS. Dense BF16 with
FP32 accumulation has half that Tensor throughput, `61.2864` TFLOPS; no
structured sparsity is granted. Mandatory 1M matrices plus ordinary attention
total `214,165,790,024,007,680` FLOPs. The source-oriented BF16 control needs
`3,494.5076` seconds (`58.2418` minutes), failing the complete 30-minute gate
before all omitted work. The L3 FP16 diagnostic needs `1,747.2538` seconds and
leaves only `52.7462` seconds for the entire omitted prefill; it remains a
numerical hypothesis, not a passing mode.

For 8K, the faster L3 arithmetic grant costs `2.0441` seconds. After freely
streaming all non-routed tensors and reserving exact BF16 KV plus three arenas,
12 decimal GB can preload only 375 complete experts. The first 4,096 positions
of PW-0157 already touch 4,585 distinct records, so even perfect-foresight
residency leaves 4,210 records or `105,973,985,280` bytes to install. One and
two 3.5-GB/s lanes have serial floors of `32.3224` and `17.1832` seconds;
three lanes first clear the impossible 15-second bound at `12.1369` seconds.

The captured active card is `$446.72` delivered before tax. Three ambiguous
256-GB drives and three single-drive adapters add `$128.28`, producing a
`$575.00` subtotal before unknown tax, cables, and any missing installation
parts. The captured branch is already `$75.00` over the complete cap. A
historical sold card would have produced `$490.78` before its unknown delivery
and all tax, but it is not purchasable. The maximum delivered card price that
could reopen this exact parts ledger is `$371.72` before unknown tax.

One 350-W card plus the 170-W CPU leaves 212 W below the PSU's combined 732-W
+12-V label, a better nameplate margin than two P100s but not an electrical,
thermal, cable, or fit proof. Exact BF16 1M KV exceeds card HBM by
`11,065,559,040` bytes before arenas. PW-0020 Turbo4 fits arithmetically but is
an unqualified L3 format whose direct fork was already rejected and whose
accumulated fidelity is unproven.

Gate 8 passes across five snapshots with 63% minimum free memory,
294,305,792-byte peak RSS, 130,827,968-byte maximum physical footprint, zero
swap growth, zero new throttled pages, and stable protected services. The
report records zero accepted tokens, no performance claim, and no endpoint
TPS. No measured throughput-model constant changes.

## Decision

Reject the source-oriented BF16 RTX 3080 12-GB embodiment on the mandatory 1M
arithmetic gate. Reject the captured active three-lane procurement branch on
cost before tax. Do not purchase this card or the captured storage parts.

Retain only a price-triggered L3 FP16 architecture hypothesis: it requires an
active functional card delivered materially below `$371.72`, a complete
under-cap BOM after tax, three verified sustained-read lanes, exact physical
installation, a qualified 1M KV mechanism, and full numerical/hosted fidelity
evidence. The historical bargain and nameplate rooflines do not satisfy any of
those conditions.

## Precision-rate correction before authoritative execution

The first clean analyzer implementation incorrectly labeled 123 TFLOPS as a
dense BF16 rate. Inspection of the already frozen official GA102 whitepaper
showed that the 10-GB RTX 3080's published dense BF16-with-FP32-accumulate rate
is 59.5 TFLOPS, while 119 TFLOPS is either dense FP16-with-FP16-accumulate or
the structured-sparse BF16 figure. Scaling the same official SM equation from
68 to the 12-GB card's 70 SMs gives 61.2864 and 122.5728 TFLOPS, respectively.

The erroneous `analysis-001` is preserved externally and must not be cited as
evidence. Its manifest hashes to
`01d8cfdf8d6603fcec3c867b8eed14a71bacf2ad9734539d5479b3c3136fd48e`.
This amendment is committed before the corrected analyzer and authoritative
execution. No threshold changes; the correction separates two numerical modes
that the first implementation conflated.
