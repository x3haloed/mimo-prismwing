# PW-0159 — 12-GB Ampere complete-system envelope

- Status: ready
- Disposition: unexecuted
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
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0158 rejects ordinary dense one-million-token attention on two inexpensive
P100s, but it does not reject every accelerator below the project cap. A
12-GB RTX 3080 is the narrow current-market counterexample worth testing: its
Ampere Tensor Cores support BF16, its favorable dense BF16/FP16-class roofline
is approximately 123 TFLOPS, and recent used prices approach the residual
budget. It has much less memory than two P100s, but one card removes
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

Authorized embodiment boundary: grant Ampere BF16 Tensor Core execution and
perfect utilization as a favorable L3 numerical ceiling. This does not promote
the reduction topology or establish target fidelity. The 8K route/storage
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
   work. Grant 123 TFLOPS continuously, perfect Tensor Core occupancy, and
   zero cost for softmax, RoPE, cache traffic, routing, storage, dispatch,
   protocol, and every other operation. Report the remaining wall budget
   inside 1,800 seconds. Kill the candidate if this favorable sum alone
   exceeds the gate.
3. For the 8K TTFT slice, add the same matrix and attention work at 123 TFLOPS
   to exact source-expert installation. Grant the card all 12 decimal GB of
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

Reject the RTX 3080 12-GB candidate outright if the favorable 1M arithmetic
floor exceeds 1,800 seconds. Otherwise, reject the captured procurement branch
if the minimum storage-lane BOM plus a functional active card exceeds `$500`
before or after any known delivery charge; an already-sold bargain only defines
a reopening price and is not an available BOM.

Passing both checks retains only a pre-purchase candidate. Runtime work still
requires a frozen CUDA toolchain, deterministic BF16 correctness fixtures,
measured NVMe concurrency, exact or quality-qualified 1M KV, complete physical
installation, and end-to-end fidelity/performance evidence. A captured cost
failure is not a permanent market-wide impossibility proof; state the maximum
delivered card price that would reopen it.

## Result

Pending completion of PW-0157, source capture, implementation, and execution
from a clean commit.
