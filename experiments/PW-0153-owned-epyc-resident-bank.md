# PW-0153 — Owned EPYC source-resident bank envelope

- Status: completed
- Disposition: rejected
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; remote-header census
  `8ac4a179c7b0a06baee05e380dc76acd0a1a64cff4d3e2abe9572ce59afb5c52`;
  PW-0151 analysis
  `d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1`;
  Supermicro manual
  `ec9a6b57cba938f74f555a731a0642df76ba83cdb350e51b855f6d0f9ad2dd1a`;
  dated Newegg capture
  `2cabdefea523a9daeedb2547162f0b9e32e502db126a8210fc78f3a3b448ad4c`
- Hardware candidate: owned H11SSL-i rev 1.01 and EPYC 7351P with a
  replacement source-resident DRAM bank and PW-0151's two-P100 direct-FP32
  compute survivor; analytical pre-purchase bound only
- Related records: PW-0112, PW-0127, PW-0151, PW-0152; E7
- Implementation commit and dirty state:
  `729149845a485e4512422ded63a40eef3e9727a6`, clean

## Question and changed premise

PW-0152 shows that continually streaming source experts leaves an extreme
single-transaction acceptance prerequisite. The owned EPYC platform has eight
memory channels and eight DIMM slots. Determine whether installing enough DRAM
to make the complete pinned source tensor payload resident changes that
prerequisite, whether the configuration is physically supported, and whether
a dated complete project branch can remain inside the `$500` incremental cap.

The risk frontier is procurement economics, not just capacity. Keep the
physically possible resident embodiment separate from its dated project
admissibility. A current listing can reject a procurement branch without
proving that the hardware can never be obtained within budget.

## Exactness and red-line check

This is target-faithful L2 capacity and nameplate analysis over unchanged
source bytes. DRAM residency changes acquisition location, not weights,
routes, arithmetic, modalities, or acceptance thresholds. It reports no
endpoint TPS and authorizes no purchase.

## Contract

1. Authenticate the complete remote-header census, PW-0151's clean report,
   the official H11SSL manual PDF, and the dated compatible-memory listing by
   SHA-256. Fail closed unless the census remains pinned to the model revision,
   PW-0151 retains the exact `q=137` selected expert bytes and two-P100 compute
   floor, and the listing contains the named part, capacity, type, speed,
   price, seller, and an add-to-cart control.
2. Sum every tensor category rather than routed experts alone. Report physical
   capacity and byte/GiB headroom for 320, 384, and 512 GiB. Do not count page
   cache, swap, storage, compressed encodings, or uncommitted overcommit as
   resident source capacity.
3. Bind the official AMD-7001 population rules: eight channels and slots,
   64-GiB 2R RDIMM or 64-GiB LRDIMM support up to 512 GiB, same type/size/speed,
   and fewer than eight populated channels supported but not recommended. The
   existing four 4-GiB modules are not additive to a five-by-64-GiB bank.
4. Evaluate the minimum five identical 64-GiB bank and balanced six/eight-DIMM
   alternatives. Grant DDR4-2400 its theoretical 19.2 GB/s per populated
   channel and two independent PCIe-3-x16 links their encoding-adjusted
   nameplate aggregate. Label all of these impossible ceilings, not measured
   bandwidth.
5. Reuse PW-0151's real `q=137` selected expert bytes and direct-FP32 block
   compute time. Report the serial DRAM/PCIe transfer plus compute floor and
   minimum integer `A` for 34.3 and 50 TPS. Preserve the 15-second 8K prefill
   result and every omitted host, dispatch, topology, and endpoint cost.
6. Build separate physical and project ledgers. The physical ledger answers
   whether a supported capacity exists. The project ledger prices five
   currently documented compatible 64-GiB modules plus PW-0151's two P100s,
   then identifies every still-unpriced cable, cooling, tax, shipping, and
   installation item. A component subtotal above `$500` rejects the current
   procurement branch without needing invented prices for omitted items.
7. Do not infer market-wide or permanent impossibility from one active listing.
   Preserve the observation date, URL, captured page hash, seller, quantity
   assumption, and compatibility limitations. Do not purchase from this
   report.
8. Apply Gate 8 during source authentication and evidence publication. Report
   zero accepted tokens, zero endpoint TPS, and no throughput-model constant
   change.

## Promotion and kill rule

This record cannot promote a runtime. Retain the resident-bank embodiment as a
physical architecture only if complete tensor capacity fits supported memory
and the idealized resident `q=137` floor materially lowers both acceptance
prerequisites. Reject the dated procurement branch if the authenticated RAM
subtotal alone exceeds `$500`. Reopen project admissibility only with a new
dated, compatible, complete-BOM observation; do not erase the physical result.

## Result

The authoritative `analysis-003` manifest hashes to
`b11989c53cb93da52140e61c5d16b0152ffc80322184451c08a63c66712444c4`.
It authenticates the complete remote-header census, PW-0151, the official
manual PDF, the dated Newegg page capture, and the clean implementation commit.

Every pinned tensor category totals `315,683,674,448` bytes, or `294.0033`
GiB. Five 64-GiB modules provide 320 GiB and only `25.9967` GiB raw headroom;
that is the byte minimum, but the manual does not explicitly enumerate a
five-module population. Six provide 384 GiB and `89.9967` GiB headroom; the
manual enumerates six as unbalanced and not recommended. Eight provide a
balanced 512-GiB population and `217.9967` GiB headroom. The existing four
4-GiB DIMMs cannot be added because the manual requires the same type, size,
and speed.

The physically resident premise materially changes decode arithmetic. Granting
five DDR4-2400 channels `96.0 GB/s` and two PCIe-3-x16 links an aggregate
encoding-adjusted `31.508 GB/s`, PCIe is the idealized bottleneck. PW-0151's
real `q=137` selected payload transfers in `0.72142` seconds; adding its
`0.20994`-second two-P100 direct-FP32 compute floor gives `0.93137` seconds.
The 34.3-TPS prerequisite falls from `A>=86` to `A>=32/137`; Prismwing 50
falls from `A>=125` to `A>=47/137`. These are impossible nameplate ceilings,
not measured throughput. The unchanged matrix-only 8K prefill floor remains
`12.2596` seconds for two P100s.

The dated project ledger rejects procurement. The captured active listing is
`$247.19` per Hynix `HMAA8GL7MMR4N-UH` 64-GB DDR4-2400 LRDIMM, sold and
shipped by A-Tech with an add-to-cart control. Five total `$1,235.95`, so RAM
alone exceeds the complete `$500` cap by `$735.95`. Adding PW-0151's two used
P100 card-only subtotal yields `$1,384.69`, or `$884.69` over cap, before tax,
shipping, compatible power cables/adapters, forced-air cooling, or physical
validation. No purchase is authorized.

Gate 8 passes across five snapshots with 66% minimum free memory,
139,083,776-byte peak RSS, 85,919,360-byte maximum physical footprint, zero
swap growth, zero new throttled pages, and stable protected services. The
first invocation failed closed on an incorrect assumed census evidence class;
the second failed on an operator-supplied commit typo. Neither published a
manifest. Both empty stdout artifacts are preserved externally. Accepted
tokens and endpoint TPS remain zero, and no throughput-model constant changes.

## Decision

Reject the dated source-resident procurement branch under the `$500` cap. This
is not a permanent market-wide impossibility result: the physically resident
architecture survives, and its idealized acceptance prerequisite is much less
extreme. Reopen project admissibility only with a new dated, compatible,
complete BOM—preferably an explicitly supported six- or eight-module
population—whose total including accelerators, cabling, cooling, tax, and
shipping fits the cap. Do not train a resident-only proposer or buy hardware
from this report.
