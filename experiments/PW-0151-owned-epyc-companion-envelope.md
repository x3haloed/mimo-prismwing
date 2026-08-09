# PW-0151 — Owned EPYC companion pre-purchase envelope

- Status: completed
- Disposition: conditional
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0112 analysis
  `e93d930549ee9fe761d7fc98bf59642088b3eb9f41c712968f8df26d5b2c8b98`;
  PW-0127 arithmetic `6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140`
- Hardware candidate: owned Supermicro H11SSL-i rev 1.01, EPYC 7351P,
  16 GiB DDR4-2133, EVGA SuperNOVA NEX750B, empty PCIe slots, no accelerator,
  no NVMe; analytical pre-purchase bound only
- Related records: PW-0048, PW-0102, PW-0110 through PW-0112, PW-0127,
  PW-0128, PW-0150; E7
- Implementation commit and dirty state: `7affcf01feffd3514afbe744e023c5e88410f465`,
  clean

## Question and changed premise

The project owner has supplied a machine-readable census and PSU-label photo
for an already-owned companion host. Unlike PW-0127/PW-0128's hypothetical
R720, its chassis and motherboard cost zero against the incremental `$500`
cap. Determine which conventional source-preserving configurations are already
impossible, and state the exact surviving physical prerequisite before any
purchase or CUDA implementation.

## Exactness and red-line check

The main branch is target-faithful direct FP32 arithmetic over unchanged source
weights. Pascal and Volta lack native BF16 tensor execution, so advertised
FP16/tensor peaks are reported only as named L3 diagnostic branches requiring
the complete near-equivalence suite. No GPU, storage, RAM, cable, fan, or
adapter is purchased by this experiment.

## Contract

1. Authenticate both supplied census attachments and the PSU photo by SHA-256.
   Fail closed unless the text contains the exact CPU, board, memory, slot,
   storage, and network observations used by the report.
2. Authenticate PW-0112's real `q=94`/`q=137` route unions and PW-0127's
   complete mandatory matrix-operation ledger.
3. Grant the EPYC all 16 cores at 2.9 GHz and 16 FP32 operations/cycle. Reject
   CPU-only if even this impossible peak misses a named decode target.
4. Evaluate one P40, one/two P100, and one V100 at advertised FP32 peak. Report
   FP16/tensor alternatives separately as L3 diagnostics. An 8K prefill fails
   if mandatory matrices alone exceed 15 seconds.
5. For each real route window, derive exact selected expert bytes and ideal
   striped-storage floors at one through four independent PCIe-3-x4 lanes,
   using 2.5 and 3.5 GB/s per lane only as nameplate sensitivity points. Add
   direct-FP32 block compute without assuming storage/compute overlap.
6. Report the minimum accepted prefix `A` required for 34.3 and 50 TPS. Do not
   call perfect acceptance, nameplate bandwidth, or summed independent phases
   measured performance.
7. Bind the EVGA label/manual facts: 750 W continuous at 50 C, 61 A/732 W
   combined +12 V, four 20-A rails, VGA1/VGA2 on +12V2 and VGA3/VGA4 on
   +12V4. One or two passive 250-W cards remain unproven until original cable,
   connector, airflow, clearance, and measured wall-power evidence exists.
8. A purchase branch survives only if compute, storage, capacity, slots,
   electrical/thermal installation, and a dated complete BOM including tax and
   shipping all remain possible. Market observations are not a purchase order.

Gate 8 applies to the analyzer. Report zero accepted tokens and no endpoint
TPS. Preserve every omitted cost and physical uncertainty explicitly.

## Result

The authoritative `run-003` report hashes to
`d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1`.
It authenticates both host censuses, the PSU photo
(`3c398ea5c2a12b71908c5b9adcf16d58fc6e26e867cd7c38c550f42bea367b42`),
PW-0112, PW-0127, and the clean implementation commit.

CPU-only fails before implementation work. Even granting all 16 cores
2.9 GHz and 16 FP32 operations/cycle, the impossible ceiling is only
`25.0463 TPS`, below both `34.3` and `50`. Mandatory 8K matrix-only prefill
floors are `18.6095` seconds for one P40, `23.6128` for one P100, `16.0848`
for one V100, and `12.2596` for two P100s. Thus two P100s are the sole tested
direct-FP32 compute configuration to clear the 15-second impossible floor.

The real `q=137` route window selects 903 layer-expert records totaling
`22,730,287,104` source bytes. Two P100s at advertised FP32 peak plus the
impossible EPYC grant need `0.2100` seconds for its mandatory matrices. At
four independent storage lanes granted 2.5 GB/s each, serial expert read plus
matrix compute is `2.48297` seconds. Perfect acceptance would be only
`55.1758 TPS`; `34.3 TPS` requires at least `A=86/137`, while `50 TPS`
requires `A=125/137` (`91.24%`). No measured storage, compute, proposer, or
endpoint result supports those grants.

Nameplate sensitivity does not make four lanes logically unique. Three lanes
at 3.5 GB/s leave the `q=137` idealized 50-TPS branch open at `A>=119`; four
such lanes leave it open at `A>=92`. A `q=94` window can reach the 50-TPS
arithmetic only at the most optimistic tested four-by-3.5-GB/s sensitivity
point. The retained conservative prerequisite is therefore two P100s,
aggregate expert storage around 10 GB/s or better, and a base-aligned wide
proposer with exceptionally high exact acceptance—not a claimed BOM.

`run-001` is preserved but rejected because the operator supplied a nonexistent
full commit expansion; it hashes to
`86ce9908211e0cf6ff90d731f295bc0ab3a729c56e06b30f0459d34fbfe48b6a`.
`run-002` corrected the value and hashes to
`1f3bc8538a3806743039ebc1580a694e46644d8b809f26c485704ff027dfc3e5`,
but exposed that the analyzer accepted rather than authenticated the supplied
commit. The authoritative rerun adds fail-closed HEAD and clean-tree checks.

Gate 8 passes across five `run-003` snapshots with 65% minimum free memory,
30,932,992-byte peak RSS, zero swap growth or new throttled pages, and stable
protected services. Accepted tokens and endpoint TPS remain zero.

## Decision

Reject the owned EPYC by itself and every tested single-card direct-FP32
configuration. Conditionally retain the two-P100/wide-speculation/striped-NVMe
envelope only as a pre-purchase prerequisite. Do not buy from this report.
Next falsify delivered BOM, drive sustained-read behavior, physical clearance,
original-compatible cable availability and pinout, forced-air cooling, and
proposer acceptance. FP16 and V100 tensor execution remain separate L3 modes
requiring full near-equivalence evidence.
