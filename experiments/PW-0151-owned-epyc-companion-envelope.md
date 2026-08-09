# PW-0151 — Owned EPYC companion pre-purchase envelope

- Status: ready to execute
- Disposition: unexecuted
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

Pending execution.

## Decision

Pending execution.
