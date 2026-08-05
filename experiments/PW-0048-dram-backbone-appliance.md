# PW-0048 — Local DRAM-resident backbone appliance

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: unassigned
- Commit and dirty state: proposal based on clean `4eedd12`; no execution
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; every candidate representation
  and node artifact must have a separate lock
- Hardware, OS, compiler, storage, memory pressure: candidate not selected or
  purchased
- Related records: PW-0002, PW-0010, PW-0012, PW-0039; prospective E7

## Hypothesis and mechanism

A single inexpensive used machine with enough DRAM to hold the complete
language backbone can execute contiguous layers more naturally than the M1 can
stream experts from storage. The M1 remains the user-facing host and sends one
request boundary to the appliance; the appliance keeps attention, routers,
experts, norms, KV, and LM-head state together so that 47 per-layer network
round trips disappear.

The first branch is target-faithful source FP8 or a proven L1 representation.
Any INT4, learned codec, or changed arithmetic branch is separately named and
evaluated as L3. Capacity, bandwidth, and advertised integer TOPS are not
performance evidence.

## Contract

Do not purchase hardware on a capacity calculation. The pre-purchase gate
requires measured production-shaped execution on one borrowed, already-owned,
or returnable candidate node.

Pass the pre-purchase gate only if:

1. a dated BOM includes chassis, CPUs/accelerators, all DRAM, storage,
   networking, adapters, power supplies, and cooling at no more than USD $500;
2. measured complete-system peak wall power is projected below 1,000 W with a
   stated safety margin, and the intended configuration requires no unsafe
   electrical, firmware, or thermal modification;
3. the candidate holds the declared complete representation without swap and
   records NUMA placement, memory channels, page policy, ECC state, resident
   bytes, and measured sustained bandwidth;
4. a complete MiMo-shaped layer stage includes norms, QKV, RoPE, global or SWA
   attention as appropriate, router, eight heterogeneous experts, weighted
   reduction, residuals, and all format conversion at representative verifier
   widths;
5. measured stage time, multiplied across the actual layer schedule and joined
   with measured draft, embedding/LM-head, sampling, host-network, and rollback
   costs, predicts at least 12.5 accepted TPS for the Prismwing 10 protocol;
6. the result reports cold and warm state, `q`, `A`, `U`, actual bytes, batch,
   concurrency, latency barriers, CPU/accelerator utilization, energy, and
   confidence bounds; and
7. the node is compared with whole-backbone placement, NUMA-contiguous stages,
   and remote-expert-only execution. The latter is a control, not the assumed
   architecture.

The 12.5-TPS gate provides the required 25% headroom over Prismwing 10 before a
purchase. After acquisition, promotion still requires complete target-faithful
endpoint correctness and accepted-TPS measurements on the declared M1-plus-
appliance system. Kill purchase plans if only nominal bandwidth passes, if
NUMA/network barriers consume the margin, or if BOM/power cannot comply.

## Baseline and candidate

Baseline is the best complete M1 endpoint available at execution time; until it
exists, PW-0039 is component context only. Candidate keeps the complete
language backbone resident on one node. A multi-node pipeline may receive a new
experiment only if one-node evidence establishes why partitioning is needed.

## Isolated attribution

Unexecuted. Market listings and roofline arithmetic may narrow candidates but
cannot satisfy the measured stage gate.

## End-to-end result

Unexecuted. No hardware has been selected, purchased, or measured, and no
accepted-TPS claim exists.

## Correctness result

Unexecuted. New CPU, accelerator, quantization, and transport semantics require
their own correctness fixtures before the complete layer-stage comparison.

## Decision

Unexecuted. Inventory and borrowed-node testing are authorized research;
purchase remains blocked until the predeclared full-stage, BOM, and power gates
pass.
