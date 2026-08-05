# Embodiment-jump portfolio

This document packages architecture hypotheses that change the physical unit,
representation, or hardware mapping of MiMo inference. They are not results and
must not be cited as evidence. Each hypothesis has a stable, append-only
experiment record with a cheap falsification gate.

The portfolio exists because incremental kernel rearrangements alone have not
changed PW-0039's dominant source-FP8 expert work. PW-0040 through PW-0043
preserve useful controls, but reject union-phase scheduling, resident F32
expansion, fused gate/up dispatch, and the tested SIMD-group matrix tile as the
missing mechanism.

## Compression-depth contract

### Capability invariants

- Preserve the complete native MiMo-V2.5 capability surface in `TARGET.md`.
- Preserve the pinned tokenizer, processor, modality paths, context behavior,
  tool behavior, and required evaluation slices.
- Keep target-faithful and modified modes visibly distinct.
- Count only accepted tokens through the complete measured decode path.
- Apply the exactness vocabulary and behavioral gates in `RED_LINES.md`.

### Authorized embodiment boundary

The search may replace algorithms, representations, scheduling units, storage
layouts, runtime/framework machinery, OS interfaces, native kernels, and the
mapping across locally owned hardware. Hardware-specific artifacts are allowed.
The search may use local companion hardware within the target contract.

Modified weights or learned replacement programs are research candidates only
under explicit L3/L4 names and artifacts. They do not silently replace the
target-faithful runtime. Remote inference, unsafe hardware modification, and
unverified accepted surrogate output remain outside the boundary.

### Project constraints

- The 16 GiB M1 Mac mini remains the user-facing host.
- New inference-system hardware remains at or below USD $500 total and 1,000 W
  peak measured wall power.
- No rented or internet inference is used after installation and reference
  acquisition.
- Model weights, credentials, private fixtures, and large raw evidence remain
  outside Git; manifests and hashes remain in Git.
- Cheap kill tests precede large implementation, training, or purchases.

Project effort and implementation difficulty are tracked separately from
physical fitness. A candidate can be physically superior and still be rejected
for project reasons; the record must say which ledger caused rejection.

## Current embodiment pressure

PW-0039's real layer-43 fixture makes the current physical problem concrete:

- one native target-faithful dynamic MoE block processes eight positions with
  nine unique experts in about 17.1 ms warm;
- its idealized repetition across 47 routed layers is about 9.95 routed-only
  TPS;
- the diagnostic excludes attention, dense work, storage misses, drafting,
  acceptance failure, sampling, and endpoint overhead;
- source FP8 materializes about 303 GB of routed experts and about 8.8 GiB of
  selected expert weights for an ordinary cold token.

The portfolio therefore favors candidates that remove weight movement, turn
serial time into reusable width, or move the complete model to a substrate that
naturally holds it.

## Candidate portfolio

| Record | Physical premise changed | Mode | Cheapest decisive question |
| --- | --- | --- | --- |
| [PW-0044](../experiments/PW-0044-route-coherent-phrase-lattice.md) | A token is not the physical execution unit; a route-coherent future lattice is | L2 target-distribution-preserving goal | Can a fixed candidate pool cover comparable target probability with materially fewer unique expert bytes per accepted token? |
| [PW-0045](../experiments/PW-0045-routed-mixture-compiler.md) | Experts need not exist as independent matrices; compile the weighted mixture actually observed by the layer | Explicit L3/L4 modified mode | Does direct mixture compilation dominate per-expert compression at matched executable bytes and FLOPs on held-out routed activations? |
| [PW-0046](../experiments/PW-0046-expert-bank-exception-store.md) | The source expert bank is backing/exception state rather than the primary instruction stream | Explicit L3 hybrid mode; exact fallback | Can a conservative gate avoid most exact bytes without hiding tail or modality failures? |
| [PW-0047](../experiments/PW-0047-texture-native-weight-codec.md) | GPU fixed-function texture decode can be the executable weight codec | Explicit L3 modified representation | Does texture fetch plus complete projection beat the promoted source-FP8 path, not merely decode bytes quickly? |
| [PW-0048](../experiments/PW-0048-dram-backbone-appliance.md) | The M1 need not embody the whole backbone; cheap local DRAM can be its body | L0/L1 target-faithful first; modified modes separate | Can one measured, NUMA-local complete layer stage extrapolate to Prismwing 10 with 25% headroom inside BOM and power limits? |

## Dependency and selection order

1. **PW-0044 is the first algorithmic bet after a slow complete text path.** It
   needs real target and draft traces. It is the highest-upside path that can
   preserve the target distribution.
2. **PW-0048 can proceed through inventory and borrowed-node measurement without
   purchasing hardware.** It is the most physically conventional route to a
   target-faithful Prismwing 10.
3. **PW-0045 begins only when representative routed activations exist.** It is
   the deepest software embodiment change and must remain a modified mode.
4. **PW-0046 depends on a useful PW-0045 program.** An exception policy cannot
   rescue a poor resident approximation.
5. **PW-0047 is independent but consumes the black-swan budget.** Stop after one
   week or its fixed kill gate, whichever comes first.

PW-0044 and PW-0048 compose naturally: route-coherent wide verification can
make a DRAM-resident CPU or accelerator operate on useful batches. PW-0045 and
PW-0046 form a separate M1-dominant modified branch. Do not combine candidates
before each mechanism independently passes its attribution gate.

## Pickup protocol for an implementation agent

1. Read `TARGET.md`, `RED_LINES.md`, `LEARNINGS.md`, `docs/WORKFLOW.md`, and the
   selected `PW-NNNN` record.
2. Confirm every prerequisite in the record from committed artifacts. A
   missing prerequisite leaves the experiment `proposed`; it is not a reason
   to substitute synthetic evidence for a real-path claim.
3. Commit the predeclared contract before candidate implementation. If the
   contract must change, record the change and why before observing final
   results.
4. Place large raw outputs in the external evidence root and commit only their
   schema, manifest, hashes, and small redistributable fixtures.
5. Record physical embodiment metrics separately from engineering effort,
   procurement difficulty, and maintainability.
6. On completion, preserve failures, assign a workflow disposition, and update
   `LEARNINGS.md` and `spec/throughput-model.json` only when evidence changes a
   belief or constant.
7. A combined branch receives a new experiment ID. Do not overwrite the
   component records or promote a microbenchmark as endpoint TPS.

PW-0049 is reserved for the prerequisite real base-model decoder-layer causal
transition. The next unreserved experiment ID is `PW-0050`.
