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

PW-0049 and PW-0050 completed the prerequisite real layer and slow text causal
transitions. The intervening ledger now runs through PW-0205; the old
`PW-0051` allocation note is historical, not current ID authority.

## 2026-08-10 axiom attack: corrected semantics plus RAM as scheduling capital

This section is a post-PW-0205 abductive search record, not measured evidence.
It incorporates the corrected QKV mapping, the coherent arbitrary-text path,
the newly authorized 13 GiB process ceiling, and current primary research. Its
purpose is to expose premises that earlier negative results quietly depended
on and assign cheap tests before implementation.

| Inherited axiom | Attack | Abductive jump | Record |
| --- | --- | --- | --- |
| Low residency is inherently safer | Safety depends on bounded lifetime, reserve, pressure response, and zero swap—not on an arbitrary 8 GiB ceiling | Treat 8–12 GiB as an explicitly declared, evictable working-set cache | [PW-0207](../experiments/PW-0207-pressure-elastic-resident-working-set.md) |
| Old proposer traces remain authoritative after a verifier repair | QKV row semantics affect hidden states, routes, posterior tokens, and therefore both `A` and `U` | Regenerate all proposal authorities before carrying forward rejection or promotion | [PW-0206](../experiments/PW-0206-corrected-qkv-authority-regeneration.md) |
| Acceptance length is the speculative objective | On a streamed MoE, a longer branch can be slower when it scatters into more experts | Optimize accepted tokens per unique expert byte and marginal miss latency | [PW-0208](../experiments/PW-0208-native-mtp-cost-aware-proposer.md) |
| Prefill and decode want the same physical kernel | Prefill has token width and reuse; decode is chiefly a weight-movement problem | Give prefill a separate layer-major, high-residency Metal embodiment | [PW-0209](../experiments/PW-0209-layer-major-high-residency-prefill.md) |
| GPU SIMD is an alternative GEMM API | Mature GEMM paths are not the obvious loss; packed-code transforms, reductions, routing, RoPE, and barriers are | Keep values packed until a SIMD group consumes them and fuse the irregular envelope around projection | [PW-0210](../experiments/PW-0210-simdgroup-packed-domain-fusion.md) |
| Cache the most frequent tensor | Frequency ignores load size, miss concurrency, reuse distance, and critical-path stall | Rank residency by measured marginal wall time avoided per byte under the actual route trace | [PW-0207](../experiments/PW-0207-pressure-elastic-resident-working-set.md) |

The central abductive claim is that Prismwing's next large gain is more likely
to come from composing *width plus residency plus route cost* than from another
isolated arithmetic kernel. PW-0205 reads about 1.636 TB to emit 47 tokens
while peaking below 4 GiB. The newly available 8 GiB of persistent headroom is
therefore not merely capacity; it is an opportunity to remove repeated
critical-path acquisition. This remains a hypothesis until PW-0207 attributes
real physical reads and complete wall time.

Research triangulation strengthens, but does not prove, this direction:
MiMo-V2-Flash reports useful native-MTP acceptance; FastMTP argues that draft
training must match recursive inference; EcoSpec identifies expert scattering
as a speculative-decoding cost; Apple documents SIMD-group execution and
shared CPU/GPU storage; BaseRT separates compute-rich prompt processing from
memory-bound decode. None of those results uses the pinned MiMo-V2.5 checkpoint
on the 16 GiB M1, so every imported mechanism remains only a prior.

Selection order is PW-0206 first, then PW-0207 and PW-0208. PW-0209 and
PW-0210 may proceed after their fixtures exist, but cannot displace endpoint
work on kernel-only numbers. A combined high-residency/MTP/SIMD path is allowed
only after each component clears its own kill gate.
