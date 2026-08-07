# PW-0128 — Legacy 24-GiB accelerator full-target ceiling

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0112 analysis
  `e93d930549ee9fe761d7fc98bf59642088b3eb9f41c712968f8df26d5b2c8b98`;
  PW-0127 raw arithmetic report
  `6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140`
- Hardware candidate class: 512-GiB dual-E5-2680-v2 R720 plus one or two
  Tesla M40 24-GB cards, or one Tesla P40 24-GB card; analytical
  pre-purchase bound only
- Exactness: target-faithful direct FP32 CUDA arithmetic over unchanged
  source-FP8 weights; no endpoint implementation or measured hardware
- Related records: PW-0048, PW-0105, PW-0110 through PW-0112, PW-0127; E7

## Question and changed premise

PW-0127 rejects the cheap high-memory R720 as a CPU-only Prismwing-50 host,
but a legacy 24-GiB accelerator changes both arithmetic placement and the
physical transaction. PW-0112's `q=94` and `q=137` route unions do not need to
reside globally: a layerwise expert-major verifier can transfer one layer's
selected records, execute all positions for those experts, and reuse bounded
device arenas. Determine whether this conventional host-plus-accelerator
class clears an impossible-perfect physical envelope before requesting remote
access, writing CUDA, or considering a purchase.

The full target includes more than decode. `TARGET.md` also requires an 8K
text time-to-first-token of at most 15 seconds. The same authenticated matrices
must be evaluated once for every uncached prefill token. Give the candidate
only 8,000 positions rather than interpreting 8K as 8,192, and grant server
CPUs plus all GPUs their advertised peaks simultaneously. If mandatory matrix
work alone misses 15 seconds, that is a decisive rejection of the named direct
FP32 embodiment even if wide decode traffic appears favorable.

## Frozen authorities and calculations

Authenticate and read only the immutable PW-0112 and PW-0127 reports. Fail
closed on hashes, revisions, evidence classes, route topology, expert-record
bytes, matrix categories, accepted-token claims, or source decisions.

For every `q=94` and `q=137` sliding window, derive:

- total and maximum per-layer unique expert records;
- exact source expert bytes transferred once per verifier block;
- impossible PCIe 3.0 x16 transfer time at 15.754 GB/s per accelerator;
- impossible accepted TPS with `A=q`, no rejection, perfect balancing, no
  protocol overhead, and no dense/static bytes; and
- one-, two-, and three-arena peak expert residency from the maximum layer
  union, compared with 24 decimal GB of device memory.

For one M40, two M40s, and one P40, add the already-overstated 1.152-TFLOP/s
dual-CPU peak to 7, 14, and 12 FP32 TFLOP/s respectively. Derive mandatory
matrix-only ordinary decode TPS, required peak fraction at 34.3 and 50 TPS,
and the 8,000-position prefill floor. GPU memory bandwidth is diagnostic only;
the direct path still must decode FP8, execute non-matrix work, and transfer
weights from host memory.

Record market evidence separately from physical fitness. A sold `$303.75`
R720 and a current `$150` M40 observation are not a complete BOM. Dell requires
an R720 rather than R720xd, two CPUs no greater than 115 W, low-profile GPU-kit
heatsinks, support brackets, power cables, filler brackets, redundant 1100-W
supplies, and at most a 30 C inlet. Missing kit, storage, networking, shipping,
tax, and cooling remain project-ledger gaps. CUDA 12.x is the last toolkit
family supporting Maxwell; CUDA 13 removes its offline compilation and library
support.

## Gates and interpretation

1. Both source manifests authenticate exactly and all arithmetic and route
   ledgers recompute without tolerance.
2. Reject a named configuration for the target-faithful direct FP32 branch if
   its impossible mandatory-matrix prefill floor exceeds 15 seconds for only
   8,000 uncached positions.
3. Report decode arithmetic, PCIe traffic, and device-arena ceilings even when
   prefill rejects the configuration. These are diagnostic and never accepted
   TPS.
4. A configuration can advance to an owned/borrowed production-shaped stage
   only if it passes the prefill floor, has a dated complete BOM at or below
   `$500`, has a credible complete-system path below 1,000 W, and has supported
   cooling and power hardware. An analytical pass cannot authorize purchase.
5. Report zero accepted tokens, `A=0` for the experiment itself, no endpoint
   timing, and no performance claim. Apply normative Gate 8 during analysis.

Failure rejects only the named direct FP32 legacy-accelerator embodiment. It
does not reject a different accelerator found within a complete BOM, an exact
codec whose executable arithmetic changes the bound, or a clearly named
modified low-bit branch. Passing the decode envelope does not compensate for a
failed full-capability TTFT gate.

