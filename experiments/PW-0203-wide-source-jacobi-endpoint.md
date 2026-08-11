# PW-0203 — Wide source Jacobi endpoint

- Status: completed
- Disposition: rejected for one TPS; retained as an exact-posterior accelerated control
- Date: 2026-08-10
- Execution mode: source-authority weights with explicitly named L3 reduction arithmetic
- Hardware/runtime: existing 16 GiB Apple M1 and internal SSD only
- Related records: PW-0114, PW-0181, PW-0187, PW-0195 through PW-0202

## Hypothesis

PW-0187's `A=5`, `q=8` Jacobi block can turn the compile-time-width,
direct-checkpoint Metal MoE into at least one accepted token per second once
all 47 routed layers, attention, dense layer zero, retained K/V, final norm,
and all eight LM-head rows execute in one complete verifier.

Use PW-0187's hash-locked source prefill states to hydrate real per-layer K/V
outside the timed steady-state interval. In the timed interval execute the
unchanged proposal `[264,13,15,13,15,15,15,15]`, source checkpoint tensors,
real routes, cache updates, residuals, and greedy correction. Record cold and
warm physical reads, `A`, `U`, complete wall, accepted TPS, memory, commit and
dirty state. The target posterior must remain
`[13,15,13,15,481,13,15,15]`; no component timing may be reported as endpoint
TPS.

## Axiom attacks during realization

The implementation falsified four inherited assumptions:

1. Exact source-BLAS BF16 association is not required by the endpoint gate;
   the wide L3 verifier preserves the exact target posterior.
2. Replaying CPU expert prefill is not required for steady-state timing;
   authenticated layer states causally reconstruct every real K/V cache by
   replaying attention only.
3. Expert kernels are not the sole remaining bottleneck; CPU-expanded
   attention and LM-head projections dominated the first complete run.
4. Per-request FP8 content scans and per-layer OS safety subprocesses are not
   model work. Checkpoint hashes authorize prevalidated content, while safety
   remains sampled around the complete transaction.

The final runtime directly binds page-coverable original tensors. The one
layer-9 QKV tensor ending exactly at shard EOF uses a bounded 60.8 MB copied
buffer because a page-rounded no-copy interval cannot exist beyond EOF.

## Results

All three completed variants preserve both cold and warm posterior token IDs
exactly. Run 001, with Metal MoE only, reaches `0.06422` warm accepted TPS.
Run 002 adds direct-checkpoint Metal attention, dense-layer, and LM-head
projections and reaches `0.09302` warm TPS. Run 003 removes duplicate FP8
content scans and moves OS monitoring outside the timed interval. Run 004
repeats that implementation under the corrected `metal_native_l3` semantic;
it reaches `0.21985` warm accepted TPS at `A=5`, `U=2.085106`, and 22.743
seconds.

Run 004 reads 27,508,178,944 physical bytes warm for 27,478,125,440 logical
source bytes. The 47 wide MoE transactions alone take 10.435 seconds and move
19,812,057,088 page-rounded mapped bytes. Peak resident memory is 666,435,584
bytes, system free memory remains at least 68%, swap growth and new throttled
pages are zero, and protected services remain resident.

Evidence:

- run 001 SHA-256
  `1f24f12a1d7a843748bc71dd71387a2bb6b98f5ec93e6a79ca310480d96f3d14`;
- run 002 SHA-256
  `870d309fb969ff236af99f529e9469e4083f991af60588fc3bf4bd6a67998a06`;
- run 003 SHA-256
  `ef0dbcef7ff5596cfcef276a7475b749047ccd46fadb1bb54feb372ec6e7c5ab`;
- corrected-semantic run 004 SHA-256
  `8febe98c77fe779b7ff896205bdcf9086efed5ffc6052ca6ddb173fc5d563b01`.

## Decision

Reject this `q=8` exact-source internal-SSD embodiment for one TPS. It needs a
further `4.5546x` complete-path gain, while `A<=8` and the measured 27.5 GB
transaction cannot fit a persistent exact cache in the 16 GiB machine.
PW-0108's stronger Metal-I/O control can reduce only part of acquisition;
PW-0181's impossible 8 GiB Belady cache already caps the exact path below one
TPS before the full spine, and PW-0182/PW-0183 reject the executable four-bit
escape on fidelity.

Retain PW-0203 as the first complete accelerated wide target verifier and as a
correctness fixture for the new direct-checkpoint linear substrate. Do not
promote a throughput default or claim one TPS. Reopen only with a changed
premise already named by prior evidence: a representative, held-out-passing
executable-byte reduction or a MiMo-specific long-horizon proposer. New
hardware sidecars remain excluded.
