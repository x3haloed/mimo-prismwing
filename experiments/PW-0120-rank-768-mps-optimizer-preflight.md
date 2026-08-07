# PW-0120 — Rank-768 MPS optimizer preflight

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0119 analysis
  `166f56b0b56c82099520acd6696647d8bc350b52d5b33d8649d51a7971cf7a34`
- Hardware/runtime: Apple M1 shared 16 GiB; PyTorch 2.13 MPS; verified
  internal-SSD checkpoint
- Exactness: optimizer-substrate diagnostic; prospective L4 trained artifact
- Related records: PW-0045, PW-0116 through PW-0119; E5

## Question and mechanism

PW-0119 rejects rank 128 as the first fidelity target and requires an
activation-weighted rank-768 pilot before a shared bank. First determine
whether the production `(r=768,m=4)` identity-basis parameterization and its
optimizer state can coexist safely with the interactive services on the 16 GiB
M1. This is a memory/optimizer causal preflight, not a fit or quality result.

Instantiate one complete 256-expert canonical `[2048,4096]` projection:

- `A [256,2048,768]`: 402,653,184 F32 values;
- `B [4,768,4096]`: 12,582,912 F32 values; and
- `alpha [256,4]`: 1,024 F32 values.

The 415,237,120 parameters occupy 1,660,948,480 bytes. Parameters, dense
gradients, and Adam first/second moments have a 6,643,793,920-byte semantic
lower bound before allocator and driver overhead. That is close enough to Gate
8 that no rank-768 fit is authorized without a measured phase-level release
test.

## Frozen execution

Reuse the PW-0118 production module and source-tile path with rank 768, four
bases, identity activation, seed `260120`, layer-24 gate expert 23, source rows
`0..7`, columns `0..127`, Adam learning rate `0.001`, and exactly one optimizer
step. The small tile does not stand in for training; it merely forces a real
forward, backward, dense-gradient, Adam-state creation, update, and MPS
synchronization over the production parameter tensors.

Add explicit safety/release observations after module creation, source-tile
load, parameter migration, forward/loss, backward/gradient creation, optimizer
step, parameter/optimizer deletion, MPS cache clearing, and final service
health. Record MPS current/driver allocation, process RSS/peak/physical
footprint, system free-memory percentage, swap growth, throttled pages, disk
bytes, protected-service PIDs, phase wall time, and every released resource.
Write only a small external JSON report; never save parameters, gradients,
optimizer state, source weights, or reconstructed tiles.

Set PyTorch's per-process MPS memory fraction to `0.60` before allocation and
record the resulting recommended maximum. On this host that creates a hard
allocator stop near 7.63 GB, below Gate 8's 8 GiB process ceiling; an allocator
failure is a valid rejected result and must still trigger cleanup and a failed
evidence report.

Before the real process, add a small CPU fixture covering the rank-768/four-
basis equation and proving that the optimizer changes the loss. Authenticate
the checkpoint manifest and exact source tensor name, shape, dtype, scale
layout, and tile hash. Fail closed on unknown MPS availability or memory API
results.

## Gates and stop conditions

1. The frozen checkpoint, tensor, tile, dimensions, parameter count, and
   semantic allocation lower bound must match the contract.
2. Forward, loss, backward, Adam state creation, update, and synchronization
   must be finite and complete exactly once. The final loss need not establish
   convergence, but the update must change at least one selected parameter.
3. Stop before the next phase if system free memory falls below 20%, process
   RSS/peak or physical footprint exceeds 8 GiB, swap grows by more than
   512 MiB, a new throttled page appears, or a protected start-resident service
   disappears. Preserve the last completed phase as failed evidence.
4. After deleting gradients, optimizer state, and parameters and clearing the
   MPS cache, current MPS allocation must be zero and process physical footprint
   must be below 4 GiB. Final protected-service health must remain stable.
5. Report zero accepted tokens, `A=0`, no endpoint timing, and no TPS.

Passing authorizes only a bounded rank-768 activation-weighted fitting
contract. It does not authorize a full 256-expert fit, a shared basis bank, a
runtime artifact, or an inference claim. If allocation, backward, or Adam
crosses Gate 8, reject direct full-state MPS Adam for this shape and use a
block-coordinate, factored-optimizer, CPU-offload, or externally trained path
under a separately frozen contract. Do not relax host safety to make it pass.

## Result

Unexecuted.

## Decision

Unexecuted.
