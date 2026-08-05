# PW-0028 — Real learned WHT KV precision sweep

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `629e857`; contract dirty
- Checkpoint/processor/reference hashes: same locked MTP source/tensors and
  deterministic activations as PW-0026
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); MLX 0.31.2; source read-only
- Related records: PW-0026, PW-0027

## Hypothesis and mechanism

Jointly increasing K and V precision should reduce PW-0026's learned sublayer
error. A WHT plus per-128-vector symmetric affine quantizer at 4, 5, 6, and 8
bits provides a controlled bit-depth sweep distinct from Turbo4's Lloyd-Max
codebook.

## Contract

Use PW-0026's exact learned fixture and source baseline. For each bit depth:

1. normalize no semantics away: apply WHT to the actual padded K or V vector,
   choose one finite symmetric scale from its maximum magnitude, round to the
   signed `[-qmax,qmax]` range, and reconstruct in the rotated domain;
2. change K and V together; preserve RoPE, value scale, learned sinks, GQA,
   softmax, inverse WHT, and learned output projection;
3. produce deterministic finite attention and 4,096-wide projected hashes plus
   relative L2 versus source;
4. charge `2 + ceil(128*bits/8)` bytes per vector block, two K blocks and one V
   block per head, nine global layers at 1,048,576 tokens, and 39 eight-KV-head
   SWA layers at 128 tokens;
5. report Turbo4 alongside the affine sweep but do not treat the two 4-bit
   quantizers as equivalent. Promote only the next candidate for accelerated
   implementation; one deterministic learned fixture cannot establish model
   fidelity.

No performance or endpoint TPS claim is in scope.

## Baseline and candidate

Baseline and Turbo4 are PW-0026. Candidates are WHT-affine 4/5/6/8 applied to
both K and V.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0028`.

## Isolated attribution

Pending.

## End-to-end result

Out of scope; no performance or endpoint TPS claim is permitted.

## Correctness result

Pending.

## Decision

Pending.
