# PW-0118 — Identity-basis MPS optimizer preflight

- Status: complete
- Disposition: conditional
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0117 analysis
  `337b57c43638025673eb494eecfc87445468d21b9a1ce384952b72f6fa47a910`
- Hardware/runtime: Apple M1 shared 16 GiB; PyTorch 2.13 MPS; verified
  internal-SSD checkpoint
- Related records: PW-0045, PW-0115, PW-0116, PW-0117; prospective E5

## Question and mechanism

Before a long weight-space fit, prove that the exact surviving
`identity-basis-mixture-compiled` parameterization, optimizer state, source-tile
loader, and Gate 8 monitor can coexist on the 16 GiB M1 at production layer
dimensions. This is an optimizer-substrate experiment, not a fidelity or
training result.

Instantiate the smallest-memory frozen shape `(r=128,m=32)` for all 256
experts of one canonical `[2048,4096]` projection:

- `A [256,2048,128]` expert factors;
- `B [32,128,4096]` shared bases; and
- `alpha [256,32]` expert coefficients.

Use identity activation and `softmax(alpha)` only as a coefficient
parameterization. Optimize against exact dequantized source-FP8 tiles from
layer 4 gate experts 64 (hot) and 10 (rare). The full production parameter
tensors and Adam state must be allocated even though the bounded preflight
touches only fixed source rows/columns. This tests the memory embodiment and
autograd path without pretending that a tile fit is a trained layer.

## Frozen execution

Use seed `260118`, MPS F32 parameters, experts `[64,10]`, rows `0..31`, columns
`0..127`, Adam learning rate `0.01`, and five steps. Authenticate the checkpoint
verification manifest and exact tensor names/shapes/dtypes. Dequantize the
source tile with its authoritative 128×128 scale block. Record source tile
SHA-256, initial/final loss, every step wall time, MPS current/driver allocated
memory, process footprint/RSS, disk bytes, and complete Gate 8 snapshots.

Add a tiny CPU fixture proving the same module's forward equation and gradient
updates before the MPS process. Write only a small JSON report under external
evidence; do not save the random production parameters or optimizer state.

## Gates

Pass only if:

1. all tensor/checkpoint identities and source tile bytes are hash-bound;
2. forward/loss/backward/Adam steps are finite and final loss is below initial
   loss;
3. all five steps complete and MPS synchronization succeeds;
4. Gate 8 remains above 20% free memory, below 8 GiB process current/peak,
   below 512 MiB swap growth, with no new throttled pages or lost protected
   services; and
5. after parameter/optimizer release and MPS cache clearing, process footprint
   is below 4 GiB.

Passing authorizes a streamed source-weight fitting contract and rank-heavy
memory preflight. It does not establish convergence, reconstruction quality,
activation fidelity, executable quantization, inference wall time, output,
accepted tokens, or TPS. Kill this direct MPS/Adam substrate if it violates
Gate 8; use a block-coordinate/CPU or externally trained artifact path rather
than relaxing host safety.

## Result

The clean `70ef12bcda4182176170c4d44d63acb9fa8910dc` run passed. It
allocated all 83,894,272 F32 parameters for the `(r=128,m=32)` 256-expert
projection plus full Adam state. MPS current allocation rose from 335,610,112
bytes after parameters to at most 1,342,505,216 bytes after optimizer steps;
driver allocation peaked at 2,167,029,760 bytes. After deletion and cache
clearing, current MPS allocation was zero and driver allocation was 2,752,512
bytes.

The authenticated hot/rare real source tile hashes to
`0b10b694f07274bbeba87356f3a45dc430cc2eef29c8550812ae8545365b7d75`.
Five finite Adam steps reduced tile MSE from `4.234154e-5` to `1.238841e-5`, a
70.742% reduction. The first allocation/state-creation step took 1,429.848 ms;
the next four took 111.407–113.334 ms. Complete wall time was 3,449.506 ms.
These are optimizer-tile diagnostics, not layer training or inference timing.

Gate 8 passed with 67% minimum system-free memory, 416,415,744-byte process
peak RSS, 2,408,975,168-byte maximum physical footprint, zero swap growth,
zero throttling, stable services, and a 2,383,596,352-byte post-release
footprint. The raw report
`/Users/chad/Models/mimo-prismwing/evidence/PW-0118/run-001.json` hashes to
`9d96d71f21f68c249b10422ec0fb479ec905874a93a5631c2488b3fc90e53c9c`;
independent analysis hashes to
`25f71aabb3d66f3142c8ff8447c451c61d7f527b79a02638b256916fe0db778e`.
There are zero accepted tokens and no TPS claim.

## Decision

Promote the direct MPS/F32/Adam substrate only for streamed identity-basis
weight fitting. The experiment proves production parameter and optimizer-state
embodiment for the smallest-memory frozen shape, not convergence or fidelity.

Next, freeze a rank-control audit and a streamed source-weight fit. Preflight
the rank-heavy shape separately before allocating it because its expert factor
and Adam state are much larger. Keep source targets tiled; do not stack a full
256-expert F32 layer. Any trained candidate must still be evaluated on all
PW-0116 positions, rare experts, validation, and untouched holdout before a
kernel or full bank is justified.
