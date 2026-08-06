# PW-0093 — Whole-sequence versus retained-cache decode

- Status: complete
- Disposition: rejected
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: oracle implementation and clean execution at
  `a42339e`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0092 run 001
  `18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25`;
  checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  independent PyTorch whole-sequence oracle versus production Rust caches
- Related records: PW-0060, PW-0091, PW-0092

## Hypothesis and mechanism

PW-0092's second token is causally driven by retained per-layer K/V, but its
arithmetic authority and cache authority still live in the same Rust process.
Evaluate the exact 28-token prefix formed by appending PW-0092's accepted token
264 to its frozen 27-token prompt through the independent PyTorch source oracle
as one uncached causal sequence. If K/V positions, RoPE offsets, attention
windows, route state, or cache updates are wrong, its final distribution or
position-27 routes will diverge from the retained-cache endpoint.

Generalize only row count and explicit token identity in the existing oracle;
preserve its default 27-token behavior and frozen semantics. Add a deterministic
fixture test for row/token authority before execution. Do not change production
arithmetic, checkpoint data, prompt serialization, sampling, or thresholds.

## Gates

The 28-token PyTorch oracle must preserve the first 27 route rows exactly from
PW-0092 step one and match route row 28 exactly to PW-0092 step two in all 47
routed layers. Expert order is authoritative. Route-weight comparison uses the
existing source threshold of `5e-7` per expert. The complete 152,576-value final
logit vector must be byte-identical to PW-0092 step two; any mismatch localizes
the retained-cache path rather than being averaged away. All values must be
finite and every identity/hash/shape check must fail closed.

Enforce normative Gate 8 at oracle open and every captured layer/final boundary:
minimum system memory-free 20%, process peak/current at most 8 GiB,
post-release footprint at most 4 GiB, swap growth at most 512 MiB, no new
throttled pages, explicit release relief, and continued health of ChatGPT,
WindowServer, `nxnode`, and Syncthing. Preserve stopped evidence. Run one full
oracle only; a second is justified only if the first passes but comparison is
nondeterministic or ambiguous.

This is a correctness experiment. Its wall time, storage traffic, and memory
are diagnostic and cannot become accepted TPS or promote a performance default.

## Result

The single authorized oracle completed safely in 702,630.189 ms. Its manifest
is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0093/oracle-001/manifest.json`
with SHA-256
`f143a6c9ee526eaddd40e809ffa18e20a3eb1cbc9e0b5d0af2a86ba80757b596`.
The committed token-identity fixture hash is
`d47667608f8038c78f8b65e7ec307c8d759d1d1dddeb9687d2914fe1932ce606`.
The oracle evaluated exactly the PW-0092 27-token prefix followed by token 264
as one 28-row causal sequence and produced all 51 declared captures plus 48
route traces.

The predeclared exact gate failed. PyTorch whole-sequence final logits hash to
`05ce9d5cdbcf55aa70f56ad20a9885263e4a0ddcbd1e1d3985b43cebfdcc4050`;
packing PW-0092's retained-cache step-two logits as little-endian F32 hashes to
`e86670ade50a8c02be5451f9233a65e6b982e80d09f8fd38b41c2d8e3ea2526`.
Of 152,576 logits, 12,711 are exact and 139,865 differ: 8.3309% equality,
`0.0246957` relative L2, and `0.5` maximum absolute error. Both paths still
choose token 13. The oracle top logit is 14.375; retained-cache Rust reports
14.1875.

Route localization is confined initially to the appended position. Mapped by
expert identity, the first route-weight error over `5e-7` appears at layer 1,
position 27 (`3.86e-6`). Unsorted expert order first differs at layer 3,
position 27. The first expert-set difference appears at layer 11, position 27;
four route rows eventually differ in expert set, 33 differ in order, and 43
exceed the route-weight threshold. Maximum weight error by common expert is
`0.0143716`. The Rust evidence still proves every cache length is 27 after
prefill and 28 after incremental decode, but it does not clear this independent
whole-sequence comparison.

Gate 8 passed. The oracle retained at least 63% system memory-free pressure,
peaked at 3,879,370,752 bytes RSS, had a maximum sampled current resident size
of 514,572,288 bytes, and ended at 219,136,000 bytes. Swap did not grow, no new
throttled pages appeared, and ChatGPT, WindowServer, both `nxnode` processes,
and both Syncthing processes remained resident at the final boundary.

## Decision

Reject the hypothesis that a 28-row whole-sequence PyTorch pass must be
byte-identical to a one-row retained-cache Rust step. Preserve the mismatch;
do not relax the gate and do not call it cache corruption yet. Matrix backends
can select row-count-dependent reduction topologies, and PW-0091 established
exactness for 27-row PyTorch versus 27-row Rust, not 28-row PyTorch versus
one-row Rust.

Open a separate localization experiment that evaluates the same 28-token
prefix through the Rust whole-sequence trace. Comparing PyTorch-28 to Rust-28
separates source/batch-topology parity; comparing Rust-28's last row to Rust
27+1 isolates retained-cache state under one arithmetic authority. No
throughput-model constant changes: this experiment accepted zero tokens and
measured a correctness oracle, not a decode default.
