# PW-0095 — Independent PyTorch cached decode

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: clean oracle implementation and execution at
  `ff4ebc2`
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0092 run 001
  `18c3ccde4a8645d9ea46d0091f877eebe256ca2c7d82c34e771f5f4114bb5f25`;
  PW-0094 PyTorch-28/Rust-28 comparison
  `1c98bb1ce5086d519ce2a3d63079f7dda11c97f5abc3cb9f91fc3fc7c9960b00`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  independent PyTorch prefill plus retained per-layer K/V
- Related records: PW-0060, PW-0091 through PW-0094

## Hypothesis and mechanism

The PW-0092 Rust endpoint's second distribution differs from the exact 28-row
whole-sequence distribution because one-row source matrix reductions differ
from 28-row reductions, not because its retained K/V is corrupt. Build a
readable independent PyTorch cache oracle: evaluate the frozen 27-token prompt
through all layers while retaining each layer's source K/V, then embed only
token 264 and propagate that one row through the 48 retained caches. Use the
same pinned source operations, BF16 boundaries, dynamic FP8 activation
quantization, RoPE position 27, global/SWA policies, router, and LM head as the
cleared whole-sequence oracle.

Do not import, call, or translate Rust cache state. The only shared inputs are
the verified checkpoint, committed token fixture, and source semantics. Add
deterministic tests for cache append position, global/SWA visible ranges,
shape/head authority, and corruption rejection before execution.

## Gates

The PyTorch prefill route rows must reproduce the cleared 27-row oracle. After
one incremental row, every cache must contain 28 positions with the pinned 4/8
KV-head schedule. The incremental row's expert sets/order and route weights in
all 47 routed layers must match PW-0092 step two; route-weight error may not
exceed the existing `5e-7` source threshold. Its complete 152,576-value F32
logit vector must be byte-identical to PW-0092 step two and choose token 13.
All identities, hashes, shapes, positions, values, and evidence schemas fail
closed. Any mismatch is preserved and localized before production changes.

Enforce normative Gate 8 at checkpoint open, every prefill layer, every
incremental layer, LM head, and capture boundary: minimum memory-free 20%,
peak/current process memory at most 8 GiB, post-release resident memory at most
4 GiB, swap growth at most 512 MiB, no new throttled pages, allocator relief,
and continued health of ChatGPT, WindowServer, `nxnode`, and Syncthing.
Preserve stopped evidence. One independent cached oracle is authorized; a new
Rust trace is allowed only if the comparison fails and layer-state localization
is necessary.

This is a correctness experiment with `accepted_tokens=0`; no wall time or
component rate is accepted TPS or a promoted performance default.

## Result

The independent oracle completed in 804,493.062 ms. Its manifest is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0095/oracle-001/manifest.json`
with SHA-256
`75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8`.
It used only the verified checkpoint and pinned source operations: no Rust K/V,
layer state, routes, or logits were imported.

All gates clear. The 27-row PyTorch prefill and PW-0092 Rust step one have
identical expert sets and order in every routed layer; maximum route-weight
error by expert is `2.97e-8`. The independent one-row cache pass appends at
absolute position 27 and leaves all 48 caches at 28 positions. The nine full
attention layers retain four KV heads and the 39 SWA layers retain eight. Its
incremental expert sets and order exactly match PW-0092 step two, with maximum
route-weight error `2.08e-7`, below the unchanged `5e-7` source gate.

The complete incremental F32 logit vectors are byte-identical. Both the
PyTorch capture and packed PW-0092 Rust step-two logits hash to
`e86670ade50a8c02be5451f9233a65e6b982e80d09f8fd38b41c2d8e3ea2526a`
and choose token 13. This independently validates the real prompt -> retained
K/V -> one-row attention/MLP -> route -> LM-head -> accepted-token path.

Gate 8 passed with at least 71% system memory-free pressure, peak RSS
4,170,235,904 bytes, maximum sampled current resident size 513,703,936 bytes,
and final resident size 420,560,896 bytes. Swap grew by only 398,459 bytes,
well below 512 MiB; no new throttled pages appeared. ChatGPT, WindowServer,
`nxnode`, and both Syncthing processes remained resident at the final boundary.

## Decision

Promote PW-0092's retained-cache text semantics as independently source-exact
for this 27+1 walking slice. PW-0093's rejected 28-row-versus-one-row byte
identity premise is explained by row-count-dependent source matrix arithmetic,
not K/V corruption. Preserve both results and do not require whole-sequence
byte identity as an incremental cache gate; compare equal execution shapes.

Close incremental-state localization and move to one-token profiling and
physical work compression. The measured 17.208 GB logical source path, 1,179
FP8 matrix expansions, and 158.5--158.6 second Rust token remain the active
bottleneck evidence. No throughput-model constant changes are needed because
this oracle accepted zero tokens and confirmed the existing PW-0092 constant.
