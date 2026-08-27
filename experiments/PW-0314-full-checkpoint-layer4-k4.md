# PW-0314 — Full-checkpoint layer-4 target-native K4 control

- Status: complete
- Disposition: conditional
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0313

## Question

Can `m1-native-k4-v1` authenticate the installed full checkpoint, construct a
new-layer expert selected by a real route, repeat locally, and preserve routed
quality across train, validation, and pilot-holdout placements?

## Hypothesis and mechanism

PW-0313 proves only two layer-28 identities from the recovered source
mini-checkpoint. A useful bank path requires construction directly from the
installed full checkpoint and must survive a distinct layer and real activation
distribution.

Layer 4 expert 64 is the strongest cheap control in the frozen PW-0116 corpus:
it is the most frequently selected identity, with 181 placements across the
224-position trace: 106 train, 56 validation, and 19 pilot holdout. Its route
weight ranges from `0.0400427` to `0.14218338`, so the control includes both
low- and high-weight placements.

## Authorities

- checkpoint revision `63651580ca774f8504f676040460aed3e1244ac1`;
- installed checkpoint receipt
  `/Users/chad/Models/mimo-prismwing/evidence/PW-0049/checkpoint-verification.json`,
  SHA-256
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
- source index SHA-256
  `f2e1774c9acf9a62338b68c144e6fc7a66495e59f2e64b3078c1b7ef5a196816`;
- all six expert-64 tensors map to the receipt-bound
  `model_pp0_ep2_shard0.safetensors`, SHA-256
  `70639d2d3ad4bd80a3b3843632e17a5089baa3b2ac5565e571fb5ad7bafb0be0`;
- PW-0116 corpus manifest SHA-256
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
- the PW-0311 QTIP, calibration-atlas, recipe, seed, codebook, TLUT, and Gate 8
  authorities unchanged.

The receipt is trusted as the completed installation proof instead of
rescanning the 34.4-GB shard. Runtime preflight must still hash the 7-MB index,
validate the receipt and revision, require exact shard path/size/inode/mtime,
and verify that every requested tensor and scale maps to that shard. Darwin
device numbers are recorded but not identity-gated because APFS device IDs can
change across reboot while inode, size, and nanosecond mtime remain stable.

## Protocol and gates

1. Add a separate PW-0314 constructor; do not weaken or repurpose the exact
   PW-0311/0312 reproducer or the layer-28 PW-0313 evaluator.
2. Load layer 4 expert 64 gate/up/down source-FP8 weights from the full installed
   checkpoint and construct a complete `m1-native-k4-v1` artifact with the
   unchanged QTIP recipe.
3. Independently decode every serialized projection and require relative L2 at
   most `2e-5` versus the in-process candidate.
4. Recompute source expert outputs for all 181 selected placements. Require
   every F32 bit to match the corresponding rows in PW-0116
   `layer_04_expert_down.f32`; this verifies input, ordering, dynamic-FP8, BF16,
   and source-tensor semantics before judging K4.
5. Replace only expert 64's source output at each selected placement, using its
   frozen route weight. Preserve all other source expert contributions from the
   captured routed output and apply the published BF16 boundary.
6. Require routed-output and layer-final relative L2 below `0.01` overall and
   separately for train `[0,112)`, validation `[112,168)`, and pilot holdout
   `[168,224)`. Require maximum per-row relative L2 below `0.05` on every slice.
7. Repeat construction in a fresh process from the same clean pushed commit.
   Require every candidate hash, packed state, manifest, fixture, and referenced
   local payload byte to match exactly, and require identical semantic metrics.
8. Record construction wall, I/O, RSS, physical footprint, release footprint,
   memory-free floor, swap/throttle growth, protected services, hardware,
   software, commit, batch size one, concurrency one, and zero accepted tokens.

## Decision rule

- If both runs are byte-identical and all source, route, partition, semantic,
  and safety gates pass, authorize a bounded multi-identity layer-4 bank test.
- If source replay fails, stop: the corpus/checkpoint semantics are not a valid
  answer key for this construction.
- If repeatability fails, reject target-native bank construction.
- If K4 routed quality fails any partition, reject this identity and require a
  different identity-selection or representation policy before scaling.

## Claims excluded

- source-exact or L1 weights;
- layer-4 identities other than expert 64;
- other layers, arbitrary routes, or a complete bank;
- hosted-reference, multimodal, long-context, or capability equivalence;
- ordinary endpoint execution or accepted-token TPS;
- Prismwing-2, 34.3 TPS, or Prismwing 50 completion.

## Execution and evidence

The constructor was committed and pushed at clean commit
`33ea64171f22fdfd8cf87b813c2a684ae595edd9`. Two fresh processes produced:

| Run | Status | Wall seconds | Peak RSS bytes | Report SHA-256 |
| --- | --- | ---: | ---: | --- |
| 001 | qualified | 502.164624 | 1,532,395,520 | `3adba8e673f75e949e5f6b37d5b0b407cbf70b51f02a20837f1d7a0f86a59cb6` |
| 002 | qualified | 501.933432 | 1,530,118,144 | `654da3f02e96a89b3fce23c3c68e5b6cafcf64444fdf9785cae596a06e4dc6dc` |

Both runs produce the same 33-file, 30,005,932-byte deterministic tree. Every
candidate-array, packed-trellis, projection-manifest, fixture, and referenced
payload hash matches. The semantic reports are identical. Only diagnostic
quantization seconds and host counters differ.

## Results

The installed checkpoint receipt, revision, index, tensor-to-shard mappings,
shard inode/size/mtime, QTIP recipe, calibration atlas, codebook, TLUT, and
PW-0116 corpus preflights pass in both runs. The source expert reproduces every
F32 bit of all 181 captured expert-64 outputs. Expert-major schedule
reconstruction reproduces all 224 source routed rows and final rows bit for
bit.

Each serialized K4 projection independently decodes to the in-process candidate
with zero relative L2. Across the 181 selected placements, candidate expert
output relative L2 versus source is `0.006314151`; maximum row relative L2 is
`0.018526057`.

Replacing only expert 64 yields these routed-output results:

| Slice | Aggregate relative L2 | Maximum row relative L2 |
| --- | ---: | ---: |
| overall | 0.000945201 | 0.005299529 |
| train | 0.000943151 | 0.004591835 |
| validation | 0.002181414 | 0.004356384 |
| pilot holdout | 0.000918785 | 0.005299529 |

Layer-final relative L2 is `0.000984831` overall, `0.000983206` train,
`0.001334509` validation, and `0.000626691` pilot holdout. Its worst row is
`0.002801227`. Every aggregate remains below `0.01` and every row remains below
`0.05`.

Gate 8 passes both runs. Minimum free memory is 61%, maximum process footprint
is 1,728,042,176 bytes, maximum peak RSS is 1,532,395,520 bytes, release
footprints are 387,911,616 and 393,383,936 bytes, and swap growth and new
throttling are zero. Protected services remain healthy.

## Decision

Conditionally qualify `m1-native-k4-v1` for layer-4 expert 64 under the frozen
PW-0116 corpus. This closes the full-checkpoint receipt and second-layer
construction gates and authorizes a bounded multi-identity layer-4 bank test.

Do not generalize this result to other identities: PW-0313 already proves that
semantic qualification is identity-local. The next experiment must construct
and gate each added identity, accumulate their routed error across the same
partitions, and preserve expert-64 as an immutable control. This experiment
accepts zero tokens and changes no throughput-model constant or runtime default.
