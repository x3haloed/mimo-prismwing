# PW-0314 — Full-checkpoint layer-4 target-native K4 control

- Status: in progress
- Disposition: pending
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
