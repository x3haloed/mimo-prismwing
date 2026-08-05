# PW-0013 — Lossless real expert container

- Status: complete
- Disposition: production
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `1feedcd`; dirty container implementation
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; source shard
  `fd89388271eac237e06ace68a832156357b42f85820856afee24da7bb36d9dcc`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Rust 1.96.0; USB platter source and
  artifact; no material memory pressure
- Related records: PW-0002, PW-0004, PW-0005

## Hypothesis and mechanism

A small aligned container can extract actual expert tensors from the
partitioned source checkpoint without changing a byte, while carrying enough
identity and layout metadata for runtime loading to fail closed.

## Contract

Target-faithful L0 storage operation. The source shard must match the model
lock. Creation must refuse overwrite, publish atomically, preserve dtype/shape/
source offsets, hash the complete source file and every copied tensor, align
payloads to 64 bytes, and verify the completed artifact before success.

The deterministic tiny test must round-trip multiple tensors and detect a
one-byte payload mutation. The real test must use an actual routed-expert
projection, not MTP as a shape proxy.

## Baseline and candidate

Source tensor pair:

- `model.layers.43.mlp.experts.32.down_proj.weight`, source FP8 shape
  4,096×2,048, 8,388,608 bytes;
- matching f32 `weight_scale_inv`, shape 32×16, 2,048 bytes.

Command:

```sh
cargo run --release -- repack \
  /Volumes/Elements/mimo-prismwing/checkpoints/MiMo-V2.5-63651580/model_pp0_ep1_shard1.safetensors \
  /Volumes/Elements/mimo-prismwing/artifacts/PW-0013/layer43-expert32-down.pwexpert \
  model.layers.43.mlp.experts.32.down_proj.weight \
  model.layers.43.mlp.experts.32.down_proj.weight_scale_inv
```

## Isolated attribution

The 3,490,619,024-byte source shard was streamed through SHA-256 during
installation. The output is 8,391,424 bytes: 16-byte binary prefix, 719-byte
JSON header, zero padding to the next 64-byte boundary, and two aligned tensor
payloads.

Storage hashing/repacking time is not reported as inference throughput. The USB
platter source made complete-source hashing take roughly two minutes; runtime
will consume the bounded container payload rather than rescan the source shard.

## End-to-end result

The `repack` command produced the artifact and verified it before publication.
The independent `verify-container` command then passed on the published path.
No inference or TPS claim is made.

## Correctness result

- Artifact SHA-256:
  `fd91204e7a87e86574445e480e868856e3e80b6826d81ff8acdb6be0fd4f5009`
- Weight payload SHA-256:
  `75706d115d6706950c6a6b147959ab64cb8bb4cfc0004bad467ace9b413f7495`
- Scale payload SHA-256:
  `db951c18ed0788b74171ce09bc523689055f82dc5787bc21d85569d2b328d06e`

The container's complete-source hash equals `spec/model.lock.json`. Unit tests
also prove sorted multi-tensor round trip, no-clobber behavior, and detection of
a one-byte payload change.

## Decision

Promote `PWEXPRT1` as the initial lossless expert-container format. This closes
the format and real single-projection round-trip slice of M1, not the full M1
milestone: gate/up tensors are in the still-downloading paired shard, a complete
expert container and executable full expert remain next.
