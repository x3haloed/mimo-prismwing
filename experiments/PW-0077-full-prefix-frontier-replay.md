# PW-0077 — Full-prefix frontier replay after layer 13

- Status: complete
- Disposition: promoted localization
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0076 comparison
  `e260595915439eaef8edd2ee0cc4f07950295a3fe209a0acb8909e710ce2f279`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0074 through PW-0076

## Hypothesis and contract

PW-0076 makes layer 13 exact from the frozen exact layer-12 state. One
production Rust replay against the immutable PW-0060 oracle is the cheapest
way to prove the accumulated frontier and locate the next causal boundary.

Repeat the frozen 27-token prefill through all 48 layers with identical
embedding, layer-final, final-norm, logit, route, and weight captures. Bind the
verified checkpoint, revision, fixture, numerical policy, schema, hashes, and
clean commit. Preserve all existing correctness thresholds and distinguish the
last bit-exact layer, first actual divergence, and first formal gate failure.

Enforce normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after release, above 512 MiB swap
growth, on new throttled pages, or on protected-service loss. Record buffer
release, allocator relief, hardware, commit, cache state, batch 1, concurrency
1, accepted tokens 0, and complete wall time. Preserve stopped evidence.

This cannot count as TPS or alter any hosted, capability, fidelity, cost,
power, safety, or performance threshold.

## Result

The walk completed in 797.596 seconds. Embedding and layers 0–13 are bit-exact,
confirming the accumulated repair and advancing the exact frontier through
layer 13. Layer 14 is both the first actual divergence and first formal
layer-final failure: 396 of 110,592 BF16 values differ, equality is 99.6419%,
relative L2 is `9.821666623436154e-6`, and maximum error is `0.0625`.

Route weights remain inside their strict gate through layer 13
(`2.6036071743007483e-8`) and first fail at layer 14
(`0.00027570375320434826`). Expert sets and order remain exact through layer
18 and first differ at layer 19. Later errors are downstream and do not
justify changes beyond layer 14.

Every Gate 8 stop passed. Streamed-layer RSS peaked at 744,439,808 bytes and
phase footprints repeatedly returned near 100–164 MB. The bounded LM-head
phase peaked at 3,945,922,560 bytes RSS and ended with a 2,680,691,648-byte
footprint, below the 4 GiB post-release stop. System-free memory stayed at or
above 82%; swap used decreased during the run, measured swap growth and new
throttled pages were zero, and ChatGPT, WindowServer, nxnode, and syncthing
remained healthy. Evidence hashes:

- Rust manifest:
  `3a35e772a30c94cadae9e0c89f418eb877504342fac3949c96ddc98568712cb3`
- Comparison:
  `60b14d05e68f06c0fe4246cb856aec960f6382090f62affb0fdf1bdb7db518be`

## Decision

Promote the localization result. The accumulated exact frontier is through
layer 13, and layer 14 is the next causal boundary as both first actual and
first formal failure. Run the generalized routed-layer trace on layer 14 from
the frozen exact layer-13 oracle input. Do not change later routing, experts,
or thresholds, and do not claim throughput from this correctness walk.
