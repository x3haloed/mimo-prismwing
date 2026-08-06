# PW-0080 — Full-prefix frontier replay after layer 14

- Status: complete
- Disposition: promoted localization
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0079 comparison
  `a37e5af67dc8cb4f95ee4ca30cd8af30e62ec3266771c6454fed27be95844ece`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0077 through PW-0079

## Hypothesis and contract

PW-0079 makes layer 14 exact from the frozen exact layer-13 state. Repeat the
frozen production 27-token prefill through all 48 layers against the immutable
PW-0060 oracle to prove the accumulated frontier and identify the next causal
boundary.

Capture identical embedding, layer-final, final-norm, logit, route, and weight
artifacts. Bind the verified checkpoint, revision, fixture, numerical policy,
schema, hashes, and clean commit. Preserve every existing correctness threshold
and distinguish the last bit-exact layer, first actual divergence, and first
formal gate failure.

Enforce normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after release, above 512 MiB swap
growth, on new throttled pages, or on protected-service loss. Record buffer
release, allocator relief, hardware, commit, cache state, batch 1, concurrency
1, accepted tokens 0, and complete wall time. Preserve stopped evidence.

This cannot count as TPS or alter any hosted, capability, fidelity, cost,
power, safety, or performance threshold.

## Result

The walk completed in 799.595 seconds. Embedding and layers 0–18 are bit-exact,
advancing the accumulated exact frontier by four additional layers beyond the
repaired layer 14. Layer 19 is both the first actual divergence and first
formal layer-final failure: 190 of 110,592 BF16 values differ, equality is
99.8282%, relative L2 is `3.068771986578986e-5`, and maximum error is `0.25`.

Route weights remain inside their strict gate through layer 18
(`2.7929687451688778e-8`) and first fail at layer 19
(`0.00008033359680170715`). Expert sets/order remain exact through layer 24
and first differ at layer 25. Later errors are downstream and do not justify
changes beyond layer 19.

Every Gate 8 stop passed. Streamed-layer RSS peaked at 708,001,792 bytes and
phase footprints repeatedly returned near 100–161 MB. The bounded LM-head
phase peaked at 4,169,187,328 bytes RSS and ended with a 2,903,563,136-byte
footprint, below the 4 GiB post-release stop. System-free memory stayed at or
above 82%; swap growth and new throttled pages were zero; ChatGPT,
WindowServer, nxnode, and syncthing remained healthy. Evidence hashes:

- Rust manifest:
  `f5e482ebbd43f4f3febd450c9afbdfb198226617651bef95799b980c56f87fab`
- Comparison:
  `eb2f578b983a6be8befc29dc2724607d33fa81ec6cc4a77311dda1ad8a7d02c2`

## Decision

Promote the localization result. The exact accumulated frontier is through
layer 18; layer 19 is the next causal boundary as both the first actual and
formal failure. Run the generalized routed-layer trace on layer 19 from the
frozen exact layer-18 oracle input. Do not change later routing, experts, or
thresholds, and do not claim throughput from this correctness walk.
