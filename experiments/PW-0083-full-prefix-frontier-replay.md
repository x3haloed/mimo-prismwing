# PW-0083 — Full-prefix frontier replay after layer 19

- Status: complete
- Disposition: promoted localization
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0082 comparison
  `90858655eda93256a8bb5abae28acf9aafe1faa8a2da457445f42d313b832bc5`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0080 through PW-0082

## Hypothesis and contract

PW-0082 makes layer 19 exact from frozen exact layer 18. Repeat the frozen
production 27-token prefill through all 48 layers against the immutable oracle
to prove the accumulated frontier and identify the next causal boundary.

Capture identical embedding, layer-final, final-norm, logit, route, and weight
artifacts. Bind checkpoint, revision, fixture, numerical policy, schema,
hashes, and clean commit. Preserve every correctness threshold and distinguish
the last bit-exact layer, first actual divergence, and first formal failure.

Enforce normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after release, above 512 MiB swap
growth, on new throttled pages, or on protected-service loss. Record release,
allocator relief, hardware, commit, cache state, batch 1, concurrency 1,
accepted tokens 0, and wall time. Preserve stopped evidence.

This cannot count as TPS or alter any hosted, capability, fidelity, cost,
power, safety, or performance threshold.

## Result

The walk completed in 800.724 seconds. Embedding and layers 0–28 are bit-exact,
advancing the accumulated exact frontier by nine additional layers beyond the
repaired layer 19. Layer 29 is both the first actual divergence and first
formal layer-final failure: 20 of 110,592 BF16 values differ, equality is
99.9819%, relative L2 is `6.250610756414843e-6`, and maximum error is `0.0625`.

Route weights remain inside their strict gate through layer 28
(`1.4548492410781932e-8`) and first fail at layer 29
(`6.262503280618503e-6`). Expert sets and order remain exact through layer 46;
the first expert-set mismatch is position 24 at layer 47. Later errors are
downstream and do not justify changes beyond layer 29.

Every Gate 8 stop passed. Streamed-layer RSS peaked at 746,176,512 bytes and
phase footprints repeatedly returned near 95–166 MB. The bounded LM-head phase
peaked at 4,170,842,112 bytes RSS and ended with a 2,908,888,064-byte footprint,
below the 4 GiB post-release stop. System-free memory stayed at or above 72%;
swap use decreased by 8,388,608 bytes, new throttled pages were zero, and
ChatGPT, WindowServer, nxnode, and syncthing remained healthy. Evidence hashes:

- Rust manifest:
  `f0c56eea9629698e4ad947f7292731ebfbc036dda2339aa95f0ecc7618b7bbbd`
- Comparison:
  `c8c6b94313aa780fe1fb1d728529d8fa903e06c4182404c2e096247b2a40c75f`

## Decision

Promote the localization result. The exact accumulated frontier is through
layer 28; layer 29 is the next causal boundary as both the first actual and
formal failure. Run the generalized routed-layer trace on layer 29 from the
frozen exact layer-28 oracle input. Do not change later routing, experts, or
thresholds, and do not claim throughput from this correctness walk.
