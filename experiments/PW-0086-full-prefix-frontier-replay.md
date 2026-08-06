# PW-0086 — Full-prefix frontier replay after layer 29

- Status: complete
- Disposition: promoted localization
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0085 comparison
  `716fa337cde3e90de10342f46afafd802d5b78f5b73a2e82e7c90ef9462da5b3`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0083 through PW-0085

## Hypothesis and contract

PW-0085 makes layer 29 exact from frozen exact layer 28. Repeat the frozen
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

The walk completed in 781.393 seconds. Embedding and layers 0–33 are bit-exact,
advancing the accumulated exact frontier by four layers beyond repaired layer
29. Layer 34 is the first actual divergence: only 6 of 110,592 BF16 values
differ, equality is 99.9946%, relative L2 is `4.4345876253413467e-7`, and
maximum error is `0.0078125`. Layer 36 is merely the first formal layer-final
failure after that error propagates.

Route weights remain inside their strict gate through layer 33 and first fail
at layer 34 (`6.902825546273306e-6`). Expert sets/order remain exact through
layer 43 and first differ at position 25 in layer 44. Later errors are
downstream and do not justify changes beyond layer 34.

Every Gate 8 stop passed. Streamed-layer RSS peaked at 744,275,968 bytes and
phase footprints repeatedly returned near 100–160 MB. The bounded LM-head phase
peaked at 3,942,072,320 bytes RSS and ended with a 2,674,105,024-byte footprint,
below the 4 GiB post-release stop. System-free memory stayed at 83%; swap
growth and new throttled pages were zero; ChatGPT, WindowServer, nxnode, and
syncthing remained healthy. Evidence hashes:

- Rust manifest:
  `2fd3d81b921e1c31989a6b2353d648a2363216bf64857207211c735a50e27c72`
- Comparison:
  `d23e411ab91712636d45553463ef162652403a60d3ee76f9bb835c007dce001f`

## Decision

Promote the localization result. The exact accumulated frontier is through
layer 33; layer 34 is the next causal boundary even though its six-value delta
does not formally fail the final-state gate until layer 36. Run the generalized
routed-layer trace on layer 34 from the frozen exact layer-33 oracle input. Do
not change later routing, experts, or thresholds, and do not claim throughput.
