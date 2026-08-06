# PW-0071 — Full-prefix frontier replay after layer 7

- Status: complete
- Disposition: promoted localization
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0070 comparison
  `62ea2df8ba4494959e5fdb9544af7ac758b31e54a73b7508bdc17212dd14d472`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0064, PW-0067 through PW-0070

## Hypothesis and contract

PW-0070 restores exact accumulated layer-7 state. One production Rust replay
against the immutable PW-0060 oracle is now the cheapest way to advance the
exact frontier and find the next causal boundary without regenerating an
oracle or speculating about downstream arithmetic.

Repeat the frozen 27-token prefill through all 48 layers with identical
embedding, layer-final, final-norm, logit, route, and weight captures. Bind the
verified checkpoint, revision, fixture, numerical policy, schema, hashes, and
clean commit. Keep the BF16 `5e-4` relative-L2, `2e-2` maximum-error, 99%
equality gates; final `4e-5`/`3e-6` gates; exact expert sets; and `5e-7`
route-weight gate. Stop semantic speculation at the first failing layer.

This is the first full walk under the normative Gate 8 shared-host policy.
At every phase, record and enforce current footprint, peak RSS, system-free
memory, swap growth, throttling, allocator relief, buffer/page release, and
protected-service health. Fail closed below 20% free memory, above 8 GiB
current/peak, above 4 GiB after declared release, above 512 MiB swap growth, on
any new throttled page, or on start-resident service loss. Preserve a stopped
run as failed evidence.

Record cold/warm state as observed, batch 1, concurrency 1, accepted tokens 0,
wall time, bytes, hardware, and commit. This cannot count as TPS or alter any
hosted, capability, fidelity, cost, power, or performance threshold.

## Result

The walk completed in 779.492 seconds. Embedding and layers 0–10 are bit-exact,
advancing the exact accumulated frontier by three layers. Layer 11 is the first
actual divergence: five of 110,592 BF16 values differ, relative L2 is
`8.828103192233292e-7`, maximum error is `0.015625`, and equality is 99.9955%.
That remains inside the layer-final tolerance. Layers 12 and 13 also remain
inside the layer-final tolerance but are not exact. Layer 14 is the first formal
layer-final gate failure, at 99.4466% equality, `1.3925652279676947e-5`
relative L2, and `0.0625` maximum error.

Expert sets and order remain exact through layer 18. Route-weight error remains
below the `5e-7` gate through layer 11 (`2.8038110733152877e-7`) and first fails
at layer 12 (`3.169771842947977e-6`). Later differences are downstream and do
not justify changes beyond the first actual arithmetic divergence at layer 11.

This first walk under normative Gate 8 passed every stop. The LM-head phase
peaked at 4,168,138,752 bytes RSS, below 8 GiB, and ended with a
2,904,022,848-byte footprint, below 4 GiB. System-free memory stayed at or above
80%; swap growth and new throttled pages were zero; ChatGPT, WindowServer,
nxnode, and syncthing remained resident. Layer-phase footprints repeatedly
returned near 150 MB before the bounded LM-head allocation. Evidence hashes:

- Rust manifest:
  `bc5d8238ca86b6910bdb827501ea0c8bd4d9fa288137a0540893ea419d0dbe47`
- Comparison:
  `744fad6a7ba4b9ea883c5f53eda2f4fafa67569e82718a65ec9cbdaac526a9c4`

## Decision

Promote the localization result. The exact frontier is through layer 10; layer
11 is the first actual divergence, while layer 14 is only the first accumulated
layer-final threshold failure. Extend the existing routed-layer diagnostic to
layer 11 and localize its first substage before changing arithmetic or running
another full walk. No throughput, hosted, fidelity, or safety threshold changes.
