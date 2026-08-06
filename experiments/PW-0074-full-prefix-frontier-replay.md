# PW-0074 — Full-prefix frontier replay after layer 11

- Status: complete
- Disposition: promoted localization
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0073 comparison
  `4f400322f1a25bb5469ca35836104d67a920b0f24648343ecfe83efb19238a17`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0071 through PW-0073

## Hypothesis and contract

PW-0073 restores exact accumulated layer-11 state. One production Rust replay
against the immutable PW-0060 oracle is the cheapest way to advance the exact
frontier and locate the next causal boundary.

Repeat the frozen 27-token prefill through all 48 layers with identical
embedding, layer-final, final-norm, logit, route, and weight captures. Bind the
verified checkpoint, revision, fixture, numerical policy, schema, hashes, and
clean commit. Preserve every existing correctness threshold and distinguish
the last bit-exact layer, first actual divergence, and first formal gate failure.

Enforce normative Gate 8 at every phase: fail closed below 20% free memory,
above 8 GiB current/peak RSS, above 4 GiB after declared release, above 512 MiB
swap growth, on any new throttled page, or on start-resident protected-service
loss. Record buffer release, allocator relief, hardware, commit, cache state,
batch 1, concurrency 1, accepted tokens 0, and complete wall time. Preserve a
stopped run as failed evidence.

This cannot count as TPS or alter any hosted, capability, fidelity, cost,
power, safety, or performance threshold.

## Result

The walk completed in 787.242 seconds. Embedding and layers 0–12 are
bit-exact, advancing the exact accumulated frontier by two layers beyond
PW-0071. Layer 13 is the first actual divergence: 21 of 110,592 BF16 values
differ, relative L2 is `1.6284499569784697e-6`, maximum error is `0.015625`,
and equality is 99.9810%. That remains inside the general layer-final
tolerance. Layer 14 remains the first formal layer-final gate failure, at
99.4475% equality, `1.3903618506794364e-5` relative L2, and `0.0625` maximum
error.

Expert sets and order remain exact through layer 18 and first differ at layer
19, position 22. Route-weight error remains below the `5e-7` gate through
layer 12 (`1.4694595318331949e-8`) and first fails at layer 13
(`4.8274204254017405e-6`). These are consequences or co-observations at the
first divergent layer; later errors do not justify downstream repairs.

Every normative Gate 8 stop passed. The streamed layer phases peaked at
744,325,120 bytes RSS and repeatedly released their footprints to roughly
100–172 MB. The bounded LM-head phase peaked at 3,837,788,160 bytes RSS and
ended with a 2,710,264,704-byte footprint, below the 4 GiB post-release stop.
System-free memory stayed at or above 77%; swap growth and new throttled pages
were zero; ChatGPT, WindowServer, the start-resident nxnode process, and
syncthing remained resident. Evidence hashes:

- Rust manifest:
  `6dac27d2fab0eb1af1200fa5eab0f7294e7fe0281b75d9b551eeec1731bf8804`
- Comparison:
  `0dd64f521715c86fea52557168a5101cdaef76421269b6ed6c1b46b964c9ced6`

## Decision

Promote the localization result. The exact accumulated frontier is through
layer 12; layer 13 is the first actual divergence, while layer 14 is only the
first accumulated formal failure. Run the generalized routed-layer trace on
layer 13 from the frozen exact layer-12 input and stop at its first differing
substage. Do not change downstream routing, experts, or acceptance thresholds,
and do not claim throughput from this correctness walk.
