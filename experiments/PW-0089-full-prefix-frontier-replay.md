# PW-0089 — Full-prefix frontier replay after layer 34

- Status: complete
- Disposition: promoted localization
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract precedes execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; frozen PW-0060 oracle run 002
  `081550060338070eaa00730877065d2752824c589c22f74eaa7e921448c61573`;
  PW-0088 comparison
  `967a7f9d0ee0c0b004c8b1b365b68cd1ff2c4cca2c280d93318de4950d1274aa`
- Hardware/runtime: Apple M1 shared 16 GiB host, verified SSD checkpoint,
  production Rust trace, existing frozen PyTorch oracle
- Related records: PW-0060, PW-0086 through PW-0088

## Hypothesis and contract

PW-0088 makes layer 34 exact from frozen exact layer 33. Repeat the frozen
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

The walk completed in 799.549 seconds. Embedding, all 48 transformer layers,
final RMSNorm, every route weight, and every expert set/order are bit-exact.
This closes the entire accumulated transformer prefix rather than merely
advancing to another layer boundary.

The sole remaining local correctness mismatch is the LM-head projection: 45
of 152,576 F32 last-token logits differ, equality is 99.9705%, relative L2 is
`5.247888621759728e-5`, and maximum error is `0.03125`. Both hosted-chosen
token logits captured by the comparator are exact (`3.609375` for token 0 and
`1.0` for token 9707), but the complete logit vector fails its existing gate.
No layer, route, or expert change is justified.

Every Gate 8 stop passed. Streamed-layer RSS peaked at 740,425,728 bytes and
phase footprints repeatedly returned near 100–163 MB. The bounded LM-head phase
peaked at 3,938,123,776 bytes RSS and ended with a 2,679,627,008-byte footprint,
below the 4 GiB post-release stop. System-free memory stayed at or above 81%;
swap growth and new throttled pages were zero; ChatGPT, WindowServer, nxnode,
and syncthing remained healthy. Evidence hashes:

- Rust manifest:
  `0e8b14621a5e3e3715c8136bbef53ae94da674df9a0e9435e3ae881fb5d11f80`
- Comparison:
  `6f00f95147aebf4f7c941893fe4aa1224f9874e74934cd8be6d42fc634cc82b8`

## Decision

Promote the localization result. The complete transformer and final norm are
bit-exact; the LM head is the next and only observed local semantic boundary.
Freeze exact final-norm input and discriminate LM-head FP8/GEMV operation
ordering before changing arithmetic. Do not alter any transformer layer,
routing, expert, threshold, or hosted acceptance contract, and do not claim
throughput from this correctness walk.
