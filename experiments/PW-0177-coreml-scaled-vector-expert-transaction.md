# PW-0177 — Core ML scaled-vector expert transaction

- Status: complete
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; authenticated PW-0116
  routed-activation corpus; layer-46 hot expert 28
- Execution mode: shadow L3 lossy candidate beside a target-faithful source
  expert control; candidate state never enters the model
- Related records: PW-0045, PW-0100, PW-0116, PW-0129, PW-0147

## Realization and compression contract

Capability: make one complete real MiMo expert execute from an explicit vector
codebook representation on the onboard M1 through Core ML, without expanding a
dense candidate artifact in the application, and expose storage, load, cold,
warm, and held-out numerical behavior.

Envelope: batch one, concurrency one, one layer-46 expert, one routed input per
prediction, validation positions `112..167`, Core ML `MLProgram`, macOS 15
minimum deployment target, and `ComputeUnit.ALL`. The pilot holdout at
`168..223` remains sealed. This is a component falsification, not accepted TPS.

Accepted revision: normalize each projection row by its maximum absolute value,
retain its FP16 output scale as a distinct multiply, and k-means-palettize the
normalized matrix with an 8-bit index for every two-weight vector, grouped over
16 output channels. This is four effective index bits per source scalar before
metadata. The rewrite is algebraically exact before FP16 conversion and lossy
only at conversion/palettization boundaries.

Exclusions: no external or sidecar hardware; no per-channel-scale Core ML
compression operator; no source checkpoint mutation; no pilot holdout; no
layer transaction, expert bank, routing, full token, endpoint, or TPS claim.

Central truth: authenticated source weights and routed inputs cause source BF16
expert outputs; the separately compiled candidate causes observable package
bytes, compilation/load latency, cold/warm execution latency, and validation
error on the same inputs.

## Contract

1. Authenticate the checkpoint verification receipt and PW-0116 corpus by
   frozen SHA-256, revision, capture hashes, shapes, and dtype. Fail closed.
2. Before the real run, add a deterministic tiny fixture proving that row
   normalization plus post-linear output scaling reconstructs the unnormalized
   linear projection, including an all-zero row.
3. Convert both the uncompressed scaled FP16 control and the vector candidate.
   Require three linear operations and explicit scale multiplications in the
   control graph. Preserve both external packages outside Git and hash every
   package file into an aggregate manifest digest.
4. Use only the 56 validation placements of layer-46 expert 28. Compare both
   candidates against PW-0116's source BF16 expert-down capture. Do not inspect
   or report the pilot holdout.
5. Record package bytes, package/source and package/FP16 ratios, model-load
   latency, first prediction, seven warm-up predictions, 49 measured warm
   predictions, warm median/p95, batch size, concurrency, accepted tokens,
   `A`, `U`, hardware, OS, Core ML Tools version, and implementation identity.
6. The FP16 control must have validation relative L2 at most 5% and maximum-row
   relative L2 at most 6%; otherwise the run is invalid.
7. The candidate numerical gate is validation relative L2 at most 5% and
   maximum-row relative L2 at most 7%. Its component physical gate is package
   bytes at most 35% of FP16, warm median at most 2.5 ms, and warm p95 at most
   3.5 ms.
8. Independently reject the one-model-per-expert endpoint topology unless model
   load plus first prediction is at most 2.5 ms. A resident arithmetic pass
   cannot conceal route-dependent model acquisition.
9. Promote only if numerical and physical gates pass. Promotion authorizes a
   separately predeclared resident multi-expert layer transaction with dynamic
   routing and complete layer-output parity. It does not authorize a runtime
   default or endpoint claim.
10. If correctness fails, kill this exact vector-code rate and fitting rule. If
    the endpoint-topology gate fails, kill per-expert Core ML package switching
    even if resident arithmetic passes. Preserve any independent positive
    component result.

## Result

Earlier interactive compiler probes are not adjudicated evidence. `run-001`
failed closed before compression because the Core ML adapter passed a `Path`
object rather than its string form; it produced no report or numerical
observation and is preserved as an invalid adapter attempt.

The repaired `run-002` is valid. Its external report hashes to
`911f1db4b0c7d3f0af068a1f55acc78c8a7b3993ae3cea228bee91adc1ad756c`.
The FP16 control graph retains three linear and four multiply operations and
passes the source-validation gate at `0.036504` relative L2 and `0.046242`
maximum-row relative L2. Its package is 50,364,779 bytes, warm median is
1.3651 ms, and warm p95 is 2.3983 ms.

The vector candidate independently passes the component physical gate. Its
13,140,830-byte package is `0.260913` of the FP16 package and `0.522170` of
the source FP8 expert representation. Warm median is 1.4222 ms and warm p95
is 2.2138 ms on the onboard M1 at batch one and concurrency one. This proves
that Core ML can execute this compressed representation without a warm
arithmetic penalty; it is not accepted-token throughput.

Reject the exact candidate numerically. Validation relative L2 is `0.159577`
and maximum-row relative L2 is `0.180525`, versus frozen limits of `0.050000`
and `0.070000`. The pilot holdout remains sealed. Also reject route-time
one-model-per-expert Core ML switching: candidate model load is 503.257 ms and
first prediction is 7.109 ms, totaling 510.365 ms versus the 2.5-ms topology
gate. Even a perfect numerical repair in the same package topology could not
execute 376 routed experts per accepted second.

Kill this untrained four-effective-bit vector fitting rule and per-expert Core
ML package switching. Preserve the positive resident-arithmetic substrate
result. A future Core ML branch must use a resident shared multi-expert
transaction and a separately justified trained/activation-aware low-rate code;
it may not merely add bits, preload the impossible full bank, or report this
component timing as endpoint TPS. The run records zero accepted tokens and no
throughput-model constant change.
