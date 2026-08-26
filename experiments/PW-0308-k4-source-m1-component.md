# PW-0308 — Modified K4/source M1 routed-component calibration

- Status: in progress
- Disposition: pending clean-commit replay
- Date: 2026-08-26
- Owner: Codex
- Starting commit: clean `942fb42cae8c3dafce6093157a769139305f5935`
- Source handoff: `/private/tmp/prismwing-k4-source-mainline-942fb42/`, patch
  SHA-256 `a2536358f60fae8b7153980160ccf9e3de9faceb02ad79405f9121d7bd91ef1d`
- Hardware: Apple M1 Mac mini (`Macmini9,1`), 16 GiB unified memory, macOS
  26.6.1 (25G76)
- Related records: PW-0203 through PW-0216, PW-0300 through PW-0307; external
  source experiments PW-0424, PW-0425, and PW-0478

## Prediction error and corrected contract

The handoff calls the bundle identity-preserving because selected expert IDs
are unchanged and substitution is forbidden. That is not Prismwing's L1
meaning: three selected experts use approximate QTIP K4 weights, and the
source-route relative L2 is `0.004701688420027494`, not zero. Exact F32 parity
against `candidate_routed_f32` proves the Metal implementation reproduces the
modified candidate; it does not prove the candidate reproduces source weights.

Import this branch only as **L3 modified weights**. Reports must state
separately that expert identity is preserved, source function is not preserved,
and the frozen candidate route passes its local one-percent component gate.
Neither `correctness_qualified` nor an identity-preserving status may be emitted
by the imported runtime. The immutable external manifest retains its original
semantic string as artifact history.

## Authenticated inputs

The user supplied the omitted runtime archive at `/private/tmp/Archive/`.
Fail closed unless all of these exact inputs match:

- bundle: 164,724,736 bytes, SHA-256
  `1851c1fe713abce8e6583908937ca831ed591c4abc23517511ff7624f3f9294c`;
- bundle manifest: 28,684 bytes, SHA-256
  `86e486d4cc3fcad237b504ffbd6d276ed7c53688f44709f7bc5c5a334479555b`;
- route/candidate fixture: 2,255,030 bytes, SHA-256
  `05439a232c2002530002d95ac29831b38a5c74b1049903406747620b3ce4f64e`;
- source patch: SHA-256
  `a2536358f60fae8b7153980160ccf9e3de9faceb02ad79405f9121d7bd91ef1d`.

PW-0425's archived manifest hash chain authenticates its bundle, manifest,
Rust readback, and M4 native result. The two external experiment-spec hashes
are recorded but unavailable, so their prose contracts are not independently
authenticated here.

## Falsification and promotion gates

1. Apply cleanly to the exact mainline base, pass formatting, all Rust tests,
   release build, runtime Metal compilation, and standalone Metal compilation.
2. Reproduce all 4,096 candidate output F32 bit patterns on the target M1 for
   the first execution, 20 warmups, and 100 samples.
3. Encode 47 repeats of the frozen layer/input/route into one command buffer.
   This is deliberately a repeated-component dispatch test, not 47 distinct
   decoder layers.
4. Preserve 20 warmups and 100 timed samples, exact output determinism, complete
   call and GPU timing, and normative Gate 8 safety/release evidence.
5. A strict complete-call p90 below 500 ms passes only the roughly-two-TPS
   routed-component necessary condition and authorizes a complete endpoint
   layer-28 overlay slice. A strict p90 below 333.333 ms is a three-TPS
   diagnostic. Neither is endpoint TPS.
6. Do not promote K4 weights, a throughput-model constant, or a production
   default until the modified complete endpoint and downstream fidelity are
   measured.

## Preliminary observation

An uncommitted review-worktree calibration produced exact candidate parity.
The target-M1 single-layer complete-call p90 was `16.514291` ms. The 47-repeat
complete-call p90 was `367.741542` ms and GPU p90 was `356.975083` ms. It
therefore passed the two-TPS component condition and failed the three-TPS
diagnostic. Minimum free memory was 64%, swap and throttled-page growth were
zero, protected PID identities were stable, peak RSS was 194,740,224 bytes,
and the final physical footprint was 10,225,152 bytes.

These observations select the endpoint overlay as the next causal slice but
are not final evidence. Replay from the clean implementation commit before
adjudication.
