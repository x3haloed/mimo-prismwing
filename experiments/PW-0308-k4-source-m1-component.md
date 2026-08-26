# PW-0308 — Modified K4/source M1 routed-component calibration

- Status: complete
- Disposition: conditional component lower bound; modified layer-28 endpoint overlay authorized
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

## Result

The corrected import is clean commit
`1301c61713f2f19d0ee5bb2b6784e6df0ba47eba`. Formatting, all 110 Rust tests,
the release build, runtime Metal compilation, and standalone compilation of all
four hash-bound Metal sources pass.

The clean `raw-002` replay reproduces all 4,096 candidate F32 output bit
patterns for the initial layer execution, 20 warmups, 100 layer samples, and
120 repeated-component comparisons. The single-layer complete-call p90 is
`15.448083` ms. The 47-repeat complete-call p90 is `351.680083` ms and GPU p90
is `341.382917` ms. The wall-minus-GPU gap is only `10.297166` ms, so command
submission and CPU synchronization are no longer the dominant routed-component
cost in this realization.

The strict two-TPS component condition passes and the three-TPS diagnostic
fails. System-free memory remains at least 56%; swap use does not grow; no new
throttled page appears; all baseline protected PID identities remain; peak RSS
is 194,854,912 bytes; and final physical footprint is 19,662,336 bytes. The
bundle mapping and phase resources are therefore released safely.

The valid raw manifest is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0308/raw-002/manifest.json`,
SHA-256 `d395cd1844ee46a938578063ab7c68ba156b6e3b1e53f29b29c58c6e33949613`.
Its readback, layer, and repeated reports hash respectively to
`d2cc16bd35b0a445e0affb6350bf7a4f0594dc67078b0ac0e197b2cd1ec58328`,
`c222cdda47f894924fe4154c67e606e13324314fbe0ee4576d95487c6a216ca1`,
and `754cb36ba8d3831a3d7e3c59f5faebd7ea17c924b9d34f34343541ff3e7d9c4e`.
The durable 15-file input ledger hashes to
`a080398b1de4cc78fae39a1004947a5f50fac40fb8adbeef3d048021db5a3618`.

`raw-001` is preserved as invalid evidence because its readback was given a
guessed commit argument; its manifest hashes to
`6c393b4379704cdc42e015df169dce0bad6f12a9f1edbed9d0a475db0fd84d5a`.

## Decision

Retain the mixed K4/source runtime as an authenticated L3 modified component
lower bound and proceed to the complete source endpoint's layer-28 overlay
slice. Do not promote the K4 weights, claim three TPS, infer distinct-layer or
cache behavior, or update a measured endpoint throughput constant. The result
confirms that another top-eight batching pass is not the missing mechanism:
GPU arithmetic owns about 97% of p90 complete-call time after 47 dispatches are
amortized. A greater-than-three-TPS embodiment still requires materially
smaller executable expert records or a different representation, while the
overlay determines this fallback's actual complete-path and downstream-error
cost.
