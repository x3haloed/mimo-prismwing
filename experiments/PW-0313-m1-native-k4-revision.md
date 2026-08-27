# PW-0313 — M1-native K4 revision and semantic gate

- Status: in progress
- Disposition: pending
- Date: 2026-08-26
- Owner: Codex
- Parent experiment: PW-0312

## Question

Can a separately named `m1-native-k4-v1` representation produce deterministic
complete K4 experts on the target M1 while preserving the frozen source and
routed quality needed to continue toward Prismwing-2?

## Hypothesis and mechanism

PW-0312 rejects cross-device payload identity but separates two mechanisms:
semantically exact trellis aliases and small weight-dependent candidate drift.
Neither requires silently replacing the authenticated M4 artifacts. A
target-native revision can instead bind its own constructor host, software,
source tensors, calibration, seeds, arrays, and payload hashes, then be judged
by local repeatability and the project's external semantic gates.

Expert 199 is policy-relevant K4 in PW-0425 and already shows exact candidate
semantics despite an aliased up trellis. Expert 41 is deliberately source-FP8
in that mixed policy and supplies the harder numerical-drift control.

## Protocol and gates

1. Reuse all PW-0311 authority, source, calibration, QTIP, TLUT, commit, and
   Gate 8 checks unchanged. Name every output `m1-native-k4-v1`; never present it
   as PW-0352 payload identity or source-exact.
2. Construct complete gate/up/down artifacts for experts 199 and 41 without
   aborting on an M4 payload difference. Preserve the exact M4 comparison and
   classify each projection as payload-identical, semantically identical with
   an aliased packed form, or numerically different.
3. Independently decode every local payload and require relative L2 at most
   `2e-5`, inherited from PW-0352.
4. Repeat each target-native construction in a fresh process from the same
   clean pushed commit. Require every target-native candidate array, packed
   state, manifest, fixture, and referenced payload byte to match its first
   local run exactly.
5. On the frozen PW-0352 input, require M1-native versus M4 expert-output
   relative L2 at most `0.005`. Require M1-native expert-output relative L2
   versus source not to exceed the corresponding M4 value by more than `0.005`.
6. Replace only expert 199's M4 candidate with its M1-native candidate in the
   authenticated PW-0424 three-K4/five-source route. Require candidate-route
   relative L2 versus the M4 route at most `0.001` and preserve PW-0424's
   source-route relative-L2 gate below `0.01`.
7. Record construction wall, I/O, RSS, physical footprint, release footprint,
   memory-free floor, swap/throttle growth, services, hardware, software, and
   commit. Construction accepts zero tokens and makes no TPS claim.

## Decision rule

- If both local repeats are byte-identical and all semantic/safety gates pass,
  authorize a new-layer `m1-native-k4-v1` construction and policy-quality test.
- If local repeatability fails, reject target-native artifact construction.
- If policy-relevant expert 199 fails semantic or routed gates, keep the
  authenticated M4 bundle only and require construction on the M4 authority
  host for any expanded mixed-policy bank.
- If only the harder expert-41 drift control fails, retain the current
  three-K4/five-source policy and do not broaden K4 coverage to that identity.

## Claims excluded

- source-exact or L1 weights;
- arbitrary identities or layers;
- complete-bank construction;
- hosted-reference or multimodal equivalence;
- ordinary endpoint execution;
- accepted-token TPS, Prismwing-2 completion, or Prismwing 50 completion.
