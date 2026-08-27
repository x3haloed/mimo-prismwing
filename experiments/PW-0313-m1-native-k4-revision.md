# PW-0313 — M1-native K4 revision and semantic gate

- Status: complete
- Disposition: conditional for expert 199; rejected for expert 41 expansion
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

## Execution and evidence

The constructor and repeated-evidence analyzer were committed and pushed before
their respective evidence runs. The two decision runs per expert bind clean
commit `922641c17d2f532f36e76e9cafc4638beb6759f9`.

| Expert | Run | Status | Wall seconds | Peak RSS bytes | Report SHA-256 |
| ---: | --- | --- | ---: | ---: | --- |
| 199 | 002 | qualified | 503.955450 | 1,461,338,112 | `d3aa12831c12d5d214f403c4bdcb597a50ca29b50857e0ae9be84a939bff0992` |
| 199 | 003 | qualified | 505.760279 | 1,472,675,840 | `66bce59587ef656a0969fa00aa187f03b6bd4f89bac09a09f41519d6aa44bc39` |
| 41 | 001 | semantic gate failed | 505.926391 | 1,464,041,472 | `3758952e971a57bffd3449366696b46a46148c8597be607ebfa8bb1967259c23` |
| 41 | 002 | semantic gate failed | 508.361189 | 1,508,933,632 | `0f0157e1d0e5373bfeabec26b9b8a500185a4b9cac1dbd96295507811fd51957` |

For each expert, the two fresh processes produce the same 33-file deterministic
tree and identical semantic report. Expert 199's tree contains 29,992,910
bytes; expert 41's contains 29,991,879 bytes.

The canonical analyzer output is
`/Volumes/Elements/mimo-prismwing/evidence/PW-0313/summary.json`, SHA-256
`e61c1487055cce54a7a72e3505003eb7f6c5c4c70fca7ab8f9fa3bd037397ddd`.
It binds all four report paths, byte sizes, hashes, commits, repeat trees,
semantic results, and safety decisions.

## Results

### Policy-relevant expert 199

- gate and down executable payloads are M4-identical;
- up uses a different packed trellis but independently decodes to every M4 F32
  weight bit exactly;
- M1-native versus M4 complete-expert relative L2 and maximum absolute error
  are both zero;
- source degradation relative to M4 is zero;
- substituting the bit-identical expert output leaves the authenticated
  PW-0424 candidate route hash unchanged at
  `6b7b0459c75aa1885009a44c31b4653e405d30921e6a6c85a8192516aaf55104`;
- the route remains at `0.004701004` relative L2 versus source, below `0.01`.

The preserved first attempt at commit `a78c9a0` failed only because the
evaluator demanded reconstruction of PW-0424's one-off route assembler, whose
source was never preserved. Its report hashes to
`a8beac83812dfae5a77b6bf566823933930ed947bcb17be270429128c9914890`.
The corrected proof does not approximate that missing assembler: it requires
the replacement expert output to be bit-identical and then proves that exact
substitution cannot change the frozen route. Any non-identical replacement
still fails closed.

### Harder expert-41 control

- up and down executable payloads are M4-identical;
- gate is a deterministic numerical-drift case: decoded M1 versus M4 relative
  L2 is `0.002102904`, maximum absolute error `0.0001217201`;
- complete-expert M1 versus M4 relative L2 is `0.006918367`, exceeding the
  predeclared `0.005` gate, with maximum absolute error `0.0078125`;
- additional source error is only `0.0000442093`, within its separate gate.

The threshold remains unchanged. Expert 41 fails this revision even though its
source-relative quality penalty is small.

All four decision runs pass Gate 8: minimum free memory is 69%, maximum process
footprint is 1,592,464,640 bytes, maximum peak RSS is 1,508,933,632 bytes,
release footprints are 346,574,720--357,781,504 bytes, and swap growth and new
throttling are zero. Protected services remain healthy.

## Decision

Promote `m1-native-k4-v1` only for the policy-relevant expert-199 identity under
the authenticated layer-28 fixture. It is locally byte-repeatable, executable-
semantic identical to M4, and route-preserving. Reject expert-41 expansion
under this revision because its deterministic complete-output error exceeds the
frozen gate.

This is a split conditional result, not arbitrary-expert or new-layer
authorization. A subsequent bank experiment must classify and gate each new
identity; it may not infer portability from expert 199 or silently include
expert 41. Construction accepts zero tokens, so this result changes no
throughput-model constant and makes no endpoint-TPS claim.
