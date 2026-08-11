# PW-0192 — Direct-checkpoint batch-eight shared L3 expert

- Status: completed
- Disposition: promoted to a direct-checkpoint heterogeneous L3 scheduling probe
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Execution mode: explicitly modified `metal-native-l3`
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0036, PW-0114, PW-0187, PW-0191

## Contract and gate

Regenerate PW-0036's deterministic eight-row layer-43 expert-32 oracle from
the original internal checkpoint shards. Execute the existing shared-weight
GEMM8 gate/up/SwiGLU/down transaction with all six source tensors bound through
PW-0191's page-rounded no-copy mappings and explicit offsets.

Pass only if source-copy bytes remain zero; the independent batch kernel and
SwiGLU fixtures pass; complete output relative L2 is at most `3e-5` and maximum
absolute error at most `2e-8`; warm complete-expert wall is at most 3.5 ms; and
per-position speedup over PW-0034 batch one is at least 2.5x. Record cold and
warm state, bytes, hardware, batch eight, concurrency one, zero executed or
accepted tokens, and one diagnostic shared expert set.

This is a reuse primitive in named modified mode, not a heterogeneous route
union, accepted-token verifier, or endpoint result. A pass promotes a direct-
checkpoint heterogeneous scheduling probe using PW-0187 routes.

## Result

The six original-shard bindings copy zero source bytes and preserve the same
nonzero offsets as PW-0191. The batch kernel fixture is exact, the SwiGLU
fixture remains within `1.97745e-7`, and the complete eight-row output reaches
`1.62608e-6` relative L2 and `2.47383e-10` maximum absolute error. Its SHA-256
is `8c198563b12f73a7c5fd181e2d173ffa55692c64ddd1d5386cb0ccdfac2a1393`.

The first command takes 5.257 ms and the warm median is 2.104 ms, or 0.263 ms
per position and 3.881x faster per position than PW-0034 batch one. Both timing
gates pass. The regenerated fixture and authoritative corrected runtime report
hash to `bb0755c3d06dc87742d8e464b2e2c6facf5c0d07fdb1fddcb61f25927e5ac48d`
and `0471f6932abecd830da7cc42ac3da05345d67b249f6937d3cabae85d52a8eb24`.

The corrected report distinguishes eight fixture rows from accepted work:
`accepted_tokens=0`, `A=0`, and diagnostic shared-expert `U=1`. Promote a
heterogeneous direct-checkpoint scheduling probe using authenticated PW-0187
routes. This remains modified L3 and changes no endpoint throughput constant.
