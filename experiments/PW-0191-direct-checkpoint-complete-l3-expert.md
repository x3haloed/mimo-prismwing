# PW-0191 — Direct-checkpoint complete L3 expert

- Status: completed
- Disposition: promoted to direct-checkpoint heterogeneous L3 scheduling
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Execution mode: explicitly modified `metal-native-l3`
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0034, PW-0114, PW-0188 through PW-0190

## Contract and gate

Regenerate PW-0034's deterministic batch-one layer-43 expert-32 oracle from
the two original internal checkpoint shards. Execute gate, up, F32 SwiGLU, and
down in one serialized Metal command using six page-rounded no-copy source
bindings and their explicit logical offsets. Do not read an external sidecar
or construct a repacked expert artifact.

Pass only if every projection layout authenticates; all six source-copy byte
counts are zero; the complete 4,096-value expert output remains within
`3e-5` relative L2 and `2e-8` maximum absolute error of the independent oracle;
the deterministic SwiGLU fixture passes; and the existing 3-ms warm timing gate
passes. Record all offsets, mapped/logical bytes, cold/warm state, device,
batch, concurrency, `A`, and `U`.

This is the next physical rung for PW-0114's named modified arithmetic. It is
not source-BF16 target fidelity, accepted-token execution, or endpoint TPS. A
pass promotes direct-checkpoint heterogeneous expert scheduling.

## Result

All six original-shard tensors bind with nonzero logical offsets and zero
copied source bytes. Their page-rounded regions total 25,264,128 bytes versus
25,204,736 logical source-and-I/O bytes. The complete gate/up/F32-SwiGLU/down
output reaches `4.69754e-7` relative L2 and `4.45652e-11` maximum absolute
error against the independently regenerated deterministic oracle; its SHA-256
is the original PW-0034 value
`1fb7fb1755eff0c72ab3a0de7744bf62455fc3177a6f6e0b35617978ca97247e`.

The first command takes 2.997 ms and the warm median is 1.078 ms on the Apple
M1, passing the unchanged 3-ms gate. The independent fixture manifest hashes
to `c737021b495f77bb313623f371896004df688ed9f0f1f4c23f7ef516894d1092`
and the runtime report hashes to
`f45ed1c4becbb3640948bf26d7455c8a8b8f1a3bb29e9ecc6b3ed5d1cf3f61d4`.

Promote direct-checkpoint heterogeneous expert scheduling under the explicit
L3 label. No external sidecar participates. Zero tokens are accepted and no
endpoint throughput constant changes.
