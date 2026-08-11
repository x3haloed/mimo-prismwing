# PW-0193 — PW-0187 layer-43 direct-checkpoint L3 MoE

- Status: completed
- Disposition: promoted to count-aware heterogeneous direct-checkpoint scheduling
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; authenticated PW-0187 routes
- Execution mode: explicitly modified `metal-native-l3` route replay
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0037, PW-0114, PW-0187, PW-0192

## Contract and gate

Freeze layer 43's real PW-0187 eight-position route matrix (`17` unique
experts, normalized `U=2.125`) over PW-0192's deterministic eight-row input.
Generate an independent readable source-FP8/F32 weighted-mixture authority from
the authenticated original shards. Execute the existing heterogeneous
shared-weight Metal schedule with each expert's six tensors page-rounded and
bound no-copy from those same shards. Authenticate shard identities through
the pinned checkpoint-verification manifest; do not rescan every multi-GB
shard inside the timed process and do not use an external sidecar.

Pass only if all 64 placements and weights reproduce the PW-0187 matrix; all
source-copy bytes are zero; the output passes the existing heterogeneous MoE
parity gate; the transaction records cold and warm walls, logical/mapped and
resident bytes, route union, padding, batch eight, concurrency one, and zero
executed/accepted tokens; and Gate 8 remains viable. The existing copied
artifact modes remain separate controls.

This is a one-layer L3 scheduling result, not source-BF16 verification,
accepted-token execution, or endpoint TPS. A pass promotes replacing every
layer's repacked expert artifacts with authenticated original-shard mappings in
the wide verifier.

## Result

The authenticated PW-0187 layer-43 route replay contains 17 unique experts and
64 real placements (`U=2.125`). All tensors bind from the original internal
shards with zero copied source bytes. The weighted Metal output reaches
`1.49824e-6` relative L2 and `1.67347e-10` maximum absolute error against the
independently generated mixture authority. The runtime report hashes to
`cf86d431140848bb090eac05e0ad2309c0fb61bed3b0ca071f3cdd1cc3818e6a`.

Checkpoint content, sizes, inodes, and nanosecond modification times match the
pinned verification authority. APFS's device identifier changed uniformly
from `16777233` to `16777231`; the runtime records that single transition and
fails closed on any nonuniform identity change. No content threshold is
weakened and no shard rescan is hidden inside timing.

The first transaction takes 45.519 ms and the warm median is 32.906 ms. The
legacy 25-ms heterogeneous timing flag is false. This is not a route-union
failure: the fixed batch-eight kernel executes 136 padded expert rows for only
64 real placements, a 112.5% padding overhead. Promote a count-aware kernel
that preserves shared-weight reuse but loops over each expert's actual 1--8
rows. Zero tokens are accepted, `A=0`, and no endpoint constant changes.
