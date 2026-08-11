# PW-0194 — Count-aware direct-checkpoint L3 MoE

- Status: completed
- Disposition: rejected; promote compile-time count specialization
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0193 fixture
- Execution mode: explicitly modified `metal-native-l3` route replay
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0036, PW-0187, PW-0192, PW-0193

## Contract and gate

Replace fixed eight-row loops in PW-0193's per-expert gate, up, and down
projections with a count-aware shared-weight kernel accepting each expert's
authenticated active count from one through eight. Bound SwiGLU to the same
active rows. Keep tensor mappings, routes, weights, scatter, source-FP8
equation, buffer offsets, and checkpoint authority unchanged.

Before the real run, execute a deterministic Metal fixture at a partial active
count and compare every active output with a scalar source-FP8 reference while
proving inactive output remains untouched. Pass only if fixture maximum error
is at most `2e-5`, PW-0193 mixture parity remains within its existing gate,
zero source bytes are copied, exactly 64 rather than 136 expert rows are
executed, and warm layer wall improves by at least 1.5x over PW-0193's 32.906
ms. Record cold/warm state, bytes, `U=2.125`, batch eight, concurrency one, and
zero executed/accepted tokens.

This remains one modified L3 layer. A pass promotes count-aware bindings across
the wide verifier; it is not accepted-token or endpoint TPS.

## Result

The active-count kernel fixture passes at `1.67638e-8` maximum absolute error,
and the real route union remains byte-identical to PW-0193's output with
`1.49824e-6` relative L2. It records exactly 64 executed rows, zero source
copies, `A=0`, and `U=2.125`.

Performance fails decisively. Cold wall rises from 45.519 to 77.030 ms and warm
median regresses from 32.906 to 62.600 ms (`0.526x`, not the required `1.5x`).
Runtime-bounded position loops prevent the fixed-width compiler optimization
that made PW-0192 fast. The report hashes to
`6c7a22ce209fc6ac429d21daab3691cf584d8ea3b5cf26932aef83bc47a5493e`.

Reject dynamic active counts. Promote one final count-aware form: eight
compile-time-specialized 1--8-row kernels selected by authenticated expert
count. Preserve this negative result, zero accepted tokens, and unchanged
endpoint constants.
