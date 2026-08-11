# PW-0195 — Specialized-width direct-checkpoint L3 MoE

- Status: completed
- Disposition: promoted across the direct-checkpoint wide L3 verifier
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0193 fixture
- Execution mode: explicitly modified `metal-native-l3` route replay
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0192 through PW-0194

## Contract and gate

Replace PW-0194's runtime-bounded loop with eight compile-time-specialized
shared-weight kernels for widths one through eight. Select the pipeline solely
from each authenticated expert placement count. Keep mappings, routes, source
weights, arithmetic order within every active row, SwiGLU bound, scatter, and
reporting unchanged.

Run deterministic correctness fixtures for every width and require at most
`2e-5` maximum error with inactive outputs untouched. On PW-0193's real route
union, require unchanged mixture parity, exactly 64 executed rows, zero copied
source bytes, and at least 1.5x warm speedup over the fixed-eight 32.906-ms
control. Record compile, cold/warm wall, bytes, `U=2.125`, batch eight,
concurrency one, and zero executed/accepted tokens.

This is still a modified one-layer scheduling result. A pass promotes the
specialized pipelines across the wide verifier; failure kills position-count
specialization and returns to fixed batch eight.

## Result

All eight specialized widths pass, with worst fixture maximum absolute error
`1.67638e-8` and inactive outputs untouched. The real 17-expert union remains
byte-identical to PW-0193/PW-0194 at `1.49824e-6` relative L2 and
`1.67347e-10` maximum error. Exactly 64 rows execute, model source-copy bytes
remain zero, and `A=0`, `U=2.125` are reported honestly.

Cold wall is 34.013 ms and warm median is 19.745 ms. The candidate is 1.667x
faster than PW-0193's 32.906-ms fixed-eight control and 3.170x faster than
PW-0194's dynamic-count failure, clearing the 1.5x gate. The authoritative
report hashes to
`bfbfed78d13ad80b47a6dc1cedefea3fdb9ce7ef2ff8add015f071283a0a0450`.

Promote compile-time width specialization throughout the direct-checkpoint
wide L3 verifier. At this representative union, repeating warm layer wall over
47 routed layers and dividing by PW-0187's `A=5` is about 0.1866 seconds per
accepted token before attention and all other work. This is a diagnostic, not
endpoint TPS; source-BF16 fidelity and whole-path execution remain open.
