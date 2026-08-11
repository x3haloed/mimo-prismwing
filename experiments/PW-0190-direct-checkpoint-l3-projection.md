# PW-0190 — Direct-checkpoint L3 projection equivalence

- Status: completed
- Disposition: promoted to a complete direct-checkpoint L3 expert
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`
- Execution mode: explicitly modified `metal-native-l3`
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0033, PW-0114, PW-0188, PW-0189

## Contract and gate

PW-0189 correctly rejects the current Metal kernel as a source-BF16 projection.
PW-0114 already conditionally qualifies that same repair-free arithmetic as a
named L3 distribution premise. Isolate the physical change under that semantic:
generate the readable mapped-FP8/F32 reference for layer-4 expert-64 gate, then
run the identical Metal kernel with weight and scale bound directly from the
original checkpoint shard through page-rounded no-copy regions and explicit
offsets.

Pass only if both source-copy bytes are zero, both offsets are recorded, and
the direct binding retains relative L2 at most `2e-5` and maximum absolute
error at most `2e-4` against the readable L3 reference. Preserve PW-0189's
failure and target-faithful/modified labels. A pass promotes a complete
direct-checkpoint L3 expert and no target-faithful or endpoint claim.

## Result

The original layer-4 expert-64 gate weight and scale bind at offsets `3,264`
and `7,360` from two page-aligned regions totaling 8,421,376 mapped bytes.
Metal copies zero source bytes. Against the readable mapped-FP8/F32 authority,
the complete projection reaches `9.13839e-7` relative L2 and `1.19209e-5`
maximum absolute error, passing the frozen L3 thresholds.

The command takes 1.549 ms on its first dispatch and 0.579 ms warm median on
the existing Apple M1. Timing excludes source hashing and is diagnostic, not a
layer or endpoint claim. The report hashes to
`f3b2aaa099cd0c47b29efa4bfe41279ec608b6ea974528d0945112f113a70f31`;
the readable reference and Metal output hash to
`61caddd1d8fc52df3f10e81713f11af15f40b5924fd9aae30e8130d7df51cf70`
and `009614a47ef628013437e3f8b5f50fd321eca06839e93e7b94f41bcd1f8621d0`.

Promote one complete direct-checkpoint L3 expert. Preserve PW-0189's source
failure: this changes the physical embodiment of PW-0114's named modified
arithmetic, not its target fidelity. Zero tokens are accepted and no endpoint
throughput constant changes.
