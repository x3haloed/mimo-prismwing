# PW-0199 — Source projection reduction-width sweep

- Status: completed
- Disposition: rejected; no ordinary tree width passes
- Date: 2026-08-10
- Execution mode: target-faithful layer-local numerical falsifier
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0196, PW-0197, PW-0198

## Hypothesis

PW-0197's residual error is caused by the 64-lane tree reduction topology. A
bounded sweep of 16, 32, 64, 128, and 256 lanes changes only the association of
the same source-FP8 products and may identify a topology whose BF16-staged
results cross the frozen source gate.

## Contract and gate

Use the unchanged PW-0197 source reference, static routes, GPU dynamic-FP8 and
BF16 staging, original-shard no-copy bindings, and compile-time expert widths.
Run each declared lane width independently; require exact kernel fixtures,
relative L2 at most `2e-5`, maximum absolute error at most `2e-4`, finite output,
and zero source-copy bytes. Failed candidates retain diagnostic output hashes
but no timing claim. A passing candidate must also have a warm median no worse
than 30 ms. Record zero accepted tokens.

If no width passes, reject ordinary power-of-two tree reassociation and move to
an explicitly source-calibrated accumulator or endpoint-level error analysis;
do not search arbitrary tolerances.

## Result

No candidate passes. Relative L2 is `0.00263641` at 16 lanes, `0.00272782` at
32, `0.00272864` at 64, and `0.00271343` at both 128 and 256. Every maximum
absolute error remains `2.38419e-7`; none approaches the frozen `2e-5` relative
gate. The rejected outputs for 16, 32, 128, and 256 lanes hash to respectively
`ec58262fabcfe3d701143fc0aa58cc1111ddca89aba53d2e906b9557cdcec3ae`,
`043fc309199af5139a456d8524d33609a73b726d54a1ec661558e1bb8adf40c8`,
and `e9b018875690bd547d10c7e082d314aba34fd6227f434306decfc77589feef0f`
for both wider cases. Reject ordinary power-of-two reassociation. No timing is
promoted and zero tokens are accepted.
