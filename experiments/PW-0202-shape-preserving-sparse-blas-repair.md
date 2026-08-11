# PW-0202 — Shape-preserving sparse BLAS repair

- Status: completed
- Disposition: rejected; conservative certificate flags most outputs
- Date: 2026-08-10
- Execution mode: target-faithful numerical/performance falsifier
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0101, PW-0200, PW-0201

## Hypothesis

Source BLAS association can be preserved without decoding or computing every
real weight row. A zero-filled matrix retaining the original projection shape,
with only numerically ambiguous rows populated at their original indices,
reproduces source BF16 values while reading only those checkpoint rows. A
conservative forward-error certificate will select few enough rows for this
repair to remain cheaper than the wide Metal transaction.

## Contract and gate

First preserve raw Metal pre-round outputs for all PW-0200 projections. For
each row compute `sum(abs(weight_i * input_i))` and a conservative float32
forward-error bound using unit roundoff and the full reduction length. Flag a
row only when that bound intersects a BF16 rounding boundary. Report flagged
fraction, capture of all nine known mismatches, and false-positive density.

Then benchmark a source-library projection with original output dimensions and
row indices, zeroing every unflagged row and decoding only flagged checkpoint
rows. Promote only if all nine frozen values are recovered, every mismatch is
flagged, and extrapolated repair cost leaves the PW-0187 complete-path bound
below one second per accepted token. Record zero accepted tokens until endpoint
integration.

If the conservative certificate flags too many rows, reject row-level repair
and test a calibrated tile certificate only if it remains fail-closed.

## Result

The conservative `2*gamma_n*sum(abs(weight_i*input_i))` certificate captures
all nine PW-0200 mismatches, but flags 43,087 of 65,536 outputs
(`65.7455%`). That density destroys the sparse-repair premise before a repair
benchmark: retaining original source-BLAS shape for nearly two thirds of all
rows cannot be cheaper than the wide projection it was meant to correct.

The validated evidence hashes to
`2f838aeb8344051428d1d27ddef09f0e2c62501243ad15b4ecc194d88f0289a8`.
Reject conservative row-level exact repair. Do not calibrate the bound against
these nine visible misses: that would cease to be fail-closed. The whole-model
L3 endpoint gate remains the appropriate next falsifier because PW-0114
already showed that isolated BF16 disagreements need not alter the accepted
distribution. Zero tokens are accepted here and no throughput constant
changes.
