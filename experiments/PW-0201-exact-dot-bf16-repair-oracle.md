# PW-0201 — Exact-dot BF16 repair oracle

- Status: completed
- Disposition: rejected; exact arithmetic reproduces Metal at eight of nine sites
- Date: 2026-08-10
- Execution mode: target-faithful numerical falsifier
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0101, PW-0200

## Hypothesis

At PW-0200's nine disagreement sites, correctly rounding a high-precision dot
product selects the frozen source BF16 value. If true, a fast Metal dot can be
paired with a midpoint-ambiguity certificate and selective accurate reduction
without reproducing CPU BLAS association everywhere.

## Contract and gate

Authenticate the PW-0200 manifest and PW-0101 references. Reconstruct exactly
the source dynamic-FP8 input and source-FP8 weight row for every mismatch. Sum
each product in float64, round the result to BF16, and compare its bits with both
the frozen source and Metal candidate. Promote only if all nine high-precision
results match the source and disagree with the candidate where recorded. This
is a numerical oracle, not throughput evidence; record zero accepted tokens.

If any exact result disagrees with the source, reject generic exact repair and
derive a source-topology correction instead.

## Result

The hypothesis fails. Correctly rounded float64 accumulation matches the source
at only one of nine sites and matches the Metal candidate at the other eight.
The manifest hashes to
`3559b621c5060b328fffa68eb4894aa54b9133a4cbd1c5bc8e1198a79a62d9b5`.
Thus mathematical accuracy is not a source-fidelity repair: the frozen output
encodes the source library's float32 association. A follow-up shape-preserving
sparse matrix experiment does reproduce all nine source values when it retains
the original output dimension and row indices while zeroing non-candidate rows.
Promote that mechanism to a bounded ambiguity-density and timing test. Zero
tokens are accepted.
