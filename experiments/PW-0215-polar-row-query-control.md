# PW-0215 — Recursive polar row-query control

- Status: complete
- Disposition: rejected at the frozen early-projection gate
- Date: 2026-08-11
- Owner: Thimble with project-owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0116
- Related records: PW-0141, PW-0211 through PW-0214

## Question

The row-sketch controls in PW-0213 preserve Cartesian scalar codes. Test the
distinct proposal from the project-owner attachment: represent each complete
projection row by one radius and a recursive tree of quantized pair angles,
then consume that representation as an inner-product code rather than an
ordinary quantized matrix.

## Frozen cheap falsifier

Use layer 4, expert 96, gate projection and PW-0116's 56 validation queries.
Apply one deterministic signed normalized Hadamard transform to rows and
queries. Recursively factor every transformed row into pair radii and angles
until one root radius remains. Fit one deterministic 64-entry shared F16 angle
codebook per tree level using projection rows only; encode each of the `4095`
angles with six bits and each row radius with F16.

Charge row-padded fixed addressing, all codebooks, the query transform, angle
lookups, and recursive decode arithmetic. The matched control is F16-metadata
affine6/group-128 RTN under the same real query authority. Report projection
relative L2, row tails, weight reconstruction, bytes, and counted extra work.

Kill before another projection if polar6 does not improve held-out projection
relative L2 over affine6 or exceeds 75.1% of source-FP8 weight bytes. A pass
authorizes a complete-expert test only; it does not authorize a packed bank,
Metal kernel, endpoint, or TPS claim.

## Result

The bounded run is valid and rejects this representation before another
projection. Its 6,297,088-byte physical ledger is `75.0671%` of source-FP8
weight bytes: 3,074 fixed-address bytes per row plus 1,536 bytes of shared F16
angle codebooks. The reconstructed transformed weights have `3.1907%` relative
L2.

On the 56 real PW-0116 validation queries, affine6 reaches `0.5945%`
projection relative L2 and `0.6956%` maximum-row error. Polar6 is worse at
`0.8312%` and `0.9553%`. It also requires 49,152 query-transform add/subtracts,
8,386,560 row-angle lookups, and 16,773,120 recursive multiply/adds before or
during the ordinary dot products. It therefore fails both the accuracy premise
and the intended cheap direct-query premise at equal traffic.

Reject this fixed signed-Hadamard, shared-level 64-angle recursive tree. This
does not claim every learned directional representation is impossible, but it
does close the concrete PolarQuant-inspired row code authorized by the project
owner's attachment. Do not spend a Metal kernel or another expert on it.

Evidence `PW-0215-layer04-expert096-gate.json` hashes to
`3c0e0232758ded9aff26aa8ce33c2fdc53319f68212ab0f7ad18ff42ab995e01`;
its F16 codebooks hash to
`23444bd944424b908c9abfe7e3bdb01f2d441ccd6aff6dbdb4900bdc6264a291`.
