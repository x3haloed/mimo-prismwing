# PW-0110 — Cold-storage speculation width bound

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: pending
- Evidence authority: PW-0108 raw report
  `6f7d816b4f39c00b967642bdf300e7baea8563a5fca593ab5d0943b5df047d68`;
  clean analysis
  `5281fd36c06e2a2e5767918bbb63f0fe33cbec4a1478b4281806d6fdf56ac43d`
- Hardware/runtime: Apple M1 shared 16 GiB host; internal SSD; unchanged exact
  source-FP8 selected-expert representation
- Related records: PW-0010, PW-0011, PW-0044, PW-0102 through PW-0104,
  PW-0106 through PW-0109

## Question and bound

PW-0010's earlier speculation bound used warm Metal projection bandwidth.
PW-0108 now supplies a stricter physical authority for a streamed exact
runtime: the best cold median to acquire the authenticated eight-expert payload
is 58.033833 ms per routed layer with three concurrent Metal-I/O command
buffers. MiMo has 47 routed layers. What proposal width is mathematically
necessary before a source-FP8 speculative verifier can reach 34.3 or 50
accepted TPS on the internal SSD?

Compute an optimistic lower bound only. For one verifier block, assume:

- every layer selects the same minimum eight-expert set across every proposed
  position, so `U=1`;
- every proposed token is accepted, so `A=q`;
- layer acquisitions compose as `47 * 58.033833 ms`;
- all dense/attention weights, compute, I/O/compute barriers, draft work,
  correction, rollback, KV work, sampling, and thermal effects are free; and
- the PW-0108 selected-byte payload does not grow with width.

Then `TPS <= q / acquisition_seconds`. For nonideal union and acceptance,
`TPS <= (A/U) / acquisition_seconds`. This is a necessary bound, never an
endpoint estimate.

## Protocol and gates

Add a hash-pinned analyzer that reads the immutable PW-0108 report, recomputes
the one-, two-, and three-command cold medians, selects the measured best
configuration rather than a copied constant, verifies exact bytes and Gate 8,
and emits:

- the 47-layer acquisition floor;
- ideal ceilings for `q=8,16,32,64,128,137`;
- minimum integer `q` and `A/U` for 10, 25, 34.3, and 50 TPS;
- sensitivity for any observed `U>1` or acceptance below one; and
- explicit omitted costs.

The analysis fails closed if the PW-0108 hash, device, artifact, trial counts,
physical-read ledger, integrity hash, or safety evidence differs. Unit tests
cover dimensional conversion, ceilings, integer-width rounding, and invalid
inputs.

**Decision gate:** reject `q=16` and `q=32` as Prismwing-50 source-FP8 widths if
their impossible-perfect ceilings are below 50. Reject any future source-FP8
proposal whose measured or bounded `A/U` is below the recomputed 50-TPS
requirement. Do not reject speculation in general if a wider, base-aligned,
route-coherent proposer remains mathematically possible; instead replace
PW-0044's old width prior with the measured cold-storage requirement.

This experiment changes no model, threshold, or runtime. It cannot promote
TPS, authorize training, or prove a wide proposer can preserve capability.

## Result

Not yet executed.
