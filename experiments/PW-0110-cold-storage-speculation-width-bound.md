# PW-0110 — Cold-storage speculation width bound

- Status: completed
- Disposition: scope-decision
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Contract commit: `61cfc899455f938bc4f9628b1b27bd143a0a7fd9`
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

The clean analyzer at
`41f1dd3deeea997da577cb0bb116ef3d757c2eea` independently recomputes the
PW-0108 cold medians as 72.875, 59.094, and 58.034 ms for one, two, and three
Metal-I/O command buffers. It selects three buffers and derives a 47-layer
acquisition floor of 2,727.590151 ms per verifier block at impossible-perfect
`U=1`. Every underlying cold trial loads 201,375,744 authenticated tensor bytes,
records 201,719,808 physical read bytes, completes every command, and preserves
one destination hash. Gate 8 authority also revalidates.

Ideal accepted-TPS ceilings, with every omitted cost free, are:

| Width `q` | Ceiling at `A=q`, `U=1` |
| ---: | ---: |
| 8 | 2.933 TPS |
| 16 | 5.866 TPS |
| 32 | 11.732 TPS |
| 64 | 23.464 TPS |
| 128 | 46.928 TPS |
| 137 | 50.227 TPS |

The necessary `A/U` requirements are 27.276 for 10 TPS, 68.190 for 25 TPS,
93.556 for 34.3 TPS, and 136.380 for 50 TPS. Even with complete acceptance and
minimum union, those imply integer widths `q=28`, `69`, `94`, and `137`
respectively. Any `U>1` or acceptance below 100% raises width proportionally;
all dense/attention acquisition, target arithmetic, draft execution, KV work,
barriers, correction, rollback, sampling, and sustained effects remain omitted.

The immutable analysis at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0110/analysis-001/manifest.json`
hashes to
`844047de4d009d0d7bd6f803e56e097ee6efce66e6a0c2c7d96315962a5cd8b6`.
The updated throughput model hashes to
`41cd3fa7f17bd06f65c23c94347a61b4f250b67c3d663ec261df9c36b6388bba`.

## Decision

Reject `q=16` and `q=32` as source-FP8 internal-SSD widths for both the formal
50-TPS target and the valuable 34.3-TPS horizon. Replace the old “at least 16,
probably 32” prior with a necessary minimum of `q=137` for 50 TPS and `q=94`
for 34.3 TPS under impossible-perfect acceptance/union. This does not prove a
137-wide proposer is executable or useful; it establishes the minimum bar any
base-aligned route-coherent speculation branch must clear before training or a
wide verifier is built.
