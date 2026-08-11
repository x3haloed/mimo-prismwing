# PW-0197 — Specialized-width direct-checkpoint source-BF16 MoE

- Status: completed
- Disposition: rejected at the frozen relative-error gate
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0187 routes
- Execution mode: target-faithful layer-local wide verifier
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0101, PW-0193, PW-0195, PW-0196

## Hypothesis

PW-0195's compile-time expert-width scheduling remains useful after restoring
the exact source numerical boundaries proven sufficient by PW-0196. GPU-resident
dynamic group-128 activation quantize/dequantize before gate/up and down,
BF16-staged SwiGLU, BF16 down staging, and BF16 final route reduction will match
an independently generated source authority without forfeiting the wide route
union's amortization.

## Contract and gate

Generate a new source-BF16 reference from PW-0192's deterministic eight-row
input and the authenticated PW-0187 static routes. Preserve its manifest and
hash independently of the L3 reference. Execute all 17 real experts and 64 placements from original
checkpoint shards with page-rounded no-copy source bindings and compile-time
width selection. New BF16 staging must have a deterministic fixture before the
real run. Require relative L2 at most `2e-5`, maximum absolute error at most
`2e-4`, finite output, zero source-copy bytes, exactly 64 executed rows, and a
warm median no worse than 30 ms. Record `A=0`: this is one layer, not endpoint
accepted throughput.

If correctness fails, isolate the first semantic boundary rather than weakening
the gate. If timing fails, preserve PW-0195 as an L3 diagnostic and reject this
implementation as the wide target-faithful primitive.

## Result

The GPU-resident semantic path runs all 17 experts and 64 real placements, and
its new BF16 staging fixture is exact, but the routed output fails the unchanged
relative gate: relative L2 is `0.00272864` versus the required `2e-5`. Maximum
absolute error is only `2.38419e-7`, but the reference norm is only
`0.00202957`; 13,814 of 32,768 BF16 outputs differ. The reference manifest,
reference, and rejected diagnostic output hash respectively to
`204748ae819fd24cb17eb69060682540f3939b96a93ae88232830c17c08423a9`,
`66631914ee789a682ca365c289857432f83c8e8d9358fc806c4ea306fa52a9e1`,
and `05224cf40a7950a9cbf153382d56718778ed57bd880e71bebd4462472a1ddc1b`.

Reject the wide primitive under the source contract without reporting timing.
PW-0196 showed that the boundary can be exact for one projection; PW-0197 shows
that exact BF16 boundary placement alone does not make the full batched expert
transaction backend-invariant. Next isolate the finite BF16 SwiGLU map from
projection reduction topology. Zero tokens are accepted.
