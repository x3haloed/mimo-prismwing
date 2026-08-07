# PW-0117 — Shared-basis routed-transaction algebra

- Status: complete
- Disposition: scope-decision
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0115 analysis
  `41cc9b745561a09073902ba65354889d6b87e7d8716aea4db85940cbafc9c67a`;
  PW-0116 analysis
  `6007e93aa9cc280d20cab3db0f72851ad9f9722e9f225c07c3c1309cc5ef5e08`;
  inclusionAI/MoBE `7f3501da2a9f7b12d773cb52c454a0be0ceeb185`
- Hardware/runtime: pure algebra plus production-shape operation accounting;
  no checkpoint execution
- Related records: PW-0045, PW-0115, PW-0116; prospective E5

## Question and causal mechanism

PW-0115's physical envelope assumes each layer-shared basis is evaluated once
per routed mixture. The published MoBE implementation represents

`W_e = sigma * A_e * phi(sum_j softmax(alpha_e)_j * B_j)`

and supports `phi` equal to SiLU or tanh. Since `phi` is nonlinear, it cannot
in general commute with the expert-specific basis combination. The selected
experts therefore require separate combined matrices and basis-side work. Test
whether that published nonlinear form can satisfy PW-0045's compute gate, and
prove the exact identity-activation orientation required by the deeper
all-three-projection Prismwing candidate before spending hours fitting it.

This experiment audits executable algebra, not reported MoBE quality. The
paper and implementation remain evidence that learned bases can reconstruct
weights; they are not evidence that nonlinear bases provide the transaction
reuse assumed by PW-0115.

## Frozen representations

Normalize all three MiMo projections to a canonical source matrix
`M_e [p,d]`, where `p=2048` and `d=4096`:

- gate/up: `M_e = W_e`;
- down: `M_e = W_e^T`.

Let `A_e [p,r]`, `B_j [r,d]`, and expert coefficients `c_ej`. Analyze:

1. published nonlinear: `M_e = A_e phi(sum_j c_ej B_j)`;
2. transaction-linear: `M_e = A_e (sum_j c_ej B_j)`.

For gate/up with common input `x`, the linear form must equal
`A_e sum_j c_ej (B_j x)`, so every `B_j x` is evaluated once and reused by all
selected experts. For down with expert intermediate `h_e`, route weight `g_e`,
and `W_e=M_e^T`, the weighted mixture must equal

`sum_j B_j^T [sum_e g_e c_ej A_e^T h_e]`.

Thus expert latent contributions are reduced before each shared output basis.
The analyzer must verify both identities against direct dense reconstruction
on deterministic F64 tiny fixtures. It must also construct a counterexample
showing that applying SiLU or tanh after expert-specific basis combination
cannot be replaced by a linear combination of once-evaluated activated bases.

## Production-shape accounting

For each frozen `(r,m)` in `(768,4)`, `(512,8)`, `(128,32)`, report one
projection's multiplication counts for eight selected experts:

- source: `k*p*d`;
- transaction-linear: `m*r*d + k*p*r + k*m*r`;
- published nonlinear lower bound:
  `k*m*r*d + k*r*d + k*p*r`, excluding activation, softmax, materialization,
  and memory traffic.

Apply the same ratio to all three equal-shape canonical projections. Preserve
PW-0115's storage accounting. The nonlinear form fails the necessary compute
gate if this optimistic lower bound exceeds 50% of source multiplication work.
The linear form remains only physically eligible; it has no fidelity result.

## Gates

Pass only if:

1. direct and transaction-linear gate/up outputs agree within F64 roundoff;
2. direct and latent-reduced down mixtures agree within F64 roundoff;
3. deterministic nonlinear counterexamples differ materially;
4. exact integer operation counts reproduce PW-0115's linear compute envelope
   apart from the newly explicit coefficient-mixing term; and
5. every conclusion names its representation and does not transfer published
   MoBE quality evidence to the untrained identity form.

If all nonlinear lower bounds exceed 50%, reject the published activated form
as a Prismwing transaction architecture. If every linear shape remains below
50%, continue only the explicitly novel `identity-basis-mixture-compiled` form
to PW-0116 weight-reconstruction and activation-weighted fitting. No runtime,
accepted token, endpoint output, or TPS claim is allowed.

## Result

The clean analyzer passed. F64 direct dense and transaction-linear gate/up
outputs agree within `4.441e-16`; direct dense and latent-reduced down mixtures
agree within `1.110e-16`. The frozen counterexample differs by `0.132106` for
SiLU and `0.552382` for tanh, proving that activation after expert-specific
basis combination cannot be replaced by a combination of once-evaluated
activated bases.

For one production-shaped projection, the optimistic nonlinear lower bounds
and exact transaction-linear counts are:

| `(r,m)` | Source multiplies | Nonlinear lower bound | Nonlinear ratio | Linear transaction | Linear ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(768,4)` | 67,108,864 | 138,412,032 | 206.25% | 25,190,400 | 37.537% |
| `(512,8)` | 67,108,864 | 159,383,552 | 237.50% | 25,198,592 | 37.549% |
| `(128,32)` | 67,108,864 | 140,509,184 | 209.375% | 18,907,136 | 28.174% |

The nonlinear counts exclude activation, softmax, materialization, and memory
traffic, yet all already exceed source work by more than 2x and PW-0045's 50%
gate by more than 4x. The small coefficient-mixing term raises PW-0115's
identity ratios only from 37.5% to 37.537%/37.549% and from 28.125% to 28.174%.
All three identity forms remain physically eligible but untrained.

The immutable analysis manifest is
`/Users/chad/Models/mimo-prismwing/evidence/PW-0117/analysis-001/manifest.json`
and hashes to
`337b57c43638025673eb494eecfc87445468d21b9a1ce384952b72f6fa47a910`.
The updated throughput model hashes to
`a914eb9949ae201d109ca2c107088687bf9f3101b67fd17b0dddd5551300c7ad`.
No model ran; there are zero accepted tokens and no TPS claim.

## Decision

Reject the published activated form as the routed-layer transaction
architecture. It remains a storage-oriented quality reference, but its
nonlinearity invalidates the shared-evaluation compute premise.

Continue only a separately named `identity-basis-mixture-compiled` L3/L4 form
to source-weight reconstruction and PW-0116 activation-weighted evaluation.
The down projection must be trained in canonical transposed orientation so
route-weighted expert latents reduce before shared output bases. Do not import
published MoBE quality claims into this untrained identity form.
