# PW-0117 — Shared-basis routed-transaction algebra

- Status: proposed
- Disposition: unexecuted
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

Unexecuted.

## Decision

Unexecuted.
