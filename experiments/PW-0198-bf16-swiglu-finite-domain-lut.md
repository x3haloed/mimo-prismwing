# PW-0198 — BF16 SwiGLU finite-domain lookup

- Status: completed
- Disposition: rejected; output is unchanged
- Date: 2026-08-10
- Execution mode: target-faithful layer-local numerical falsifier
- Hardware/runtime: existing Apple M1 and internal checkpoint storage
- Related records: PW-0196, PW-0197

## Hypothesis

PW-0197's remaining backend variance is introduced by evaluating the SiLU
transcendental independently on CPU and Metal. Because the source rounds gate
projection outputs to BF16 before SiLU, the nonlinear input has only 65,536
bit patterns. A 256-KiB resident lookup generated with the readable Rust source
semantic can make that boundary deterministic while removing an exponential
from every expert activation.

## Contract and gate

Preserve PW-0197's input, source reference, routes, direct-checkpoint bindings,
specialized widths, and unchanged `2e-5` relative-L2 / `2e-4` maximum-error
gates. Generate the lookup deterministically in-process, hash it, and validate
the new lookup kernel against a CPU fixture before the real run. The lookup is
an application buffer on the existing M1, not a hardware sidecar. Require zero
source-copy bytes and a warm median no worse than 30 ms. Record zero accepted
tokens.

If parity remains materially unchanged, reject backend SiLU as the dominant
cause and isolate projection reduction. If it passes, promote the finite-domain
lookup into the wide verifier only after its complete-path timing passes.

## Result

The deterministic lookup kernel passes its exact CPU fixture, but the complete
wide output SHA-256 is
`05224cf40a7950a9cbf153382d56718778ed57bd880e71bebd4462472a1ddc1b`,
identical to PW-0197's rejected output. Relative L2 remains `0.00272864` and
maximum absolute error remains `2.38419e-7`. Reject SiLU transcendental variance
as a contributor on this fixture. Preserve the finite-domain lookup idea as a
possible performance mechanism only; it has no correctness value here. Next
isolate batched projection reduction topology. Zero tokens are accepted.
