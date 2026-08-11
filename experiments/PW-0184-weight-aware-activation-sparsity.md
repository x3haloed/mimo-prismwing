# PW-0184 — Weight-aware activation sparsity control

- Status: completed
- Disposition: rejected
- Date: 2026-08-10
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0116
- Execution mode: shadow L3 activation sparsity with unchanged source weights
- Hardware/runtime: existing Apple M1; CPU source-semantic oracle
- Related records: PW-0129, PW-0139, PW-0181, PW-0183

## Contract

Test whether per-token activation sparsity can reduce exact source-weight column
traffic enough to reverse PW-0181's onboard one-TPS lower bound. Use layer 46,
expert 28, positions `112..167`, with source routes, source-FP8 weights, dynamic
input quantization, BF16 boundaries, and expected outputs unchanged. Keep
positions `168..223` sealed.

For both gate/up inputs and the down input, zero exactly the lowest-scoring 25%,
40%, or 50% of channels per row. Freeze two scoring rules: activation magnitude,
and activation magnitude times the combined source-weight column RMS (gate/up)
or down column RMS. Scores only choose source columns; they do not change stored
weights. Add a deterministic fixture for exact cardinality, tie behavior, and
finite output.

The minimum primary point is 25% column traffic avoided: it reduces PW-0181's
1.090015-second miss-acquisition term to at most 0.817512 seconds under the same
deliberately favorable overlap, leaving 0.051268 seconds beyond the 0.131220-
second attention subtotal before one second. This is only a necessary numerical
premise; it does not grant a sparse SSD layout, kernel, endpoint, or TPS.

Promote only if a candidate at at least 25% sparsity reaches complete-expert
validation relative L2 at most 2%, maximum-row relative L2 at most 5%, and
gate/up relative L2 at most 2%. A pass authorizes all-validation routed-layer
testing and a page/column locality experiment. If 25% weight-aware sparsity
exceeds 5% complete error, reject direct channel deletion as the missing
onboard mechanism; larger sparsities are attribution only. Report zero accepted
tokens and leave throughput constants unchanged.

## Result

The authoritative report at
`/Users/chad/Models/mimo-prismwing/evidence/PW-0184/run-001/report.json`
hashes to `6bd4a396d9c4139bf6a60c1c920ae8ff7169040857b093399b97b815350798d7`.

No candidate passes. At the minimum useful 25% sparsity, magnitude scoring
reaches `0.110157` complete-expert validation error and weight-aware scoring
reaches `0.108212`; their maximum rows are `0.134162/0.131193`. Gate/up remain
`0.064557/0.046195` even for the stronger rule. At 40% and 50%, error rises
monotonically to `0.208763` and `0.287188` for weight-aware scoring.

Reject direct activation-channel deletion as the source-exact byte escape.
Weight norms do not repair the approximately fivefold numerical miss at the
smallest physically useful point, so no sparse layout or kernel is authorized.
The holdout remains sealed; zero tokens are accepted and no endpoint TPS or
throughput constant changes.
