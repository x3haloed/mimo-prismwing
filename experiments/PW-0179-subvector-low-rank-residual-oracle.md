# PW-0179 — Subvector-code low-rank residual oracle

- Status: complete
- Disposition: rejected
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Model/reference: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0178 run-001
- Execution mode: favorable shadow L3 weight-domain repair oracle beside the
  target-faithful source expert; candidate state never enters the model
- Related records: PW-0131, PW-0132, PW-0178; E5

## Contract

Test whether PW-0178's large error residual has a compact, input-independent
low-rank program rather than attempting another activation-fit repair. Reuse
and authenticate PW-0178's exact UINT8 indices and FP16 codebooks. For each
gate/up/down weight residual, compute one frozen rank-128 randomized SVD with
four power iterations and deterministic seeds; store FP16 left/right factors
and evaluate nested ranks 16, 32, 64, 96, and 128 by prefix only.

Add a tiny fixture proving dense code weight plus two-factor residual semantics
before the real run. Use source BF16 projection/SwiGLU boundaries and all 56
validation positions. Keep the pilot holdout sealed. Source replay must be
bit-exact.

Promotion requires complete-expert relative L2 at most 2%, maximum-row error at
most 5%, every gate/up projection at most 2%, rank at most 96, combined code
and factor bytes at most 75% of the 13,369,344-byte affine-INT4 expert, and
factor MACs at most 8% of source expert MACs. Rank 128 is diagnostic only.

If rank 128 remains above 5% complete-expert validation L2, kill low-rank
weight residuals on this two-bit core. If rank 128 is below 5% but rank 96
misses, retain only a non-low-rank trained residual hypothesis; do not increase
rank until the traffic advantage disappears. Report zero accepted tokens and
no TPS.

## Result

The valid report hashes to
`afbf05fde482f234f2bf6f19176cdf363d25835ae282407f3d39436a9fe9d4df`.
Source replay is bit-exact and every safety gate passes.

Reject the residual at every rank. Complete-expert relative L2 changes only
from PW-0178's `0.207785` to `0.206096/0.201955/0.198695/0.198209/0.196296`
at ranks 16/32/64/96/128. Rank-128 gate/up errors remain
`0.108825/0.077837`. It captures only `23.28%/23.18%/40.11%` of gate/up/down
residual energy while growing the expert representation to 11,010,048 bytes
(`0.823529` of affine INT4) and repair work to `9.375%` of source MACs.

Kill low-rank weight residuals on this two-bit core. Rank 96 is already the
last physically promotable point and remains at `0.198209` complete error.
Do not increase rank or build a packed kernel. Only a non-low-rank trained
representation remains distinct. The holdout stays sealed; zero tokens are
accepted and no TPS or throughput constant changes.
