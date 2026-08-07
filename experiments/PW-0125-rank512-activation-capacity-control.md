# PW-0125 — Rank-512 activation-weighted capacity control

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: preimplementation contract; clean tree
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  PW-0116 corpus
  `b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e`;
  PW-0119 raw control
  `3e7729dfff3d9ab6793d8e74d29ad20bb3c877bea328ae53d9325737c717c8fb`;
  PW-0122 analysis
  `5b5a21be9438e81e9b05a155ca365cd0dc4180be1b06a18a873362e88f60e0eb`;
  PW-0124 analysis
  `a6c98d0469e2e788e5c54833975277ebcffa822a3d0b426a8bb39dbf3606d32a`
- Hardware/runtime: Apple M1 shared 16 GiB; NumPy Accelerate SVD, PyTorch
  2.13 MPS optimization and CPU source-FP8 evaluation; internal SSD
- Exactness: L4 fitted diagnostic; unchanged source control
- Related records: PW-0045, PW-0108, PW-0115 through PW-0124; E5

## Question and changed premise

PW-0123 and PW-0124 reject the rank-768/four-basis identity form after sharing
is genuinely forced. That does not transfer automatically to PW-0115's
structurally distinct balanced `(r=512,m=8)` form: twice as many shared bases
may reduce identity contention, while rank 512 halves the prospective streamed
factor ratio from 18.75% to 12.50%. But lower expert rank also imposes a
stricter per-expert capacity ceiling. Test that ceiling independently before
building a nine-expert/eight-basis sharing optimizer.

Use the same layer-46 hot expert 28, source targets, positions, and partitions
as PW-0122: 100 train, 56 validation, and 56 untouched pilot-holdout rows.
Replace only rank 768 with rank 512. This is a cheap representation-capacity
control, not a sharing result or endpoint performance experiment.

## Frozen execution

Refactor the existing activation-weighted pilot around an explicit rank in its
immutable specification; preserve PW-0121 and PW-0122 behavior and evidence
names exactly. For PW-0125 use seed `260125`, balanced rank-512 SVD
initialization, the same source-derived per-projection targets, F32 normalized-
MSE surrogate, Adam `0.001`, 100-step maximum, five-step validation interval,
four-check patience, source-FP8/BF16 complete-expert evaluation, factor
non-persistence, MPS 0.60 cap, and Gate 8 observations.

The authoritative PW-0119 rank-512 control for expert 28 is:

| Partition | Relative L2 |
| --- | ---: |
| train | `0.6822727543140975` |
| validation | `0.6730991256068856` |
| pilot holdout | `0.6568507915821798` |
| overall | `0.6747763876584113` |

The already-selected PW-0122 rank-768 activation result is `0.195667` on
validation and `0.288128` on holdout. It is the stronger capacity comparator;
PW-0125 may not call a large improvement over weak rank-512 SVD competitive if
it materially loses the working rank-768 fit.

Add fixtures proving rank injection changes factor shapes, parameter and Adam
byte ledgers, SVD evaluation rank, evidence labels, and thresholds without
changing PW-0121/PW-0122 defaults. Unknown ranks or mismatched baseline
authority fail closed.

## Gates

1. The source oracle remains bit-exact. The authoritative `U,S,Vt` rank-512
   control reproduces PW-0119 within `1e-6`; the balanced initialization stays
   within `5e-6` relative L2 of that authority.
2. Every projection's selected validation normalized MSE is finite and below
   step zero. Selection never reads the holdout.
3. The complete candidate improves at least 25% over rank-512 SVD: validation
   at most `0.5048243442051642` and holdout at most
   `0.49263809368663485`.
4. The complete candidate remains within `1.25x` PW-0122's fitted rank-768
   control: validation at most `0.24458385116689985` and holdout at most
   `0.36016001389755276`.
   These stricter bounds govern the final decision.
5. Gate 8 and zero-current-MPS release requirements remain unchanged. Report
   zero accepted tokens, `A=0`, no endpoint timing, and no TPS.

A pass authorizes only a separately frozen nine-expert/eight-basis forced-
sharing pilot. It does not promote a representation, artifact, kernel, or full
bank. A failure rejects the `(r=512,m=8)` branch before sharing because its
independent capacity cannot approach the already-working rank-768 control; do
not compensate by weakening the gates or inspecting the holdout during
selection. PW-0124's rejected four-basis form remains rejected either way.

## Result

Unexecuted.

## Decision

Unexecuted.
