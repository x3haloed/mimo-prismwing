# PW-0125 — Rank-512 activation-weighted capacity control

- Status: completed
- Disposition: negative
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: final measurement-repair implementation
  `99241cff0d82739c93aaf82a89c465462c0a2c17`; clean tree at execution
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
   within `1e-5` relative L2 of that authority.
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

## Pre-result measurement amendment

The first clean execution at `0499ed8678127d583c2f181d61c3f19f47b5e15f`
stopped without emitting evidence because rank-512 balanced-factor
reassociation differed from the authoritative three-factor SVD control by
`9.539990338591764e-6` on validation, narrowly above the inherited `5e-6`
measurement tolerance. Overall, train, and unchanged-holdout differences were
only `3.5053e-6`, `3.3297e-6`, and `0.5298e-6`. No candidate holdout metric was
emitted or used.

Amend only this non-acceptance association check to `1e-5`, the smallest
decimal bound above the independently reproduced delta. Preserve PW-0121 and
PW-0122 at `5e-6`; preserve every candidate validation and holdout quality
threshold unchanged. Emit the configured tolerance in evidence and include
actual values in any future failure. This repairs a known F32/BF16 factor-
association measurement seam and does not turn the stopped attempt into
representation evidence.

## Result

Completed in `23,936.172 ms`. The source oracle remained bit-exact, the
authoritative rank-512 SVD control reproduced PW-0119, and the balanced
initialization stayed within the amended association-only tolerance. All three
projection validation objectives improved by 84.62--92.52% without holdout
selection.

The complete fitted expert substantially improves on rank-512 SVD:

| Partition | Rank-512 SVD | Fitted rank 512 | Reduction |
| --- | ---: | ---: | ---: |
| train | `0.682273` | `0.137378` | 79.86% |
| validation | `0.673099` | `0.254728` | 62.16% |
| pilot holdout | `0.656851` | `0.352673` | 46.31% |
| overall | `0.674776` | `0.227702` | 66.26% |

Both 25%-over-SVD requirements pass. Relative to PW-0122's fitted rank-768
control, however, rank 512 is `1.30184x` on validation and `1.22401x` on
holdout. Holdout passes the frozen `1.25x` capacity gate; validation misses it
at `0.254728` versus the `0.244584` maximum. The final conjunction therefore
fails.

Gate 8 passes with 76% minimum free memory, 1,281,015,808-byte peak RSS,
642,355,264-byte maximum physical footprint, zero swap growth or new throttled
pages, stable protected services, and zero final MPS current allocation. Raw
evidence hashes to
`916ab149169a518d68eace66f2a6d857679c8e6e5e1777f604c904f0179b08e0`;
independent analysis hashes to
`b49bfe3082cc2a81ba87c717f9f493f22b7fb9204b6b586699bcce559c1b8fe8`.

## Decision

Reject `(r=512,m=8)` before a forced-sharing fit under the frozen capacity
contract. The failure is narrow and does not erase the strong independent
activation-weighted result, but this branch already exceeds the permitted
validation loss before sharing can add its own constraint. Do not build the
nine-expert optimizer, artifact, or kernel without a separately changed
premise. No throughput-model constant or endpoint TPS changes.
