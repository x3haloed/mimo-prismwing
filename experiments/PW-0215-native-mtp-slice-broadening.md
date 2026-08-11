# PW-0215 — Native MTP complete-path slice broadening

- Status: in progress; code and multilingual slices complete
- Disposition: conditional positive
- Date: 2026-08-11
- Execution mode: L2 target-distribution-preserving native MTP draft under exact verifier-only commit
- Hardware: Apple M1 Mac mini, 16 GiB, batch one, concurrency one, internal APFS checkpoint
- Related records: PW-0208, PW-0211, PW-0214

## Question and inherited contract

PW-0211 conditionally promotes a repeatable ordinary-text lower milestone but
withholds a general default until code, multilingual, rare-route, longer-context,
and held-out complete paths are measured. This record advances that already
predeclared slice-broadening branch. Missing 2x or 50 TPS does not kill a slice;
every repeatable positive complete-path gain is preserved.

For each category, run cold candidate-control-candidate from clean processes on
the same prompt, checkpoint, kernel, commit, seven-token request, batch size,
and concurrency. The candidate uses the authenticated three-layer native MiMo
MTP q4 proposer against live target state. The control uses seven same-model
proposal steps and an ordinary q8 verifier. Only verifier-authorized tokens are
observable. Record accepted tokens, per-transaction `A` and `U`, prefill,
proposal, verification, complete wall, logical and physical bytes, residency,
and safety state. Promote a category only when both candidates emit identical
target output and beat the interleaved control end to end. General promotion
still requires the remaining required categories and holdouts.

The first attempted code launch pointed at a relocated external checkpoint
whose inode identity does not match the pinned installation receipt. It failed
closed before inference and created no accepted report. A redundant full rehash
of that 294-GB copy was stopped after the original internal APFS installation
was recovered. The internal installation exactly matches the receipt's sizes,
inodes, and mtimes; only the previously authorized mount-device drift remains.

## Code result

All three clean processes emit token IDs
`[8420,374,264,4583,8129,315,264]`, decoded as
`Here is a complete implementation of a`. Both candidates run two q4
transactions, accept three draft tokens in each, and report `U` values
`5.505319` and `5.579787`. The q8 control runs one transaction, retains six
proposal rows, and reports `U=4.672872`.

| Measure | Candidate 1 | q8 control | Candidate 2 |
| --- | ---: | ---: | ---: |
| Prefill wall ms | 328,576.022 | 333,451.975 | 337,808.341 |
| Proposal wall ms | 22,504.815 | 157,691.946 | 22,197.196 |
| Verification wall ms | 51,528.642 | 38,508.553 | 52,786.967 |
| Complete wall ms | 403,286.409 | 530,360.080 | 413,746.414 |
| Logical source bytes | 533,597,988,224 | 628,197,122,944 | 533,597,988,224 |
| Process disk bytes read | 527,122,358,272 | 628,911,505,408 | 527,159,226,368 |
| Conservative peak resident bytes | 4,507,009,024 | 379,830,272 | 4,528,406,528 |

Candidate complete walls differ by 2.5605%. Their 408,516.411-ms median is
`0.0171352` accepted TPS versus the control's `0.0131986`, a repeatable
`1.298259x` complete-request gain. Post-prefill proposal plus verification falls
from 196,200.499 ms to a 74,508.810-ms candidate median, a `2.633252x` wall
gain. Candidate logical source traffic falls 15.06%; measured process reads fall
16.18%. Every run records zero swap growth, zero newly throttled pages, and no
protected-service loss. Minimum free memory is 53% for the candidates and 56%
for the control.

Report SHA-256 values are:

- candidate 1: `4237717083f8ddb5b1e9d0e2ecac3de8440f56c363f4abd44a3da1f9ea71e2cf`
- control: `49e5140a2aa2250b45385bea78d19285637293ee222007514fbc55e90e03020e`
- candidate 2: `84610209c4022626c0c9dba607974c7ddffbce54bb7a03ba3cbc3f16e5b6aa5a`

Their progress-log hashes are respectively
`feb9f55811c2e92ee2e7c47ad0792e13ff70b5a8766357da34413db1896f79a1`,
`06a0af13149597256ddb2d0366d7b600bd556da3c28a6bc902d8d5db94b471e3`,
and `cddf707350806ca82358d151debd2f5973388fba79703f4c471b5700cf0efbf5`.
All reports bind clean commit `180491db5039d0e72213f3c4bb040ba7165688c3`.

## Decision and continuation

Promote native q4 as a conditional code-slice lower milestone. This expands the
PW-0211 result beyond ordinary text without making it a general default. The
next falsifier was multilingual candidate-control-candidate because its PW-0211
pilot accepted only one token.

## Multilingual result

The complete live request reverses the isolated pilot's pessimistic acceptance
prior. All three processes emit token IDs
`[52510,101353,20412,116180,52510,18493,99604]`, decoded as
`水循环是地球上水在不同`. Both candidates accept three draft tokens in each of
two q4 transactions, with `U=5.526596` and `U=4.904255`. The q8 control retains
six proposal rows and has `U=4.226064`.

| Measure | Candidate 1 | q8 control | Candidate 2 |
| --- | ---: | ---: | ---: |
| Prefill wall ms | 292,364.411 | 285,928.421 | 291,793.060 |
| Proposal wall ms | 20,497.074 | 149,100.333 | 21,111.586 |
| Verification wall ms | 52,170.836 | 35,988.045 | 52,242.771 |
| Complete wall ms | 366,098.966 | 471,685.581 | 365,856.195 |
| Logical source bytes | 466,263,613,952 | 559,730,354,176 | 466,263,613,952 |
| Process disk bytes read | 459,381,829,632 | 560,355,995,648 | 459,634,053,120 |
| Conservative peak resident bytes | 4,525,424,640 | 358,678,528 | 4,514,021,376 |

Candidate complete walls differ by only 0.0663%. Their 365,977.581-ms median
is `0.0191269` accepted TPS versus the control's `0.0148404`, a repeatable
`1.288837x` complete-request gain. Post-prefill wall falls from 185,088.378 ms
to a 73,011.134-ms candidate median, a `2.535071x` gain. Logical traffic falls
16.70%; process reads fall 18.00%. Every run again records zero swap growth,
zero newly throttled pages, and no protected-service loss.

Multilingual report hashes are
`c8c4b2ac938fbf8820bb13ef12c2f6e0b5ae7e141e62ebb04cbc4e8c3330a48e`,
`6a8632c40732b8a6150967b5ecf73cc5f95587a386f4dc6166588fafe66df83f`,
and `1bdddc7f2bc17df1731a76394b9d0c29837bf2956d1a1ddf61a4e9c73eeb8710`.
Their progress hashes are respectively
`1f29ee6ac797c08fa34fd860ea8b19640fce8ad9c7033136a234e97f6275029d`,
`f7290b1913fcfd84693a3b0935d60c60ae97d289e7f1004365d4d716c5ab6c48`,
and `e7bbc9f135170ab7c37a3274cc8a3395575d050a92d46baf85d1e05ab798c436`.

Promote multilingual as a second conditional slice lower milestone. Rare-route
is the next complete-path falsifier. Longer-context and untouched holdouts
remain required after category broadening.
