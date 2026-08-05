# PW-0017 — Router-union correlation sweep

- Status: complete
- Disposition: scope-decision
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `0bd9d88`; dirty sensitivity tool
- Checkpoint/processor/reference hashes: revision
  `63651580ca774f8504f676040460aed3e1244ac1`; router artifact SHA-256
  `12c1579d28b78dd69ec9342eb9d1f378efc5aa3c2f2a28b5ec73578e6a8bbcdd`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); PyTorch source semantics; no material
  memory pressure
- Related records: PW-0010, PW-0011, PW-0016

## Hypothesis and mechanism

PW-0016 observed only nine unique experts across eight positions, but its input
amplitude was `0.01`, far below an RMS-normalized transformer hidden state and
therefore likely let correction bias dominate. Sweep input correlation at RMS
one to test whether the favorable route union is robust.

## Contract

Target router, synthetic sensitivity evidence. Use the actual layer-43 router,
source noaux_tc semantics, 100 trials per fixed correlation, eight positions,
and independently RMS-normalized standard-normal rows. Seed is 160043.

This experiment must not claim a real activation distribution or endpoint
throughput. Its success condition is distinguishing a robust low union from a
correlation-sensitive one. If median union exceeds nine at correlation 0.99,
PW-0016's `U=1.125` must not be used as a representative DFlash assumption.

## Baseline and candidate

For each trial, draw one base vector and eight noise vectors. At correlation
`rho`, construct `rho*base + sqrt(1-rho^2)*noise`, then independently normalize
each row to RMS one. Correlations are 0, 0.5, 0.9, 0.99, 0.999, and 1.0.

## Isolated attribution

| Input correlation | Median unique experts | p90 | Mean U | Fraction with exactly 8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 54.0 | 57.0 | 6.7825 | 0% |
| 0.5 | 46.0 | 50.0 | 5.7500 | 0% |
| 0.9 | 23.5 | 27.0 | 2.9238 | 0% |
| 0.99 | 12.0 | 14.0 | 1.5163 | 0% |
| 0.999 | 9.0 | 11.0 | 1.1863 | 17% |
| 1.0 | 8.0 | 8.0 | 1.0000 | 100% |

## End-to-end result

No endpoint or real-route result is claimed. The measured router is highly
sensitive to small position-to-position input differences around the narrow
union regime.

## Correctness result

The sweep calls the same source-derived router implementation used by PW-0016,
which already matches the committed noaux_tc semantics and actual router
fixture. Its deterministic output evidence SHA-256 is
`0e0b8c6e35ace02375e78c28c552fb775ceb2b901f175bbe3be4ff2830c515d6`.

External evidence:
`/Volumes/Elements/mimo-prismwing/evidence/PW-0017/layer43-router-union-sweep.json`.

## Decision

The kill condition fired: median union is 12 even at input correlation 0.99.
Supersede any interpretation of PW-0016's low-amplitude `U=1.125` as
representative. DFlash-8 performance now requires actual hidden-state route
traces from an executable source-derived layer/model path; neither independent
synthetic inputs nor nearly identical inputs can substitute for that evidence.
