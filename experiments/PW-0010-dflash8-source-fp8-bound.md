# PW-0010 — DFlash-8 source-FP8 throughput bound

- Status: complete
- Disposition: rejected
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: based on `a3fe6a5`; dirty derived analysis
- Checkpoint/processor/reference hashes: base revision
  `63651580ca774f8504f676040460aed3e1244ac1`; DFlash source SHA-256
  `da5ab1738b954800950405131f1d1d97c3345f37e32676d511d3a25dfddd9d75`;
  PW-0008 candidate evidence hash
  `1312ded931be10c93fe7c9f5fb8357281a520c5469146c6d95bc270e8cc95814`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); PW-0008 warm shared Metal buffers
- Related records: PW-0002, PW-0008, PW-0009

## Hypothesis and mechanism

The published DFlash block of eight might supply enough accepted-token leverage
to make the measured source-FP8 projection path reach 50 TPS on the M1.

## Contract

Target-faithful, greedy L2 speculation. This is a necessary routed-weight bound,
not endpoint TPS. Give the candidate every optimistic advantage: perfect full-
block acceptance, the minimum possible expert union, no dense/attention/draft/
KV/sampling/synchronization cost, warm resident buffers, and PW-0008's measured
projection bandwidth across all routed bytes.

Reject this architecture if its optimistic ceiling is below 50 TPS. Passing
the bound would not promote it; a complete endpoint would still be required.

## Baseline and candidate

PW-0008 measured 37.764771 GiB/s. PW-0002 derives 9,464,659,968 routed source
bytes per ordinary token, or 8.814651 GiB. DFlash source and config fix
`block_size = 8`; its loop commits `acceptance_length + 1`, so one verification
pass commits at most eight output tokens.

For expert-union factor `U`, at least one one-token-equivalent expert set must
be consumed by any non-empty target pass. Therefore `A <= 8` and `U >= 1`.

## Isolated attribution

```text
ordinary routed-only rate = 37.764771 / 8.814651 = 4.284318 TPS
required A/U for 50 TPS   = 50 / 4.284318       = 11.670469
maximum DFlash-8 A/U      = 8 / 1               = 8
optimistic DFlash-8 ceiling = 4.284318 * 8       = 34.274545 TPS
```

The source-FP8 path would require at least 55.091572 GiB/s merely to make 50
routed-only TPS at the impossible-to-improve `A/U = 8` bound.

For context only, applying the same measured bandwidth to the unbuilt
groupwise-INT4 estimate yields a 60.947401 routed-only TPS ceiling. It requires
`A/U >= 6.563036`; even with `A = 8`, that permits only `U <= 1.218948`, or an
average union no larger than about 9.75 experts per layer across eight verified
positions. This is a candidate constraint, not evidence that an INT4 kernel or
such route overlap exists.

## End-to-end result

No endpoint was run and no endpoint TPS is claimed. The optimistic component
ceiling is already 31.5% below the target before every omitted cost.

## Correctness result

The bound uses the target-faithful source byte count, the selected correct
PW-0008 Metal kernel, and the greedy-only DFlash interpretation established by
PW-0009. It does not rely on the incompatible bundled DFlash target weights.

Machine-readable constants and formulas are recorded in
`spec/throughput-model.json`. The underlying raw benchmark hashes remain in
PW-0008; the DFlash source identity remains in PW-0009.

## Decision

Reject published DFlash-8 plus source-FP8 routed execution on the measured M1
kernel as a Prismwing 50 architecture. Retain DFlash for experiments with a
materially smaller executable representation, a proven faster full routed
kernel, or an explicitly new longer-block draft. The next cheap INT4 kill test
is actual `A/U`: a full-block union above 1.218948 kills even its otherwise-free
routed-only budget at the current measured bandwidth.
