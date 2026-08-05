# PW-0024 — MiMo RoPE, value-scale, and sink semantics

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `a1bda88`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; source
  `modeling_mimo_v2.py` SHA-256
  `a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad`
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift/Metal runtime compiler; internal SSD
- Related records: PW-0023

## Hypothesis and mechanism

PW-0023's GQA core can represent the remaining MiMo attention semantics if the
wrapper applies partial RoPE to the first 64 of 192 Q/K dimensions, scales V by
0.707 before KV quantization, and the SWA kernel merges each learned sink logit
into the softmax denominator with zero value contribution.

## Contract

Target-faithful source semantics and shapes; deterministic synthetic values and
modified Turbo3/Turbo4 KV representation. Pass only if:

1. `rope_dim = int(192 * 0.334) = 64`; global RoPE base is 10,000,000 and SWA
   base is 10,000; cached K token `t` uses position `t` and the decode Q uses
   the final cache position;
2. V is multiplied by exactly 0.707 before quantization/cache insertion;
3. global attention has no sink; SWA uses one deterministic finite sink bias
   per Q head, adds `exp(sink)` to the stable softmax denominator, and adds no
   value numerator;
4. an independent scalar reference and Metal agree for all 8,192 outputs at
   relative L2 at most `3e-4` and maximum absolute error at most `5e-4`, with
   every head guard intact;
5. Turbo3 and Turbo4 each pass global and SWA context 128. Report batch one,
   concurrency one, one accepted token, 10 warm-ups, 30 measurements, bytes,
   cold/warm wall and GPU time, hardware, and commit. `A` and `U` are not
   applicable.

Passing promotes these semantics into the transformer-layer fixture only. It
does not validate learned sink values, real activations, quantized model
fidelity, or endpoint performance.

## Baseline and candidate

Baseline is a scalar implementation of the pinned source equations. Candidate
is PW-0023's GQA Metal schedule with explicit sink-state merging. Both consume
identical RoPE-transformed Q/K and value-scaled packed V.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0024`.

## Isolated attribution

All runs use context 128, batch one, concurrency one, one accepted token, 10
warm-ups, and 30 measurements:

| Format/mode | GPU median / p95 ms | Wall median / p95 ms |
| --- | ---: | ---: |
| Turbo3 global | 3.396 / 4.227 | 3.669 / 4.983 |
| Turbo3 SWA | 3.464 / 4.507 | 3.782 / 5.020 |
| Turbo4 global | 3.228 / 4.192 | 3.548 / 4.867 |
| Turbo4 SWA | 3.410 / 4.369 | 3.750 / 4.844 |

The wrapper records RoPE dimension 64, Q position 127, bases 10,000,000 and
10,000, value scale 0.707, and sink presence per mode. Packed buffers are warm
with no model/storage I/O. `A` and `U` are not applicable. The added semantics
do not materially alter PW-0023's context-128 timing.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All five conditions pass. Every one of 8,192 outputs and all 128 guards are
checked per run. Metal-versus-scalar relative L2 is at most `4.09e-7`, below
`3e-4`; maximum absolute error is at most `3.58e-7`, below `5e-4`.

Global mode applies partial RoPE and value scaling without a sink. SWA uses its
distinct RoPE base and merges one deterministic sink state per Q head into the
stable denominator with no numerator contribution.

Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0024`. The SHA-256 of its
`SHA256SUMS` manifest is
`7acbc17b6037187028f0b3c4a7424dd06ef671fc35de59e1c59ec7c211d34fba`.

## Decision

Promote partial RoPE, pre-cache value scaling, and SWA sink-state merging into
the transformer-layer fixture. The next correctness rung requires learned
QKV/output projection, norm, and sink tensors plus real layer input; synthetic
semantics cannot substitute for those values.
