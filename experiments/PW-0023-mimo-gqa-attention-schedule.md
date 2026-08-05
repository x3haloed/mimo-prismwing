# PW-0023 — MiMo global/SWA GQA attention schedule

- Status: complete
- Disposition: conditional
- Date: 2026-08-04
- Owner: Codex with project owner authorization
- Commit and dirty state: contract committed as `afb6cb0`; implementation dirty
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; PW-0020 Atomic source lock
- Hardware, OS, compiler, storage, memory pressure: Apple M1; Macmini9,1;
  16 GiB; macOS 26.4.1 (25E253); Swift/Metal runtime compiler; internal SSD
- Related records: PW-0021, PW-0022

## Hypothesis and mechanism

PW-0022's reduction can be scheduled as one threadgroup per MiMo query head,
mapping 64 Q heads onto four global-attention KV heads or eight sliding-window
KV heads. This should preserve scalar GQA semantics and expose complete
attention-core cost per layer without yet conflating projection or MoE work.

## Contract

Target-faithful head counts, K/V dimensions, GQA mapping, and window lengths;
modified Turbo3/Turbo4 KV representation. The candidate passes only if:

1. global mode uses 64 Q heads, four KV heads, K=192 padded to 256, V=128,
   and maps Q head `h` to KV head `h / 16`;
2. SWA mode uses 64 Q heads, eight KV heads, the same K/V dimensions, exactly
   128 cached tokens, and maps Q head `h` to KV head `h / 8`;
3. Metal agrees with an independent scalar reference for every output element
   at relative L2 at most `3e-4` and maximum absolute error at most `5e-4`, all
   64 head guards remain intact, and outputs are finite;
4. Turbo3 and Turbo4 each pass global contexts 128, 1,024, and 8,192 plus SWA
   context 128;
5. each performance run uses batch one, concurrency one, one accepted token,
   10 warm-ups, and 30 measurements, reporting cold and warm wall/GPU time,
   bytes read, hardware, commit, and warm packed buffers. `A` and `U` are not
   applicable to this attention component.

No performance threshold is predeclared for the first complete GQA schedule.
Results are layer-component diagnostics, not endpoint TPS. Passing promotes
only the schedule to the transformer-layer integration branch; real activation
and model-fidelity gates remain mandatory.

## Baseline and candidate

Baseline is a scalar CPU GQA loop using PW-0021's packed dequantization and
attention oracle. Candidate dispatches 64 PW-0022-style threadgroups in one
Metal command. Global and SWA modes have distinct names and evidence.

Raw evidence will be written under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0023`.

## Isolated attribution

All runs use batch one, concurrency one, one accepted token, 10 warm-ups, and
30 measurements:

| Format/mode | Context | Bytes read | GPU median / p95 ms | Wall median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Turbo3 global | 128 | 142,336 | 3.326 / 4.525 | 3.655 / 4.967 |
| Turbo3 global | 1,024 | 679,936 | 16.795 / 18.246 | 17.283 / 18.695 |
| Turbo3 global | 8,192 | 4,980,736 | 132.076 / 143.666 | 132.413 / 144.042 |
| Turbo3 SWA | 128 | 219,136 | 3.480 / 4.850 | 3.796 / 5.464 |
| Turbo4 global | 128 | 169,984 | 3.217 / 4.404 | 3.533 / 4.713 |
| Turbo4 global | 1,024 | 901,120 | 16.229 / 17.079 | 16.660 / 17.957 |
| Turbo4 global | 8,192 | 6,750,208 | 125.479 / 130.978 | 125.873 / 131.359 |
| Turbo4 SWA | 128 | 274,432 | 3.464 / 4.562 | 3.736 / 4.840 |

First cold GPU/wall times range from 3.480/6.497 ms to
128.693/131.379 ms. Packed buffers are application-warm with no model or
storage I/O. `A` and `U` are not applicable.

This schedule exposes a serious layer budget: reusing these component medians
at context 8,192 gives approximately 1.19 seconds for nine Turbo4 global
attention cores plus 0.135 seconds for 39 SWA cores, before projections,
norms, MoE, MTP, sampling, or any endpoint work. That is a bottleneck
diagnostic, not an endpoint throughput claim or optimized lower bound.

## End-to-end result

Out of scope; no endpoint TPS claim is permitted.

## Correctness result

All five conditions pass. The kernel maps 64 Q heads to four KV heads at GQA
ratio 16 and to eight KV heads at ratio eight. SWA uses exactly 128 tokens.
Every one of the 8,192 output values is checked on every run and all 128 guard
values remain intact.

Metal-versus-scalar relative L2 is at most `2.29e-6`, below `3e-4`; maximum
absolute error is at most `4.77e-7`, below `5e-4`. Outputs are finite across
both formats and all contexts.

Raw evidence is under
`/Volumes/Elements/mimo-prismwing/evidence/PW-0023`. The SHA-256 of its
`SHA256SUMS` manifest is
`1208a20a5830e6a753f078a982887da0eca6680d1f147f64017e0945b51bd001`.

## Decision

Promote the exact global/SWA GQA schedule to the transformer-layer integration
branch. Do not promote its performance: multi-head attention is now a measured
major bottleneck, not an omitted constant.

The cheapest next optimization is sharing each KV-head scan across its 16 or
eight Q heads instead of independently dequantizing K/V per Q head, while
retaining this schedule as the correctness control. Transformer-layer work can
proceed in parallel conceptually, but no layer or endpoint performance claim
may treat attention as free.
