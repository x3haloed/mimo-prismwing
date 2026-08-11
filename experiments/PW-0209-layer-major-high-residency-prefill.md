# PW-0209 — Layer-major high-residency prefill

- Status: completed
- Disposition: rejected at the frozen layer gate; incremental layer-major gain preserved
- Date: 2026-08-11
- Execution mode: L1 target-faithful first; arithmetic variants named separately
- Related records: PW-0050, PW-0092, PW-0205, PW-0207

## Hypothesis and mechanism

Decode and prefill are not the same physical workload. Decode repeatedly
streams weights for little token width; prefill can reuse each acquired tile
across many prompt positions. PW-0205 still spends about 207 seconds in its
small-prompt prefill. Research on newer Apple GPUs separates compute-rich
prompt processing from memory-bound decode, but its M5 tensor units do not
exist on the M1.

Hypothesis: an M1-specific layer-major prefill that retains bounded activation
and KV arenas, acquires each projection/expert tile once per layer-wide token
batch, and uses width-specialized Metal kernels can reduce 8K TTFT by at least
4x relative to the current corrected endpoint without changing logits.

## Contract

Start with a deterministic tiny causal-attention fixture, then a real sampled
layer and corrected first-token logit comparison. Preserve masks, RoPE,
sliding/global attention schedule, router decisions, and processor semantics.
Declare all arenas under PW-0207's pressure-elastic policy. Do not cite prompt
tokens/s from M5, MLX, or a dense model as M1/MiMo evidence.

## Cheap falsifier and gates

Build a byte/time model for prompt lengths 128, 1K, and 8K using measured M1
projection throughput and exact live-state bytes. Kill if the 8K arena cannot
fit below 12 GiB or the roofline cannot support 4x TTFT improvement.

Then test one complete real layer at multiple widths. Require source-derived
parity, at least 3x layer-complete speedup at width 128 or greater, and no
hidden weight rereads. Only a whole corrected text prefill may establish TTFT;
the final acceptance target remains 15 seconds at 8K.

## Decision

The arena and roofline falsifier passes the capacity and 4x-continuation gates:
the declared 8K F32 arenas occupy 5.441 GiB, and the optimistic layer-major
projection floor has 403.047x headroom over the scaled width-eight control.
The same model rejects 15-second TTFT under the measured source-FP8 transport
premise. Its report hashes to
`c3cd34d8754bd7c642918837817fe378efd82d92a1e269af02e7c5cdee173b64`.

The real layer-43, context-128 authority covers all 128 rows, 225 unique
experts, 1,024 placements, and a maximum expert width of 26. The corrected
dynamic-FP8/BF16 source reference hashes to
`8fdafc50400bc323d5fca0c30e5026f179088463ef2a9e5f7fc6981f4f933cd3`;
its report hashes to
`3e9c123a2309f58a2bb1f1b636db0215af6eb4a02fbc27da848bfae2da48ada5`.

Real execution exposed and fixed a host-side route-buffer defect: the runtime
used the 24-byte `Vec` descriptor size instead of the backing slice size for
route weights and positions, silently truncating placement entries after six.
The production helper now has an eight-placement regression fixture. With the
fix, compile-time widths 9 through 32 and the full width-128 transaction are
byte-identical to sixteen width-eight controls. Both accelerated outputs hash
to `b3c6daf1b0efc5f684fdef5826eb0dcca9f46042e3e1b7a4661799d6e14f6737`.

The unchanged absolute source gate still fails for both paths equally at
`0.0007579843` relative L2 and `0.015625` maximum absolute error. This is the
previously documented L3 source-BLAS association boundary; no threshold is
weakened and zero tokens are accepted. Consequently the branch cannot be
promoted through this layer-local gate.

Cold-requested A-control-A timing preserves a smaller but repeatable-looking
advance. Width 128 completes in 4,079.447 and 3,976.484 ms around a 4,738.247
ms width-eight control, a 1.161492x to 1.191567x complete-layer speedup. It
reduces logical repeated source work from 21,563,985,920 to 5,667,888,128
bytes, while every trial physically reads about 5.670 GB. The validated
failure report hashes to
`9a0933a132a795ed6a4a6873e90652912969a50945d675f032e75afbf499f316`.

Reject the predeclared 3x layer gate and therefore the large whole-prefill
implementation under this storage premise. Preserve the 16.1%--19.2%
layer-local gain, the byte-exact width equivalence, the route-buffer fix, and
the full-width kernels for a changed critical cut or composition with a
separately validated fusion. This is prefill diagnostic work, not decode TPS
or accepted TTFT.
