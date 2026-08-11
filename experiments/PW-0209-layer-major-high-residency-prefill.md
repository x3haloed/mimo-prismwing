# PW-0209 — Layer-major high-residency prefill

- Status: proposed
- Disposition: unexecuted
- Date: 2026-08-10
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

Unexecuted. This is a separate prefill branch and cannot be reported as decode
TPS.
