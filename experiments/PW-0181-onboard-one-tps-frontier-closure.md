# PW-0181 — Onboard one-TPS frontier closure

- Status: complete
- Disposition: frontier exhausted without reaching 1 accepted TPS
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- System: existing 16 GiB Apple M1 only; no new hardware or sidecar probe
- Mode: analytical synthesis of authenticated endpoint, transaction, cache,
  representation, and training evidence; no new performance claim
- Related records: PW-0100, PW-0104, PW-0108, PW-0111, PW-0112,
  PW-0129 through PW-0149, PW-0177 through PW-0180

## Question

After the new executable-code experiments, does any evidence-backed causal path
remain that can produce one complete, correct accepted incremental token per
second on the existing M1 without a hardware sidecar?

## Exact/source path

The preferred one-barrier source-FP8 transaction takes 15.206 ms/layer warm,
or 0.714682 seconds across 47 routed layers. The promoted attention schedule
adds approximately 0.131220 seconds, giving an optimistic resident subtotal of
0.845902 seconds before remaining spine, routing, norms, sampling, cache, and
endpoint work.

That resident premise cannot persist. PW-0108 measures 2.727590 seconds to
acquire one exact 47-layer expert set from the internal SSD. PW-0104's stronger
8 GiB offline-Belady oracle—already physically incompatible with the complete
process footprint—hits only 60.037431% of equal-size expert accesses. Scaling
the measured acquisition by its 39.962569% misses leaves 1.090015 seconds of
I/O. Even granting impossible perfect overlap with all 0.714682 seconds of warm
MoE compute, then adding attention alone, yields at least 1.221235 seconds or
at most 0.818843 TPS. Real causal caching and the rest of the endpoint are
strictly worse. The 4 GiB wide-trace oracle is lower still at 0.610 TPS under
the same impossible grants.

PW-0112 also closes wide exact verification: route union grows to `U=2.401596`
at `q=137`, so perfect acceptance cannot provide the needed acquisition
amortization. The measured source-exact endpoint remains 75.726 seconds/token
and numerically incorrect in its attempted Metal mode.

## Modified representation path

PW-0177 proves compressed resident Core ML arithmetic can be fast, but its
four-effective-bit vector representation misses validation at 15.9577% and
route-time model loading costs 510.365 ms/expert. PW-0178's physically stronger
two-index-bit input code reaches 20.7785% error. PW-0179 shows its residual is
high-rank: rank 96 remains at 19.8209%, while more rank gives back the byte/MAC
advantage. PW-0180's continuous centroid training lowers train loss 55.875%
but worsens frozen validation to 34.0665%. Earlier affine INT4/5/6, GPTQ, AWQ,
outlier, rotation, scalar-codebook, shared-basis, and QAT branches already miss
either fidelity or the physical envelope.

The remaining phrase “broader distillation or a different representation” is
not an actionable hypothesis under the repository's red lines: every cheap
capacity/generalization prerequisite tested here fails, and starting expensive
model-wide training would therefore violate the required experiment order.
Adding enough bits, rank, private centroids, or resident models erases the
traffic advantage. New storage/accelerator hardware would change the premise
but is explicitly excluded from this run.

## Decision

No current configuration reaches 1 accepted TPS, and no further evidence-
backed onboard hypothesis remains after the exact I/O/cache lower bound and the
new representation/training kill tests. Do not claim 1 TPS from warm component
subtotals, ideal overlap, zero accepted-token diagnostics, or theoretical
bandwidth. Reopen only on a genuinely changed premise: new representative
training evidence that passes a cheap held-out gate, an exact executable-byte
reduction not already covered, or user authorization to revisit hardware.

PW-0181 accepts zero tokens, reports no endpoint TPS, changes no measured
throughput-model constant, and records the requested failure honestly.
