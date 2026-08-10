# PW-0154 — Prompt-calibrated P100 HBM expert cache

- Status: planned
- Disposition: pending
- Date: 2026-08-09
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0112 route trace
  `584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e`;
  PW-0151 analysis
  `d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1`;
  PW-0153 census authority
  `8ac4a179c7b0a06baee05e380dc76acd0a1a64cff4d3e2abe9572ce59afb5c52`
- Hardware candidate: PW-0151's two P100 PCIe cards, 32 decimal GB aggregate
  HBM, bounded three-layer arenas, and striped NVMe misses; analytical
  pre-purchase bound only
- Related records: PW-0104, PW-0112, PW-0128, PW-0151 through PW-0153; E2,
  E6, E7

## Question and changed premise

PW-0153 rejects purchasing enough DRAM for complete source residency but shows
that resident experts sharply reduce the wide-verifier requirement. Test a
smaller exact embodiment that spends the two candidate P100s' HBM deliberately:
after the complete 87-position prompt has run, retain the most frequently used
layer-local source experts in otherwise available HBM, then verify the frozen
137-position suffix while streaming only union misses.

The prompt must precede the suffix. Do not learn residents from the measured
suffix, from future target routes, or from an offline replacement oracle. This
is a causal static-frequency diagnostic whose selected bytes remain exact.

## Exactness and red-line check

This is target-faithful L1 cache and capacity analysis. It changes weight
placement only; source bytes, target routes, arithmetic, tokenizer, modalities,
and acceptance thresholds remain unchanged. Aggregate HBM arithmetic is a
necessary bound, not proof that per-card sharding, peer traffic, kernels, or
the complete endpoint work.

## Contract

1. Authenticate the raw PW-0112 route trace, PW-0151 report, pinned config, and
   complete tensor census by SHA-256. Fail closed on revision, route shape,
   position count, top-k, expert bytes, accepted-token claims, config fields,
   or PW-0151's `q=137` bytes and compute time.
2. Derive the 8K BF16 KV requirement exactly from the pinned hybrid-attention
   pattern: full-attention layers retain all 8,000 positions; sliding-window
   layers retain at most 128. Count key and value heads/dimensions separately.
3. Reserve, before experts, every non-routed source tensor byte, PW-0128's
   exact maximum three-arena bytes, and the complete 8K BF16 KV bytes from two
   P100s' 32 decimal GB aggregate HBM. Convert the remainder to an integer
   count of complete 25,171,968-byte expert records. Do not fractionally cache
   an expert or treat two device memories as proven fungible.
4. Rank layer-local experts only by frequency in prompt positions `0..86`,
   breaking ties by `(layer, expert)`. Freeze the top capacity records. Apply
   that set without admission or replacement to the teacher-forced suffix
   positions `87..223`.
5. Report suffix access hits, union hits, exact residual union bytes, per-card
   contiguous-layer cache distribution, and comparison with the no-cache
   `q=137` union. Logical hits are not measured avoided I/O.
6. Add PW-0151's two-P100 direct-FP32 block compute floor to residual expert
   acquisition at one through four independent PCIe-3-x4 storage lanes, at
   2.5 and 3.5 GB/s per lane. Report perfect-acceptance ceilings and minimum
   integer `A` for 34.3 and 50 TPS. Do not grant overlap.
7. Kill one-lane Prismwing 50 if perfect `A=137` cannot reach 50 TPS. Retain
   only lane counts whose impossible ceiling passes, and carry their required
   acceptance into a separate proposer prerequisite. Do not build CUDA or buy
   hardware from a nameplate pass.
8. Preserve missing per-card balance, tensor-parallel communication, FP8
   decode, cache-install traffic, NVMe sustained reads, prefill storage,
   1M-context KV, power, cooling, cable, and complete-BOM evidence. Apply Gate
   8 and report zero accepted tokens and zero endpoint TPS.

## Promotion and kill rule

This experiment cannot promote a runtime. Kill HBM caching as a primary
mechanism if the causal prompt-trained set does not avoid at least 30% of the
suffix union bytes or if four granted storage lanes still require `A>137` at
either target. A pass retains only the combined exact-cache/striped-storage
architecture and its newly derived proposer prerequisite. Runtime work still
requires a dated complete BOM and a production-shaped CUDA capacity and
transfer benchmark.

## Result

Pending execution.

## Decision

Pending execution.
