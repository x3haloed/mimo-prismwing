# PW-0164 — affordable Blackwell complete-system envelope

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0127 arithmetic, PW-0151 owned host, PW-0158 attention, and PW-0161
  complete-envelope authorities to be authenticated at execution; official
  RTX 5060 family page `238d00c79c20939e5208e1a6507a6949e00e21ab3c3c3cc79d25b97eb0af20fd`;
  official Blackwell architecture whitepaper
  `906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa`;
  official launch release `76ca4fce0315435079d72f3725174b704b9b8990b3be7d89591471333a418394`;
  dated market transcription
  `98400749a4ca60351ff71b0450bf545ee542691051e783426c2c183219774cf6`
- Hardware candidate: one RTX 5060 Ti 16-GB card in the owned H11SSL-i/EPYC
  host; analytical pre-purchase envelope only
- Related records: PW-0127, PW-0151, PW-0158, PW-0159, PW-0161, PW-0163; E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0159 rejects RTX 3080 source BF16 but leaves its faster FP16 form as an L3
arithmetic survivor. PW-0161 and PW-0163 reject older 32-GB datacenter cards
on either source arithmetic or price. Close the strongest NVIDIA Blackwell
tier officially launched below the complete `$500` cap before treating those
older results as representative of modern consumer Tensor hardware.

NVIDIA advertises 759 AI TOPS for RTX 5060 Ti, but that mixed-format marketing
number is not a dense BF16 rate. Derive dense BF16/FP32-accumulate and the more
favorable FP16/FP16-accumulate ceiling from the official same-generation RTX
5070 table using SM count and boost-clock scaling. Ask whether either can meet
the complete one-million-position arithmetic gate even with every omitted cost
set to zero.

## Shared construction and compression-depth contract

Capability invariant: preserve every source weight, all one million positions,
the pinned nine global and 39 sliding attention layers, native modality paths,
and every TARGET gate. Ordinary dense attention remains the control.

Authorized embodiment boundary: use dense BF16 with FP32 accumulation as the
source-oriented ceiling. Treat FP16 with FP16 accumulation as a separately
named L3 diagnostic. Do not grant structured sparsity, FP8, FP4, DLSS AI TOPS,
or changed weights. Grant perfect utilization, concurrent impossible EPYC
peak, and zero time for every non-arithmetic operation.

Project constraints: the M1 remains the user-facing host; the owned EPYC host
costs zero; all new hardware must fit `$500` delivered. Complete measured wall
power remains at most 1,000 W, with the installed NEX750B's tighter 732-W
combined +12-V label binding.

## Contract

1. Authenticate TARGET, config, PW-0127, PW-0151, PW-0158, PW-0161, NVIDIA's
   product page, architecture whitepaper, launch release, and dated market
   transcription by SHA-256. Fail closed on model, arithmetic, host, rate,
   memory, power, price, or evidence drift.
2. Derive 36 SMs from 4,608 CUDA cores and the official 128 cores/SM. Scale the
   official RTX 5070 dense BF16 and FP16 rates by `36/48 * 2570/2512`. Record
   that 759 AI TOPS is not admitted as dense BF16 work.
3. Add two operations per PW-0127 MAC at all one million positions to
   PW-0158's exact attention work. Grant each GPU rate concurrently with the
   EPYC's impossible peak. Permanently reject ordinary-dense RTX 5060 Ti if
   even favorable FP16/FP16 accumulation exceeds 1,800 seconds.
4. Recompute exact BF16 1M KV, three arenas, and non-routed source tensors
   against 16 decimal GB. Report whether KV alone fits without claiming host
   streaming performance.
5. Bind 180-W board power, one 8-pin/Gen-5 connector option, official 600-W
   system recommendation, and the photographed 732-W +12-V host ceiling.
   Nameplate headroom is not cable, fit, thermal, or measured-load proof.
6. Preserve the official `$429` launch price and the dated `$479.99` out-of-
   stock market transcription. Neither proves a complete delivered BOM or
   purchase availability.
7. Apply Gate 8. Record zero accepted tokens and no endpoint TPS. Do not
   present a derived ceiling as achieved CUDA performance.

## Promotion and kill rule

Permanently reject ordinary-dense one-million-position RTX 5060 Ti execution
if even its favorable dense FP16/FP16-accumulate arithmetic floor exceeds
1,800 seconds. This rejection is price-independent and therefore survives a
future bargain card.

Do not generalize the result to RTX 5070 or faster cards, changed-attention
L3/L4 modes, FP8/FP4 modified weights, or distributed hardware. Do not purchase
or implement CUDA from this analytical result.

## Result

Pending clean implementation and execution.
