# PW-0161 — 32-GB Volta complete-system envelope

- Status: ready
- Disposition: unexecuted
- Date: 2026-08-10
- Owner: Codex with project owner authorization
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; config
  `292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587`;
  PW-0127 arithmetic, PW-0151 CPU ceiling, PW-0155 owned-host, PW-0158
  attention, and PW-0159 complete-envelope authorities to be authenticated by
  SHA-256 at execution;
  official NVIDIA captures and dated market transcription frozen after this
  contract
- Hardware candidates: one Tesla V100 PCIe 32 GB or one Tesla V100S PCIe
  32 GB in the owned H11SSL-i/EPYC host; analytical pre-purchase envelope only
- Related records: PW-0020, PW-0029, PW-0127, PW-0151, PW-0155, PW-0158,
  PW-0159; E7
- Implementation commit and dirty state: pending

## Question and changed premise

PW-0151 tested one V100 only at direct FP32 for the 8K slice. PW-0158 then
showed that two P100s fail ordinary dense one-million-token attention, while
PW-0159 showed that complete one-million-token arithmetic, not attention
alone, is the relevant accelerator floor. The newly authenticated PSU label
makes a single 250-W passive card materially cleaner than two P100s, and a
32-GB Volta card could hold exact 1M BF16 KV plus useful working memory.

Close both actual 32-GB PCIe Volta forms before any purchase request. The
ordinary V100 advertises 112 TFLOPS of favorable deep-learning arithmetic;
V100S advertises 130 TFLOPS and 1,134 GB/s HBM bandwidth. Determine whether
either clears the complete arithmetic gate, whether exact source residency
fits, and whether an active card can exist inside the complete `$500` ledger.

## Shared construction and compression-depth contract

Capability invariant: preserve all one million positions, the pinned nine
global and 39 sliding attention layers, all source heads and dimensions, every
source expert, native modality paths, and every TARGET gate. Ordinary dense
attention remains the control; no position sparsity, summarization, or changed
attention is granted.

Authorized embodiment boundary: report direct FP32 separately. Grant each
card its advertised Tensor/deep-learning peak only as an L3 FP16 numerical
ceiling because Volta Tensor execution does not establish source-BF16
reduction identity. Perfect utilization and free non-arithmetic work make the
screen maximally favorable. Exact source-FP8 experts and exact BF16 KV remain
the residency control; any compressed KV mode is separately named L3.

Project constraints: the M1 remains the user-facing host; the owned EPYC host
costs zero; every newly acquired accelerator, storage device, adapter, cable,
and cooling part must fit `$500` delivered including tax and shipping;
complete measured wall power must remain at most 1,000 W and the installed
NEX750B's tighter limits remain binding.

## Contract

1. Authenticate TARGET, the pinned config, PW-0127's mandatory
   `14,820,573,184` MAC/token ledger, PW-0151's impossible EPYC CPU ceiling,
   PW-0155's owned-host/PSU manifest, PW-0158's exact ordinary-attention
   ledger, PW-0159's complete arithmetic, the official NVIDIA V100 page and
   PCIe product brief, and the dated market transcription by SHA-256. Fail
   closed on revision, context, attention schedule, arithmetic, HBM, card
   form, price, power, or evidence-class drift.
2. Freeze exactly one million input positions. Add two operations per
   PW-0127 MAC at every position to PW-0158's exact ordinary attention work.
   For each card, report floors at official FP32 and advertised deep-learning
   rates after also granting PW-0151's impossible EPYC peak concurrently.
   Grant uninterrupted peak arithmetic, perfect scaling and Tensor occupancy,
   and zero time for softmax, RoPE, routing, storage, cache traffic, dispatch,
   protocol, and all other work.
3. Kill an ordinary-dense candidate if even its favorable L3 Tensor floor
   exceeds TARGET's complete 1,800-second 1M TTFT gate. A passing Tensor floor
   retains only an unqualified numerical branch; it is not source-faithful or
   achieved performance.
4. Recompute exact BF16 1M KV, PW-0159's three maximum routed-layer arenas,
   and PW-0154's non-routed source tensors against 32 decimal GB. Report both
   full source-resident fit and the optimistic alternative that streams all
   non-routed tensors for free. Convert the latter remainder into complete
   `25,171,968`-byte expert slots. Do not use aggregate free bytes as proof of
   an executable per-buffer layout.
5. Freeze one active, used, explicitly 32-GB PCIe listing for each form.
   Reject a captured procurement branch when the card alone exceeds `$500`
   before tax, because adding mandatory cable, forced-air cooling, and any
   storage cannot restore the ledger. Moving listings are price observations,
   not identity, health, delivery, or purchase authority.
6. Bind the 250-W passive dual-slot form, CPU 8-pin auxiliary connector,
   NVIDIA `030-0571-000` dongle, owned x16 slot, and NEX750B rail limits. One
   card plus the 170-W CPU leaves 312 W under the PSU's combined 732-W +12-V
   label, but no installation is authorized without original-compatible
   cabling, physical clearance, forced airflow, staged temperature/ECC/power
   checks, and a complete delivered BOM.
7. Apply Gate 8 to the analyzer. Record zero accepted tokens and no endpoint
   TPS. Never present Tensor, HBM, PCIe, price, or power nameplates as measured
   runtime evidence.

## Promotion and kill rule

Reject the ordinary V100 PCIe 32-GB ordinary-dense 1M embodiment permanently
if its favorable 112-TFLOPS arithmetic floor exceeds 1,800 seconds; price,
storage, kernels, or cooling cannot repair a lower bound that grants those
costs as zero.

For V100S, reject the captured procurement branch if its active card alone
exceeds `$500`. Retain it only as a price-triggered L3 hypothesis if arithmetic
passes: reopening requires an active functional card cheap enough for the
complete delivered cable/cooling/storage ledger, exact installation evidence,
a frozen CUDA implementation, and every accumulated local, hosted,
capability, modality, and long-context fidelity gate.

Neither an arithmetic pass nor an under-cap card would promote an endpoint.
Only measured full-path performance and fidelity may do so.

## Result

Pending source authentication, implementation, execution, and documentation
from a clean commit.
