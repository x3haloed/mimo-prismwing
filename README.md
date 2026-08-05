# MiMo Prismwing

**A consumer-hardware runtime research project for full-capability Xiaomi MiMo-V2.5.**

`mimo-prismwing` keeps the bird lineage of
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) and
[Swiftlet](https://github.com/leonickson1/Swiftlet). “Prismwing” points at the
reason for targeting MiMo: one language backbone integrates text, images,
audio, and video.

## Mission

Make the open MiMo-V2.5 checkpoint run locally on a 16 GB M1 Mac mini, with
inexpensive local hardware assistance permitted, while preserving the model's
full input modalities and producing demonstrably near-equivalent behavior to a
pinned hosted copy of `xiaomi/mimo-v2.5` on OpenRouter.

The primary completion target is **at least 50 accepted output tokens per
second for a single interactive request**. One hundred TPS is a stretch target.
Neither proposed speculative tokens nor aggregate batched throughput count as
interactive output TPS.

This repository begins as a research specification. It deliberately defines
success, evidence, and prohibited shortcuts before selecting an implementation.

## Definition of done

The project is complete only when every required gate in [TARGET.md](TARGET.md)
passes from a clean checkout:

1. The model, tokenizer, preprocessing, hardware, and hosted reference are
   pinned and auditable.
2. Text, image, multi-image, audio, video, and mixed-modality inputs run through
   the native MiMo path locally.
3. Logprob comparisons and capability tests meet the near-equivalence
   thresholds, including separate modality and tail-case checks.
4. Batch-one decode sustains at least 50 accepted TPS on the declared local
   consumer system.
5. The run is reproducible and publishes raw evidence—not only a summary.

See [RED_LINES.md](RED_LINES.md) for shortcuts that do not count.

## Research stance

- Treat storage capacity, storage traffic, executable memory traffic, compute,
  and sequential barriers as separate budgets.
- Measure accepted tokens per byte moved, not advertised device bandwidth.
- Put approximations on the draft side of exact verification when possible.
- Call a modified or distilled model what it is.
- Prefer cheap kill tests before runtime construction or hardware purchases.
- Preserve rare modalities and capabilities, not only average text quality.

## Repository map

- [TARGET.md](TARGET.md) — normative completion criteria.
- [RED_LINES.md](RED_LINES.md) — boundaries the project will not cross.
- [LEARNINGS.md](LEARNINGS.md) — evidence and deductions accumulated so far.
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — reference-first implementation,
  optimization, promotion, reversal, and documentation loops.
- [docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md) — hosted-reference,
  logprob, capability, and performance methodology.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — staged experiments and kill
  criteria.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — initial causal, topology, and
  implementation contract.
- [docs/EMBODIMENT_JUMPS.md](docs/EMBODIMENT_JUMPS.md) — predeclared
  architecture-level compression hypotheses and their test order.
- [docs/SOURCES.md](docs/SOURCES.md) — pinned source and decision ledger.
- [experiments/README.md](experiments/README.md) — append-only experiment ledger
  and record template.
- [spec/acceptance.yaml](spec/acceptance.yaml) — machine-readable acceptance
  thresholds.
- [evals/README.md](evals/README.md) — fixture and evidence layout.

## Status

**Active executable research foundation.** The checkpoint and hosted provider
are pinned; source-derived Rust oracles, real Metal kernels, and a native C++
MLX quantized-matmul smoke path pass from the unified test command. Component
experiments report reproducible projection measurements, but no complete local
decode endpoint or accepted-TPS claim exists yet. The active work is full
checkpoint verification, a real fused expert path, complete text decode,
modalities, hosted parity, and end-to-end optimization.

## Terminology

- **Reference checkpoint:** the exact open-weight revision and tokenizer pinned
  by checksum.
- **Hosted reference:** a frozen OpenRouter response corpus from a pinned
  MiMo-V2.5 endpoint and request configuration.
- **Accepted TPS:** committed output tokens divided by complete decode-loop wall
  time, including drafting, verification, misses, transfers, and rollback.
- **Target-faithful:** original weights, routing, and target distribution apart
  from documented finite-precision effects.
- **Modified MiMo:** any changed weights, routing, topology, expert count, or
  accepted unverified surrogate output.
