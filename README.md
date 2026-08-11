# MiMo Prismwing

**A consumer-hardware runtime research project for full-capability Xiaomi
MiMo-V2.5.**

`mimo-prismwing` keeps the bird lineage of
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) and
[Swiftlet](https://github.com/leonickson1/Swiftlet). “Prismwing” points at the
reason for targeting MiMo: one language backbone integrates text, images,
audio, and video.

## Milestone: the first complete accelerated wide verifier

Prismwing now runs a complete, eight-position MiMo-V2.5 Jacobi verification
transaction on a **16 GB base Apple M1** at **0.21985 accepted tokens per
second** warm. It uses the unmodified, pinned source checkpoint and the Mac's
internal SSD—no accelerator, server, or new-hardware sidecar.

That is not the project's final 50 TPS target. It is a substantial milestone:
the project has moved from component experiments and slow reference walks to a
real, memory-safe accelerated path through all 48 layers, all 47 routed MoE
layers, retained K/V, final normalization, and all eight required LM-head
rows.

### What the evidence supports

| Measurement | PW-0203 run 004 |
| --- | ---: |
| Warm accepted throughput | **0.219849686 TPS** |
| Cold accepted throughput | **0.218103927 TPS** |
| Accepted tokens / verifier width | **5 / 8** (`A=5`, `q=8`) |
| Mean normalized expert union | **2.085106** |
| Warm complete verifier wall time | **22.742812 s** |
| Warm physical checkpoint reads | **27,508,178,944 bytes** |
| Peak process resident memory | **695,681,024 bytes** |
| Swap growth / newly throttled pages | **0 / 0** |

Both cold and warm trials produced the exact frozen target posterior:

```text
[13, 15, 13, 15, 481, 13, 15, 15]
```

The final path is **3.42×** faster than PW-0203's first complete Metal-MoE
variant (`0.06422` warm TPS). The run report hashes to
`8febe98c77fe779b7ff896205bdcf9086efed5ffc6052ca6ddb173fc5d563b01`;
the full experiment record and the progression through all four runs are in
[PW-0203](experiments/PW-0203-wide-source-jacobi-endpoint.md).

The boundary matters. This is steady-state, post-prefill **wide verification
throughput** for one hash-pinned, target-generated Jacobi block at batch one
and concurrency one. The 34.230-second authenticated K/V hydration is setup
outside the timed interval, and proposal generation is not part of this
transaction. The result is not ordinary autoregressive generation TPS, a
production server claim, source-BLAS component parity, or completion of the
50 TPS target. The weights and routing are source-authority; reduction
arithmetic is the explicitly named Metal-native L3 mode.

## What made it possible

- **Direct checkpoint execution.** Page-rounded regions of the original
  safetensors shards are bound directly to Metal. Prismwing does not build a
  roughly 303 GB repacked expert bank. One QKV tensor that ends at shard EOF
  uses a bounded 60.8 MB copied fallback; page-coverable tensors remain
  no-copy.
- **Width-specialized FP8 MoE kernels.** Compile-time-specialized kernels for
  one through eight rows match each expert's actual placement count, perform
  dynamic group-128 E4M3FN activation quantization, and keep route-weighted
  scatter inside the GPU transaction.
- **A genuinely wide transaction.** Eight proposed positions are verified
  together across 64 routed placements per layer, allowing reused experts to
  do useful matrix-shaped work rather than repeating token-major GEMV.
- **Metal across the language spine.** Direct-checkpoint kernels cover
  attention QKV and output projections, dense layer zero, the routed layers,
  final normalization, and the eight LM-head rows. A dedicated QKV kernel
  handles MiMo's nonuniform source scale layout.
- **Authenticated cache hydration.** Hash-locked PW-0187 source prefill states
  reconstruct real per-layer K/V by replaying checkpoint attention outside
  the timed steady-state interval, avoiding a prohibitively slow CPU expert
  prefill inside the benchmark.
- **Measure model work once.** The complete checkpoint receipt authorizes
  already-verified FP8 content, so repeated per-request payload scans were
  removed. Heavy OS safety probes were moved around—not out of—the complete
  transaction. Layout checks, process-memory limits, free-memory checks, swap
  checks, and protected-service checks still fail closed.

## Build locally

The accelerated path currently targets Apple Silicon and Metal. The reference
environment is macOS on a 16 GB M1 with the internal SSD. A source build needs:

- Xcode Command Line Tools;
- stable Rust with edition 2024 support (Rust 1.85 or newer);
- Python 3.11 or newer for evidence and checkpoint utilities; and
- enough storage for the pinned checkpoint: 315.7 GB of tensor data, plus
  download and evidence headroom.

```sh
git clone https://github.com/x3haloed/mimo-prismwing.git
cd mimo-prismwing
xcode-select --install
rustup toolchain install stable
cargo build --release
cargo test --release
python3 -m unittest discover -s tests
```

The Python suite is a collection of independently runnable research fixtures;
some tests require optional packages or large artifacts. The Rust build and
tests do not download model weights.

## Reproduce the PW-0203 milestone

The benchmark fails closed unless every authority matches. You need:

1. the complete `XiaomiMiMo/MiMo-V2.5` checkpoint at revision
   `63651580ca774f8504f676040460aed3e1244ac1`;
2. a complete checkpoint-verification receipt created by
   `tools/checkpoint_lock.py`;
3. the 32 MB PW-0187 run-001 authority directory, whose `manifest.json` hashes
   to `a1066fafa979b923f9c2f5d259ff85b2f3d5aa2e77400e8b7075a48f3fa67950`;
4. the committed endpoint fixture and Metal kernel; and
5. an otherwise idle Apple Silicon Mac with enough free memory and no active
   memory pressure.

The current Hugging Face CLI can fetch the exact revision into a chosen local
directory ([download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli)):

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade huggingface_hub

export PW_CHECKPOINT_ROOT="/absolute/path/to/MiMo-V2.5-63651580"
export PW_EVIDENCE_ROOT="/absolute/path/to/prismwing-evidence"

hf download XiaomiMiMo/MiMo-V2.5 \
  --revision 63651580ca774f8504f676040460aed3e1244ac1 \
  --local-dir "$PW_CHECKPOINT_ROOT"

mkdir -p "$PW_EVIDENCE_ROOT/PW-0049"
python3 tools/checkpoint_lock.py verify \
  --lock spec/model.lock.json \
  --checkpoint-dir "$PW_CHECKPOINT_ROOT" \
  --require-complete \
  --manifest "$PW_EVIDENCE_ROOT/PW-0049/checkpoint-verification.json"
```

Checkpoint receipts deliberately bind the verified file installation, and
the frozen PW-0187 authority binds the receipt used to create it. Therefore a
freshly downloaded installation is an **independent evidence epoch**, not a
byte-for-byte replay of the original receipt. Regenerate the prerequisite
authority ladder described by [PW-0187](experiments/PW-0187-jacobi-third-iteration.md),
or obtain the original installation-bound receipt and authority bundle. Do not
edit hashes to bypass this gate.

With matching authorities in place, build from a clean commit and run:

```sh
export PW_RUN_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$PW_EVIDENCE_ROOT/PW-0203/reproduction"

target/release/prismwing wide-metal-jacobi-text-endpoint \
  "$PW_CHECKPOINT_ROOT" \
  spec/model.lock.json \
  "$PW_EVIDENCE_ROOT/PW-0049/checkpoint-verification.json" \
  evals/fixtures/real/pw0052-chat-endpoint.json \
  "$PW_EVIDENCE_ROOT/PW-0187/run-001/manifest.json" \
  kernels/block_fp8_gemv.metal \
  "$PW_EVIDENCE_ROOT/PW-0203/reproduction/report.json" \
  "$PW_RUN_COMMIT"

python3 - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(
    Path(os.environ["PW_EVIDENCE_ROOT"], "PW-0203/reproduction/report.json")
    .read_text()
)
for trial in report["trials"]:
    print(trial["cache_state"], trial["accepted_tps"], trial["posterior_token_ids"])
PY
```

For a valid comparison, record the emitted report hash, cold and warm state,
`A`, `U`, bytes moved, wall time, memory/swap observations, hardware, and Git
commit. Do not compare a kernel-only or storage-only microbenchmark with the
complete verifier number.

## Mission

Make the open MiMo-V2.5 checkpoint run locally on a 16 GB M1 Mac mini while
preserving the model's full input modalities and producing demonstrably
near-equivalent behavior to a pinned hosted copy of `xiaomi/mimo-v2.5` on
OpenRouter.

The primary completion target remains **at least 50 accepted output tokens per
second for a single interactive request**. One hundred TPS is a stretch target.
Neither proposed speculative tokens nor aggregate batched throughput count as
interactive output TPS.

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
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — causal, topology, and
  implementation contract.
- [docs/EMBODIMENT_JUMPS.md](docs/EMBODIMENT_JUMPS.md) — predeclared
  architecture-level compression hypotheses and their test order.
- [docs/SOURCES.md](docs/SOURCES.md) — pinned source and decision ledger.
- [experiments/README.md](experiments/README.md) — append-only experiment ledger
  and record template.
- [spec/acceptance.yaml](spec/acceptance.yaml) — machine-readable acceptance
  thresholds.
- [evals/README.md](evals/README.md) — fixture and evidence layout.

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
