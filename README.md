# MiMo Prismwing

Prismwing is a consumer-hardware runtime research project for the full-capability
open-weight Xiaomi MiMo-V2.5 model. The starting machine is a 16 GB Apple M1 Mac
mini; a tightly bounded local companion appliance is allowed by the target, but
hosted inference is not.

The project is not finished. Its final gate remains a near-equivalent native
multimodal runtime sustaining at least **50 accepted tokens/s** for one
interactive request. The current complete text path is still orders of
magnitude slower, hosted accumulated parity is incomplete, and native image,
audio, video, and mixed-modality delivery are not yet proven.

What has changed is the quality of the evidence: Prismwing now has a real,
bounded, arbitrary-prompt causal path through the pinned checkpoint, an
accelerated all-layer verifier, and a native MiMo MTP proposer whose exact
verifier-authorized gains repeat across categories and an untouched 32-token
holdout.

## Current frontier

All measurements below are batch one, concurrency one on the 16 GB Apple M1.
They are different cuts of the system and must not be compared as if they were
the same endpoint.

| Frontier | Best supported result | Boundary |
| --- | ---: | --- |
| Untouched 32-token native-MTP holdouts | **1.722× ordinary**, **1.557× code** versus q8 | Complete requests including prefill; multilingual and rare-route remain |
| Native-MTP seven-token category panel | **1.864× ordinary**, **1.298× code**, **1.289× multilingual**, **1.174× rare-route** | Complete exact-output requests; conditional, not a general default |
| Accelerated width-eight verifier | **0.219850 accepted TPS warm** | Post-prefill verifier transaction; proposal generation excluded |
| Layer-major routed-MoE prefill slice | **1.161×–1.192×** | One real layer; misses its 3× continuation and absolute numerical gates |
| Modified K4/source routed component | **351.680 ms p90 for 47 repeats** | L3 frozen layer/route/input; passes 2-TPS component condition, fails 3-TPS diagnostic; not endpoint TPS |
| Modified K4/source downstream slice | **same argmax; 0.000493 top-20 JSD** | One frozen layer-28 route through logits; ten later route sets change; zero accepted tokens |
| Earlier arbitrary-prompt endpoint | **0.026253 complete-path TPS**, 47 coherent tokens | SGLang-directed modified arithmetic; retained as a control |

The strongest new result is [PW-0216](experiments/PW-0216-native-mtp-longer-output-holdout.md).
The prompt panel was hash-frozen before execution. On the first 32-token
ordinary holdout, two native-q4 candidates produced byte-identical output to the
q8 control and repeated the same committed-token sequence:

```text
[3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3]
```

Candidate median complete wall time was 695.983 s versus 1,198.644 s for the
control. Post-prefill wall improved 2.264×, logical source traffic fell 34.05%,
and measured process reads fell 37.01%. All runs recorded zero swap growth,
zero newly throttled pages, and no protected-service loss. The code holdout also
passes at 1.557× complete-path TPS despite a harder thirteen-transaction
acceptance sequence and 29.90% fewer process reads. Multilingual and rare-route
32-token holdouts remain frozen and pending.

[PW-0215](experiments/PW-0215-native-mtp-slice-broadening.md) supplies the
shorter category breadth. Every ordinary, code, multilingual, and rare-route
candidate/control/candidate sequence emitted identical verifier-authorized
output and showed a repeatable positive complete-path gain. These smaller gains
matter: Prismwing preserves useful advances even when they cannot by themselves
reach 50 TPS.

## What actually runs

The current text runtime crosses one native authority from UTF-8 input to
observable decoded output:

1. pinned chat serialization and tokenizer;
2. bounded causal prefill through all 48 decoder layers;
3. retained per-layer K/V and target hidden history;
4. an authenticated three-layer native MiMo MTP q4 proposer;
5. source-checkpoint wide verification across attention, routers, all selected
   experts, residuals, final normalization, and LM-head rows;
6. verifier-only commit, rejected-suffix rollback, and continued generation;
7. content-addressed timing, route, byte, residency, and safety evidence.

The accelerated implementation binds page-rounded regions of the original
safetensors shards directly to Metal where possible. Width-specialized FP8 MoE
kernels cover actual expert placement counts; a dedicated QKV path handles
MiMo's nonuniform scale layout. Checkpoint views and large phase resources are
released aggressively enough to keep complete walks safe on the shared 16 GB
host.

The exactness boundary is important. Native MTP is used only as a draft; the
target verifier is the sole authority for committed tokens. However, the
current fast verifier uses a separately named Metal-native numerical mode, and
whole-model hosted-reference parity remains unproven. Prismwing does not call
this target-faithful delivery yet.

## Research that changed direction

Recent work has also closed several tempting paths instead of leaving them as
speculation:

- PW-0209 found and fixed a production route-buffer bug that copied only 24
  bytes of route weights/positions. Its layer-major path retained a real
  16.1%–19.2% slice gain but failed the frozen promotion gate.
- PW-0210 proved packed gate/up-to-SwiGLU fusion byte-exact, then rejected it as
  performance-neutral under the storage-dominated cut.
- PW-0212 rejected corrected-route predictive prefetch because even a bounded
  future oracle could hide only 1.62% of complete wall.
- PW-0213 preserved lower-level uncached-I/O and install gains while rejecting
  runtime promotion on a post-prefill regression.
- PW-0214 preserved category-specific horizon gains but rejected runtime
  adaptation below its frozen 5% oracle ceiling.
- PW-0300 through PW-0304 audited a new weight-representation line. Exact small
  FP8 palettes were ruled out; the tested six-bit subset, independent row-query,
  joint SwiGLU balance, and recursive polar forms all failed their predeclared
  fidelity or equal-traffic gates. Their negative results narrow the next
  representation search rather than weakening acceptance thresholds.
- PW-0308 imported a mixed K4/source Metal component from a stronger M4 worker,
  corrected its identity-preserving label to L3 modified weights, and replayed
  it on the target M1. Forty-seven repeated components take `351.680` ms p90,
  with `341.383` ms on GPU: dispatch is no longer dominant, but even this
  frozen fallback misses the three-TPS component diagnostic.
- PW-0309 authenticates the missing layer-28 residual and propagates the same
  frozen L3 substitution through layers 29–47 and logits. Internal drift is
  large enough to change ten later route sets, but the declared distribution
  slice retains the same argmax, 20/20 top-set overlap, `0.005353`-nat
  source-token error, and `0.000493` projected JSD. This is a conditional
  frozen-route fidelity result, not endpoint performance or general K4 safety.
- PW-0310 replaces fixture-supplied router values with the installed
  checkpoint's live layer-28 route. Expert order matches exactly, maximum
  weight error is `2.98e-8`, and Metal still reproduces every modified
  candidate bit. The route gate is retained as a conditional fixture, while
  full-bank construction is removed from the active 50-TPS frontier because
  the measured routed-component ceiling is only about `2.84` tokens/s.

The append-only reasoning and evidence history lives in
[LEARNINGS.md](LEARNINGS.md) and the [experiment ledger](experiments/README.md).

## Build

The accelerated runtime targets Apple Silicon and Metal. A source build needs
Xcode Command Line Tools, stable Rust with edition 2024 support, and Python 3.11
or newer for evidence utilities.

```sh
git clone https://github.com/x3haloed/mimo-prismwing.git
cd mimo-prismwing
xcode-select --install
rustup toolchain install stable
cargo build --release
cargo test --release
python3 -m unittest discover -s tests
```

Tests do not download model weights. Some Python fixtures require optional
packages or external content-addressed evidence.

## Checkpoint authority

Runtime evidence is pinned to XiaomiMiMo/MiMo-V2.5 revision
`63651580ca774f8504f676040460aed3e1244ac1`. The full installation is roughly
294 GiB on disk and must match [spec/model.lock.json](spec/model.lock.json).

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade huggingface_hub

export PW_CHECKPOINT_ROOT="/absolute/path/to/MiMo-V2.5-63651580"
export PW_EVIDENCE_ROOT="/absolute/path/to/prismwing-evidence"

hf download XiaomiMiMo/MiMo-V2.5 \
  --revision 63651580ca774f8504f676040460aed3e1244ac1 \
  --local-dir "$PW_CHECKPOINT_ROOT"

mkdir -p "$PW_EVIDENCE_ROOT/checkpoint"
python3 tools/checkpoint_lock.py verify \
  --lock spec/model.lock.json \
  --checkpoint-dir "$PW_CHECKPOINT_ROOT" \
  --require-complete \
  --manifest "$PW_EVIDENCE_ROOT/checkpoint/checkpoint-verification.json"
```

Receipts bind the verified installation, not merely filenames. Do not edit
hashes or metadata to reuse an authority after files move or change. Large raw
evidence and weights stay outside Git; the repository commits schemas, hashes,
small fixtures, and summarized results.

The CLI exposes the research entry points. Run `target/release/prismwing`
without arguments for their exact signatures. Reproducing a promoted result
also requires the experiment's frozen authorities and a clean commit matching
the report; start with the reproduction notes in the relevant experiment
record rather than treating a component command as endpoint TPS.

## Mission and definition of done

Prismwing is complete only when every gate in [TARGET.md](TARGET.md) passes from
a clean checkout. In condensed form:

- exact, auditable model/tokenizer/processor and hosted-reference locks;
- native local text, image, multi-image, audio, video, mixed-modality, tool,
  multi-turn, and long-context execution;
- near-equivalent distributions over at least 100,000 scored tokens, plus
  capability non-inferiority;
- median batch-one decode of at least 50 accepted TPS after an 8K prefill,
  with the required tail, latency, power, and sustained-run gates;
- three cold reproductions, a warm run, raw content-addressed evidence, and an
  independent reproduction.

The 100-TPS result is a stretch goal. Proposed tokens, aggregate multi-user TPS,
kernel-only timing, decompression-only timing, or modified-model output do not
satisfy the primary target. See [RED_LINES.md](RED_LINES.md).

## Near-term frontier

The active sequence is deliberately evidence-first:

1. re-evaluate the owned companion-host and sub-`$500` hardware frontier using
   measured full-path traffic/compute bounds; the K4 full-bank branch is no
   longer the next performance move;
2. finish the frozen PW-0216 multilingual and rare-route 32-token holdouts;
3. reduce native proposer embodiment cost without changing its authority;
4. attack the dominant prefill/storage cut while preserving the smaller
   verified gains already found;
5. close accumulated hosted-reference parity and expand the causal path through
   native modalities;
6. continue representation research only through predeclared cheap falsifiers
   and real-query controls.

## Repository map

- [TARGET.md](TARGET.md) — normative completion and stopping conditions.
- [RED_LINES.md](RED_LINES.md) — shortcuts that do not count.
- [LEARNINGS.md](LEARNINGS.md) — durable evidence, reversals, and deductions.
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — experiment and promotion discipline.
- [docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md) — fidelity and
  performance methodology.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — active staged research plan.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — runtime causal and topology
  contract.
- [docs/SOURCES.md](docs/SOURCES.md) — pinned source and decision ledger.
- [spec/throughput-model.json](spec/throughput-model.json) — measured and modeled
  constants with provenance.
- [evals/README.md](evals/README.md) — fixture and evidence layout.

## Terms

- **Accepted TPS:** verifier-committed output tokens divided by the declared
  complete timed interval, including drafting, verification, misses, transfers,
  synchronization, and rollback.
- **Target-faithful:** original weights, routing, model distribution, and named
  source semantics apart from documented finite-precision effects.
- **Modified mode:** any changed weights, routing, topology, expert count, or
  accepted surrogate output; it remains named separately even when useful.
- **Component result:** a kernel, layer, storage, or verifier measurement that
  diagnoses a cut but is not complete endpoint throughput.
