# PW-0050 — Slow complete text endpoint

- Status: in progress
- Disposition: unexecuted
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: contract is the first commit containing this record;
  no endpoint execution before that commit
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; model lock
  `df8c74e6f9e1cef154aae5881b9042777653206aaff72855f7b1a1340e0d1050`;
  complete checkpoint-verification manifest
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`
- Hardware, OS, compiler, storage, memory pressure: Apple M1 Mac mini,
  Macmini9,1, 16 GB; macOS 26.6 (25G72); Rust 1.96.0; verified checkpoint
  on internal APFS SSD; remaining fields recorded at execution
- Related records: PW-0001, PW-0029, PW-0030, PW-0039, PW-0049

## Hypothesis and mechanism

A single Rust authority can carry real UTF-8 text through the exact pinned
tokenizer, embedding lookup, all 48 source decoder layers, final RMSNorm,
source LM head, greedy selection, retained per-layer K/V state, and a second
incremental step to observable decoded text while remaining bounded on the
16 GiB host.

The initial walking input is the one-token raw completion `Hello` (token ID
9707). It deliberately minimizes routed union and attention history while
crossing every whole-model boundary. Passing this slice establishes a slow M2
text forward/decode endpoint, not chat-template parity, representative quality,
or usable throughput. The next expansion is the already frozen PW-0001 chat
request, followed by longer teacher-forced traces.

## Shared construction contract

Capability invariant: the unmodified MiMo-V2.5 checkpoint determines every
accepted token. Rust owns tokenizer interpretation, validated checkpoint views,
layer order, attention/KV state, routing, expert scheduling, residuals, logits,
greedy selection, and detokenization. Python may generate independent fixtures
and reports but cannot execute or supply accepted-token state.

Authorized embodiment depth includes Rust, Accelerate, Metal, memory mapping,
and specialized local storage layouts. The first endpoint may use deliberately
slow direct source-FP8/BF16 arithmetic and sequential expert streaming. It may
not expand the whole checkpoint, retain an unbounded expert union, invoke
Python for inference, use hosted inference on the local critical path, or use
modified weights/routing.

The only new runtime representation is a checkpoint-index resolver over the
existing `MappedSafetensors` authority. Tensor name-to-shard ownership remains
the pinned index; dtype, shape, scale layout, file identity, and bounds fail
closed before use. One endpoint command owns the complete causal path; existing
component commands remain tests and controls rather than parallel endpoint
authorities.

## Predeclared gates

Pass only if:

1. the complete checkpoint verification remains valid and the endpoint binds
   the exact config, index, tokenizer, tokenizer config, and revision hashes in
   `evals/fixtures/real/pw0050-text-endpoint.json`;
2. native tokenization of UTF-8 `Hello` without added special tokens is exactly
   `[9707]`, decoding round-trips, and negative tests reject changed tokenizer
   hash, unknown tokenizer schema, invalid UTF-8, and out-of-range token IDs;
3. the endpoint executes all 48 layers in order: full attention at layers
   0/5/11/17/23/29/35/41/47 according to the pinned pattern, SWA elsewhere,
   dense SwiGLU at layer 0, and native noaux-tc top-8 source-expert routing at
   layers 1--47. Routes must derive from the current local layer state;
4. Q/K/V use each layer's published head counts, dimensions, partial RoPE,
   theta, value scaling, causal/full or 128-token SWA history, and learned sink
   policy. The second token must consume retained K/V state rather than rerun a
   one-token stateless approximation;
5. final RMSNorm and all 152,576 source LM-head rows determine greedy token
   choice. The command emits token IDs, decoded bytes/text, top logits, layer
   route traces, cache lengths, tensor/source hashes, and timing/resource
   ledgers in one content-addressed report;
6. readable independent scalar checks cover sampled embedding, QKV, output,
   router, selected expert, final norm, and LM-head values. Each accelerated
   primitive retains its existing production-shape gate; no unexplained NaN,
   shape mismatch, route mismatch, or unstable repeat is accepted;
7. two clean processes produce identical token IDs, route sets, cache lengths,
   and output bytes. At least one negative runtime test rejects a wrong
   revision, missing shard, tensor moved to a wrong shard, wrong shape, bad
   source hash, and corrupted retained cache;
8. report cold and warm process wall, prefill and per-token decode wall,
   accepted tokens, `A=1`, every layer's `U`, logical/actual source bytes,
   resident memory, SSD state, concurrency one, hardware, compiler, commit,
   and output inspection. These are endpoint measurements at batch one but do
   not become a performance default without a later interleaved full-path gain.
9. before the first full walk, enforce shared-host safety at checkpoint-open,
   every completed decoder layer, final head, and token boundaries. Abort
   without writing a passing report if system memory-free pressure falls below
   20%, process physical footprint exceeds 8 GiB, post-relief phase footprint
   exceeds 4 GiB, encrypted swap grows more than 512 MiB from the process
   baseline, any new throttled VM page appears, or a protected resident service
   present at baseline disappears. Protected baseline names are `ChatGPT`,
   `WindowServer`, `nxnode`, and `syncthing`. Call Darwin malloc pressure relief
   after each large phase and record pre/post footprint, relief bytes, memory
   pressure, swap, throttled pages, and service liveness. These are safety stops,
   not benchmark exclusions; an aborted run is preserved as evidence.

The walking slice is killed or split only on a precisely preserved failing
boundary. Inability to fit through sequential source views is a physical
failure; runtime length alone is not. A split cannot call a layer prefix,
frozen-route replay, fixture-supplied activation, or logits-only probe a text
endpoint.

## Baseline and candidate

Baseline authorities are the pinned published model/tokenizer semantics,
source-derived component oracles through PW-0049, and the frozen hosted
behavioral reference for later accumulated comparison. Candidate is one native
process over the verified source checkpoint with bounded sequential expert
materialization and explicit K/V retention.

Raw evidence will be written outside Git under
`/Users/chad/Models/mimo-prismwing/evidence/PW-0050`; only schemas, hashes,
small fixtures, and summarized measurements enter Git.

The initial 2026-08-05 preflight observed 85% system memory free by
`memory_pressure -Q`, 2,005.44 MiB of pre-existing encrypted swap use, zero
throttled pages, and the four declared resident service classes. Swap is
therefore governed by growth from the run baseline rather than an absolute-zero
rule. No full walk was started before adding this gate.

## Attempt history

Run 001 used committed implementation `4f92aac` and failed closed before the
first projection with `FP8 GEMV scale grid mismatch`. The exact boundary is
layer 0 full-attention fused QKV: source weight `[13568,4096]` has a
`[108,32]` scale grid rather than the generic `[106,32]` grid. A complete index
audit found the same layout on all nine full-attention QKV tensors and no other
language FP8 tensor.

The extra two scale rows are structural authority, not ignorable trailing
padding: Q owns 96 ordinary block rows; four 192-wide K heads each own two
scale rows (8); and four 128-wide V heads own one each (4). The repair names and
tests that 108-row fused layout while leaving the generic FP8 validator strict.
The failed-attempt manifest hashes to
`07bbce444da6e3f1beda3ae9e9040884ad1dd7f10dca62791db0331da6b90a10`.
No passing endpoint report exists from run 001.

Run 002 used committed scale-layout repair `1035d69` and crossed the run-001
boundary, then the shared-host safety gate stopped it after layer 24 with
`process footprint limit exceeded`. An independent sample at 34 seconds found
6,580,448 KiB RSS, 86% system memory free, unchanged 1,845.44 MiB swap, zero
throttled pages, and 31,718 compressor pages. The endpoint had released decoded
matrix allocations after each layer, but clean pages faulted from all 17
long-lived checkpoint mappings remained resident and accumulated across the
walk. No output report or accepted token was produced. The failed-attempt
manifest hashes to
`de27be00788e7b21871c8516be2f04e4c40503b9f8555c448ffea2db27c5163f`.

The safety thresholds remain unchanged. The next candidate explicitly advises
Darwin to discard clean pages from every immutable checkpoint mapping before
each phase-level safety check. Mapping addresses and tensor views remain valid;
later access faults authoritative checkpoint bytes back from the SSD. This is
a resource-lifetime repair to test, not evidence of a passing memory bound.

Run 003 used committed layer-boundary release `e051974` and stopped after layer
23. The authoritative post-cleanup snapshot was only 650,410,688 bytes, proving
that whole-mapping `MADV_DONTNEED` plus malloc relief released retained phase
residency. The same snapshot recorded an 8,655,618,048-byte historical peak,
62.6 MiB above the unchanged 8 GiB limit. An independent 49-second sample saw
8,403,520 KiB RSS, 86% system memory free, unchanged 1,821.44 MiB swap, and zero
throttled pages. The failed-attempt manifest hashes to
`4108ea39ab2b7ff9cd1efed914f48073c456050922abddcb67aabc0ae75e8968`.

This reverses only the run-002 diagnosis that layer-boundary page release might
be sufficient. It bounds post-phase residency but does not bound the transient
overlap of decoded matrices, allocator slack, and newly faulted source pages
within a layer. The next candidate drops and relieves those resources after
each complete matrix operation. The safety gate and endpoint semantics remain
unchanged.

## Isolated attribution

Unexecuted. Initial diagnostics will separate tokenizer, embedding, attention,
dense/routed MLP, final norm/head, and storage time without treating any of
them as endpoint TPS.

## End-to-end result

Unexecuted. No endpoint or TPS claim exists.

## Correctness result

Unexecuted.

## Decision

Unexecuted. Passing promotes only the slow target-faithful raw-text forward and
incremental decode endpoint. Chat-template, hosted accumulated parity, native
modalities, MTP speculation, and Prismwing throughput remain subsequent gates.
