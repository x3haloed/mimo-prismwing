# PW-0102 — Pinned-base DFlash acceptance trace

- Status: planned
- Disposition: unexecuted
- Date: 2026-08-06
- Owner: Codex with project owner authorization
- Commit and dirty state: to be recorded before execution
- Checkpoint/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; checkpoint verification
  `9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03`;
  DFlash revision `1f58446181abcaa01030fdbde835fbd38ae9a2b1`; PW-0091
  and PW-0095 evidence hashes to be recorded by the executable manifest
- Hardware/runtime: Apple M1 shared 16 GiB host; verified SSD base checkpoint;
  pinned DFlash draft on `/Volumes/Elements`
- Related records: PW-0009 through PW-0012, PW-0016, PW-0017, PW-0044,
  PW-0091, PW-0095, PW-0096, PW-0100, PW-0101

## Hypothesis and causal mechanism

PW-0100 measures 9,464,659,968 selected source-expert bytes for one ordinary
incremental token and shows that faster one-row arithmetic alone does not avoid
installing those bytes. The pinned DFlash draft may make one exact target
verification walk commit multiple greedy target tokens while reusing a
route-coherent union of experts. The physical quantity to test is therefore
accepted tokens per one-token-equivalent expert union, `A/U`, not draft
accuracy or proposal count alone.

Execute the published five-layer DFlash draft against the pinned base target,
not the incompatible target weights bundled with DFlash. First consume the
frozen PW-0091 target hidden states at layers 0, 11, 23, 35, and 47 and its
greedy first token, 264, to produce the remaining seven positions of one
eight-token proposal. Only after that draft-side path passes its artifact,
semantic, and safety gates may one source-authoritative base-target walk verify
the block and measure acceptance and route union.

## Exactness and red-line check

This is target-faithful greedy L2 speculation. Draft arithmetic may differ from
the target because no draft token is committed without exact base-target
verification. At temperature zero, advance over token 264 and then only the
longest prefix for which each draft proposal equals the base target's greedy
token at the same position; preserve the target's first mismatch token as the
next-block anchor and discard the remaining draft suffix. Do not substitute the
DFlash-bundled target,
change target weights, routing, expert count, chat template, prompt, or
acceptance thresholds. Positive-temperature sampling is out of scope because
the published verifier has no proven speculative-sampling correction.

## Phase contract

### Phase A — Artifact and semantic audit

Fail closed unless the DFlash revision, config, source, mask embedding, tensor
index, draft payload size, and complete-file SHA-256 match a new immutable
manifest. Inventory every draft tensor name, dtype, shape, and shard mapping.
Derive the five-layer forward semantics from the pinned published source and
add deterministic tests for target-hidden selection, concatenation/projection,
mask placement, positions, causal masking, cache growth, greedy proposal order,
and malformed-artifact rejection. No full target walk is authorized in this
phase.

### Phase B — Frozen-hidden draft proposal

Load only the DFlash draft, the minimum verified base embedding/LM-head tensor
views, and the frozen PW-0091 captures. Generate one seven-token greedy draft
suffix from target token 264. Record every proposed token, draft logit hash,
per-layer state/cache shape, load and forward wall time, physical bytes read,
and all safety snapshots. Compare the implementation with the pinned published
source on tiny deterministic fixtures before treating the proposal as valid.
Draft latency is diagnostic and is not accepted TPS.

Release all draft model buffers, mapped tensor views, logits, and caches after
writing the immutable draft manifest. The 4 GiB post-release gate must pass
before Phase C is authorized.

### Phase C — One source-authoritative target verification walk

Run batch one, concurrency one, from the frozen 27-token prompt. Reproduce the
PW-0091 first-token distribution/token before verifying the proposed block.
Verify all eight target positions in one width-eight target pass using pinned
source semantics, preserving per-position target tokens, logits or sufficient
hashed captures, K/V changes, and all 47 routed-layer expert selections.
Compute:

- `q = 8` proposed/verified target positions;
- accepted length `A = acceptance_length + 1`, matching the published loop's
  start-position advance: the initial target anchor plus the matching draft
  prefix. Record the first target correction separately; it is the next-block
  anchor and is not counted in this block's `A`;
- each routed layer's absolute unique-expert count across `q` positions;
- each layer's normalized union, unique experts divided by eight;
- `U`, the mean normalized union across the 47 routed layers, and `A/U`;
- logical and physical draft, dense, attention, expert, K/V, and synchronization
  bytes and complete post-prefill wall time.

Fail closed on any PW-0091 first-token mismatch, source-layout or tensor
mismatch, non-finite value, cache/position mismatch, invalid greedy correction,
missing route, evidence-write failure, or safety stop. Preserve a stopped run
as failed evidence and do not silently restart it. One complete target walk is
authorized; a second requires a separately documented reason arising from the
first result.

## Shared-host safety gate

`TARGET.md` Gate 8 is normative and is an active stop mechanism, not passive
telemetry. Do not begin Phase C while another model walk, checkpoint copy,
checkpoint verification, download, or synchronization transfer is active.
Capture a baseline immediately before each executable phase. Check before and
after checkpoint/model open, every draft or target decoder layer, LM-head and
sampling work, every evidence capture, every declared buffer release, and final
manifest publication. An inability to read a required counter is itself a
fail-closed stop.

At each boundary record current process resident/physical footprint, process
peak RSS, system-free percentage from `memory_pressure`, absolute and baseline
swap use, absolute and new throttled pages, allocator-relief bytes, release
actions, and PIDs for ChatGPT, WindowServer, `nxnode`, and Syncthing when they
were resident at baseline. Stop and preserve evidence if:

- system free memory is below 20%;
- current process footprint or peak RSS exceeds 8 GiB;
- swap growth exceeds 512 MiB or any new throttled page appears;
- a protected baseline service disappears; or
- after a declared release, explicit reference deletion, garbage collection,
  mapped-buffer closure, and Darwin allocator pressure relief do not reduce
  current footprint to at most 4 GiB.

Peak RSS is checked immediately after each expensive operation so a transient
overshoot cannot pass merely because buffers were later freed. Release evidence
must name the buffers/views closed and record footprint before release,
allocator relief, and footprint afterward. The inference process must remain
responsive through a final health checkpoint. Recording a violation without
aborting does not satisfy this contract.

## Promotion and kill thresholds

The draft-side path proceeds to Phase C only if its identity, semantics,
determinism, and safety gates pass. After target verification:

- `A/U > 1` is the minimum evidence that this block provides routed-expert-byte
  leverage; otherwise reject DFlash for this trace except as possible dense
  reuse.
- Compare measured `A/U` with PW-0011's otherwise-free INT4 requirement
  `7.548793`. Falling below it kills this DFlash-8 trace as a Prismwing-50
  mechanism at the current measured kernel bandwidth; meeting it does not
  promote an endpoint because draft, dense, attention, K/V, and orchestration
  costs remain.
- Report the corresponding measured lower-milestone ceiling separately. A
  result around PW-0010's idealized 34.274545 TPS ceiling is valuable but does
  not change the formal 50-TPS completion gate.

No default changes in this experiment. A favorable single trace authorizes a
multi-prompt acceptance/union distribution study followed by a complete
interleaved endpoint measurement; it is not itself a throughput claim.

## Result

Not executed.

## Decision

Await the pinned draft payload, artifact verification, and Phase A/B gates.
