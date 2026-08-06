# PW-0056 — Real layer-0 BF16 localization

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-05
- Owner: Codex with project owner authorization
- Commit and dirty state: clean `6d2b362dc477433b26c54f6839f60084561118ee`
- Checkpoint/processor/reference hashes: MiMo revision
  `63651580ca774f8504f676040460aed3e1244ac1`; chat fixture
  `56dac58d602ab7bd567e9875282bf2f13ad2c4338e23f65b82affc8ec8bec9a1`;
  pinned model source
  `a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad`
- Hardware, OS, compiler, storage, memory pressure: M1 shared 16 GiB host,
  verified SSD checkpoint, PW-0050 safety contract, PyTorch 2.13.0 CPU oracle
- Related records: PW-0049, PW-0052, PW-0053, PW-0054, PW-0055

## Hypothesis and mechanism

The hosted mismatch persists because at least one real accumulated model
semantic differs below the whole-model level. Output-only walks cannot identify
which boundary first diverges. Dense layer 0 avoids routed-expert selection and
therefore isolates embeddings, RMSNorm, full fused QKV scale layout, dynamic
FP8 activations, partial RoPE, causal GQA attention, BF16 output projection,
both residuals, and the dense SwiGLU block.

## Contract

Use the exact frozen 27-token chat prefix and verified source checkpoint. Build
two independent commands:

1. a PyTorch-readable oracle implementing the pinned layer-0 equations with
   dynamic per-token/per-128 E4M3FN activations and explicit BF16 tensor
   boundaries;
2. a Rust trace using the production endpoint functions, not oracle captures,
   frozen routes, or precomputed downstream inputs.

Both emit hash-bound F32-widened captures for embedding, input RMSNorm, QKV,
post-RoPE Q/K and scaled V, attention output, output projection, first residual,
post-attention RMSNorm, gate/up projections, SwiGLU product, down projection,
and final residual. Preserve dtype, shape, source tensor names/hashes, prompt
IDs, complete checkpoint-verification hash, commands, wall time, logical and
actual bytes, peak residency, and shared-host health.

The fixture must reject wrong revision, prompt, tensor layout, checkpoint
verification, dynamic activation scheme, BF16 policy, and capture schema.
Compare every capture using relative L2, maximum absolute error, BF16 payload
equality rate, and top-difference locations. Exact source bytes and categorical
metadata remain exact gates.

Success localizes the first boundary whose error exceeds the incoming boundary
by more than 10x or whose BF16 equality falls below 99%, with all preceding
boundaries explained. If every boundary remains close (relative L2 at most
`5e-4`, maximum absolute error at most `2e-2`, BF16 equality at least 99%),
layer 0 is provisionally cleared and the same trace moves to the first routed
layer. These diagnostic limits do not weaken hosted acceptance.

Stop on any PW-0050 safety violation. Keep generated arrays outside Git;
commit schemas, generators, manifests, hashes, and small representative data.

## Result

The first oracle run failed closed because it serialized attention as
`[27,8192]` while the contracted semantic shape was `[27,64,128]`; run 001 was
preserved and replaced, not edited. Comparison 002 then established bit-exact
embedding, normalization, dynamic-FP8 QKV, RoPE Q/K, and scaled V. It localized
the first apparent divergence to attention and motivated score/probability
subcaptures.

Comparison 003 showed scaled QK scores at `5.97e-6` relative L2 and 99.988%
BF16 equality, probabilities at `1.98e-4` and 99.797%, and attention output at
`5.41e-4` and 98.793%. A reduction replay proved the probability difference
was the sole cause; probability-times-V reduction itself matched exactly.
Inspection then found that the Python oracle had omitted the pinned BF16
max-subtraction before F32 softmax. PW-0057 corrected that oracle staging and
the production exponential implementation.

Final oracle run 004 and Rust run 004 clear the complete layer. Comparison 005
has no failing boundary: maximum relative L2 is
`2.8474986748078703e-6`, maximum absolute error is `7.62939453125e-6`, and
minimum BF16 equality is `0.9999586640211641`. Attention probabilities and
every downstream capture through the final residual are bit-exact.

Evidence hashes:

- oracle run 004 manifest:
  `6bbc6562f8ac915efc40c81720d076408fa37ae391802e90dca5e54cc8271cd3`;
- Rust run 004 manifest:
  `9e4a95014b8e5cf1954e55fe36bbcb95a338ae4e2366549d940a10d994ff5681`;
- final comparison 005:
  `a741cc0a3686926ff2d4c880b08c3ab4ee046b4912f58a3b9738d2952ebbcb78`.

Rust run 004 completed in 3.046 seconds, moved 324,310,528 logical source
bytes, measured 278,417,408 process disk-read bytes, peaked at 713,244,672
resident bytes, retained 84% system free memory, grew no swap, observed no new
throttled pages, and preserved every protected service.

## Decision

Promote the layer-local trace and comparison machinery as a correctness
diagnostic. Provisionally clear dense layer 0 under the pinned PyTorch
BF16/dynamic-FP8 oracle. Preserve the failed schema and incomplete-oracle runs
as evidence of gates working. Move the trace to routed layer 1, where learned
sink attention and dynamic expert selection introduce the first untested
structural boundary. Do not infer hosted parity or endpoint performance from a
single cleared layer.
