# PW-0205 — SGLang block-scaled FP8 reduction

- Status: complete
- Disposition: correctness-repair
- Date: 2026-08-10
- Execution mode: SGLang-directed modified arithmetic candidate
- Hardware/runtime: 16 GiB Apple M1, Metal, internal SSD checkpoint
- Related records: PW-0053, PW-0091, PW-0197 through PW-0204

## Hypothesis

PW-0204 proves a real repeated endpoint but rejects its output as incoherent.
The accelerated path quantizes activations to FP8, immediately dequantizes
them, applies the weight scale to every product, and reduces across all input
blocks with one Metal tree. The pinned SGLang fallback instead retains FP8
activation codes and per-token-group scales, computes a dot for each
128-column block, multiplies that block result by the activation and weight
scales, and accumulates scaled block results in FP32.

Although the real deployment may select DeepGEMM rather than Triton, this
block-scaled equation is explicit in the officially recommended runtime and is
a stronger behavioral candidate than the Hugging Face/Accelerate arithmetic
that is locally exact but produces unusable text.

## Contract and gates

Add a separate Metal quantization output containing exact E4M3FN codes and one
FP32 scale per token-group. Add a block-scaled projection that reduces each
128-value FP8 dot before applying its two scales, then accumulates blocks in
increasing order. Preserve the old L3 kernels and names as historical controls.

Before any full endpoint run:

1. compare quantized codes and scales with the existing PyTorch byte fixture;
2. compare the complete block-scaled Metal projection against an independent
   scalar implementation of the declared equation on multiple blocks and
   nonuniform scales;
3. run one arbitrary-prompt first-token probe with complete provenance; and
4. continue to 32--64 tokens only if the result clears a predeclared
   behavioral gate and does not violate cache, safety, or verifier authority.

This experiment may change the explicitly named modified arithmetic mode. It
may not be labeled target-faithful, tuned to a desired token, or used to weaken
the frozen source and hosted comparisons. A failed first-token probe kills the
candidate without another terabyte-scale generation walk. No isolated kernel
timing is accepted TPS.

## Partial result

The first implementation preserves exact activation codes and scales and adds
the declared 128-column block-scaled projection. Its deterministic Metal probe
matches the existing CPU/PyTorch quantization authority byte for byte and
matches an independent two-block scalar projection within `0.01` maximum
absolute error. The complete Rust library suite passes 87 tests.

Run 001 then changed only ordinary FP8 spine projections while retaining the
historical QKV and routed-MoE reductions as explicit controls. It completed the
41-token arbitrary prompt in six chunks and chose token 13 (`.`), rather than
PW-0204's token 264 (` a`). The report hashes to
`db08d69f8e471128f8fdf5981fb8558235f40c6dc7a28b2d2fae291ab53cda66`
and its progress log to
`0b9c090e8342b5d3ab73d6d13389b1a90fef56f16bfc92de4cc574ddb224560c`.
Complete wall was 155,572.482 ms, including 154,832.475 ms prefill; logical
source bytes were 190,814,088,448, process disk reads 191,044,632,576, and peak
RSS 3,922,575,360 bytes.

That lone punctuation token does not clear a behavioral gate and cannot
justify generation. It does prove the arithmetic distinction is materially
causal at whole-model output. Continue the same source-directed equation
through routed experts, then repeat the one-token gate. Do not call run 001 a
coherence result or accepted TPS.

Run 003 extended block scaling through routed gate/up/down projections but
retained the old QKV row interpretation. It again emitted token 13 (`.`), in
206,429.940 ms complete wall. Its report and progress hashes are
`4858e5c622871b52147f01a655dde2cac0b173e50881de275e4cddd6f846b399`
and `9182c86cd62814930cba12014ef95ea831d836cc5007a1c37355a2fff4a79937`.
Run 004 applied block scaling to QKV as well and still emitted `.`; its report
and progress hashes are
`92e724a9213a462974057b519db958047adf22717ec6b6b8ce8af793a625c36e`
and `38ac28dbdaae90644e65222b00a09d0cb38ddcc50835c31eaf2af2a3e2acb806`.
Thus block-scaled association alone is rejected as the primary behavioral
repair.

The pinned SGLang loader then exposed a prior shared-oracle error. The
checkpoint declares `attention_projection_layout: fused_qkv` and four KV
heads. Its raw attention tensor concatenates four TP shards, each containing
its local `[Q,K,V]` rows. Both Prismwing and the independent Python fixture had
instead read raw rows as global `[all Q,all K,all V]`; the special 108-row
scale audit proved only that every raw scale was consumed, not that it was
attached to the correct semantic output row. The same error affected
sliding-window QKV, where shard boundaries happen to be 128-row aligned and
therefore looked like an ordinary scale grid.

Run 005 deinterleaves both global and sliding-window raw TP rows into logical
global Q/K/V outputs before attention. The first arbitrary-prompt token changes
from punctuation to token 30092, decoded as `Sun`, a plausible beginning for
the sky-color prompt. Its report hashes to
`01bedf3b1028b7b66ad92ab9c0662f62507c4734c8d8f8a06b147ea30785b63b`
and progress log to
`336301452aea0e2e301b2a31f208384e5c8ae4bbd9e0015f0103c0a370d27879`.
Complete wall is 209,150.826 ms, including 208,407.080 ms prefill; logical
source bytes are 272,925,048,064, process disk reads 273,246,760,960, and peak
RSS 3,840,360,448 bytes.

One plausible token is a promotion signal, not coherence evidence. Authorize
one bounded eight-token proposal/verification run next. A grammatical phrase
is required before the 32-token milestone run.

Run 006 passes that bounded phrase gate. Its eight verifier-committed tokens
decode as `Sunlight contains all colors of the spectrum`, a grammatical and
directly relevant continuation. The sole width-eight proposal converged:
posterior tokens exactly match all seven proposed suffix tokens and supply the
next punctuation posterior. Proposal wall is 159,482.884 ms, verification wall
36,814.417 ms, prefill wall 209,692.948 ms, and complete wall 406,624.414 ms.
The run records 444,494,987,520 logical source bytes, 444,997,193,728 process
disk bytes read, and 3,935,567,872 bytes peak RSS. Its report and progress-log
hashes are
`564cd967959ab3c715fb773d01439da1aecd7c2ac3de09550720f118b95f83e3`
and `17e139b9243f90a20eaf715975773618dca0eebfd41df15e80b0489dc10b0b58`.

Promote the corrected QKV mapping and block-scaled arithmetic to the 32-token
milestone run. This is a correctness promotion only; the measured rate is not
accepted endpoint TPS until the complete required output and audit pass.

Run 007 produced 32 fluent, relevant tokens, beginning `Sunlight contains all
colors of the spectrum` and continuing with a correct explanation of shorter
blue wavelengths scattering from gas molecules. Its report and progress hashes
are `638f5b4c315680a26480458110c354da976fc9d8fe62c9aacc3d7c731478992c`
and `52ca1dd369c6bd9c6ac8a8fb8371da311d6a6de2aa5e3e246e7597884cb2117e`.
Complete wall was 1,200,915.190 ms: 207,226.403 ms prefill, 806,352.619 ms
proposal, and 186,531.425 ms verification. It recorded 1,130,120,274,176
logical source bytes, 1,131,371,110,400 process disk bytes read, and
3,951,165,440 bytes peak RSS without swap growth, new throttling, or protected
service loss.

The post-run cache audit rejects run 007 as final transaction evidence despite
its coherent text. A converged width-eight block emits seven suffix tokens; its
last proposal token is the next unevaluated anchor. The implementation retained
all eight proposal rows, then evaluated that anchor again in the next block,
duplicating one hidden-history token. Proposer and verifier shared the error,
so convergence could not detect it. Correct converged retention to seven rows,
bind `next_anchor_token_id` to the last emitted proposal token, distinguish
verifier-authorized from actually observable final-window tokens in schema 2,
and repeat the milestone. Preserve run 007 as evidence that QKV repair restores
language, not as target-faithful repeated-cache evidence.

Run 008 repeats the 32-token milestone with corrected converged-cache
retention and schema 2 accounting. Its output is fluent and relevant:
`Sunlight contains all colors of the spectrum, and as it enters Earth's
atmosphere, shorter blue wavelengths are scattered more than other colors by
gas molecules. This scattered`. All five transactions preserve the distinction
between verifier-authorized rows, observable output tokens, and retained cache
rows; the final cache length exactly matches the prompt plus 31 evaluated
output positions. The report and progress-log hashes are
`ccafb4374e98626cae5027f95b517d0a5b6e59f2747dba0ce7bdd81fd9dc3ff9`
and `b848a930d0678d75c96383de5950102b503cf165f9dd9b9699c4b585afe3654a`.
Complete wall was 1,187,723.583 ms: 205,294.176 ms prefill, 798,997.396 ms
proposal, and 182,642.145 ms verification. It recorded
1,122,669,371,648 logical source bytes, 1,123,903,193,088 process disk bytes
read, and 3,951,296,512 bytes peak RSS without swap growth, new throttling, or
protected service loss.

Run 008 accepts the repaired cache semantics, but its fixed 32-token cap cuts
the second sentence after `This scattered`. It is therefore not the final
publication-quality response. Treat 32 as the minimum endpoint output and 64
as its caller-selected maximum, stopping after the second completed sentence
once the minimum has been reached. Repeat with a 64-token maximum; require the
actual committed count to remain within 32--64 and record the boundary reason.

## Accepted result

Run 009 executes clean commit
`9fc6e3cd8040c7fdcf8a391b39b89d54ded97103` with a 64-token maximum and stops
at the predeclared second-sentence boundary after 47 verifier-committed tokens:

> Sunlight contains all colors of the spectrum, and as it enters Earth's
> atmosphere, shorter blue wavelengths are scattered more than other colors by
> gas molecules. This scattered blue light reaches our eyes from all
> directions, making the sky appear blue.

The report and progress SHA-256 values are
`c87f2a12809c1accc52fc5d5092765ad4cb90cb9d1fa0a2f916a2ccb6d23e1b9` and
`9a51a914eff401050f24310c743af6443d32bea4916a3a958b4b016cb1f8dadb`.
Complete wall is 1,790,267.803 ms: 422.235 ms preprocessing, 206,814.484 ms
prefill, 1,287,279.699 ms proposal, and 295,265.684 ms verification. The
modified-mode complete-path rate is 0.026253 committed tokens/s. Logical source
bytes are 1,633,855,114,496, process disk reads are 1,635,650,719,744, and peak
RSS is 3,959,439,360 bytes. Batch and concurrency are one, verifier width is
eight, swap growth and new throttled pages are zero, and all protected services
survive.

Per-transaction `A` is `[7, 3, 7, 7, 7, 7, 7, 1]`; corresponding `U` is
`[4.582447, 4.539894, 4.117021, 4.127660, 4.688830, 4.255319, 4.704787,
4.414894]`. The last verifier authorized five tokens, but the response boundary
made only one observable and retained; schema 2 records both facts and the
audit reconstructs the final token stream exactly.

Accept PW-0205 as a correctness-repair milestone and a reproducible
arbitrary-text endpoint in its named modified mode. Do not promote its rate to
target-faithful TPS or infer hosted parity, multimodal completion, or the 50 TPS
target from this result.
