# Reproduce the PW-0205 arbitrary-text endpoint

PW-0205 is a real arbitrary-text generation endpoint for the pinned
MiMo-V2.5 checkpoint on a 16 GB Apple M1. It performs chat serialization,
tokenization, six-chunk causal prefill, repeated source-checkpoint proposal,
width-eight accelerated verification, verifier-only commit, and exact cache
rollback. It is **SGLang-directed modified arithmetic**, not target-faithful
arithmetic and not evidence that the 50 TPS or multimodal targets have passed.

The checkpoint-verification receipt is reusable for its exact installation.
Generation validates that receipt but does not hash the checkpoint again.
Creating a receipt for a new installation is a separate one-time verification
operation and is intentionally not part of this reproduction.

```sh
export PW_CHECKPOINT_ROOT="/absolute/path/to/MiMo-V2.5-63651580"
export PW_CHECKPOINT_RECEIPT="/absolute/path/to/checkpoint-verification.json"
export PW_EVIDENCE_ROOT="/absolute/path/to/prismwing-evidence"
export PW_RUN_COMMIT="$(git rev-parse HEAD)"

test -z "$(git status --porcelain)"
cargo test --lib
cargo build --release
mkdir -p "$PW_EVIDENCE_ROOT/PW-0205/reproduction"

target/release/prismwing arbitrary-text-generate \
  "$PW_CHECKPOINT_ROOT" spec/model.lock.json "$PW_CHECKPOINT_RECEIPT" \
  kernels/block_fp8_gemv.metal \
  evals/fixtures/requests/pw0204-arbitrary-text.txt \
  64 "$PW_EVIDENCE_ROOT/PW-0205/reproduction/report.json" "$PW_RUN_COMMIT"

python3 tools/audit_arbitrary_generation.py \
  "$PW_EVIDENCE_ROOT/PW-0205/reproduction/report.json" \
  "$PW_EVIDENCE_ROOT/PW-0205/reproduction/report.progress.jsonl" \
  --commit "$PW_RUN_COMMIT" \
  --prompt evals/fixtures/requests/pw0204-arbitrary-text.txt
```

The `64` is a hard maximum. The endpoint requires at least 32 committed tokens
and then stops at the second completed sentence. The audit fails closed unless
the report proves the token bound, verifier authority, exact cache/output
agreement, complete wall time, per-transaction `A` and `U`, byte accounting,
Apple M1 execution, clean Git state, stable swap/throttling, protected-service
survival, and the progress-log hash.

## Accepted reference run

Run 009 executed clean commit
`9fc6e3cd8040c7fdcf8a391b39b89d54ded97103` and stopped at 47 tokens:

> Sunlight contains all colors of the spectrum, and as it enters Earth's
> atmosphere, shorter blue wavelengths are scattered more than other colors by
> gas molecules. This scattered blue light reaches our eyes from all
> directions, making the sky appear blue.

The report SHA-256 is
`c87f2a12809c1accc52fc5d5092765ad4cb90cb9d1fa0a2f916a2ccb6d23e1b9`; the
progress SHA-256 is
`9a51a914eff401050f24310c743af6443d32bea4916a3a958b4b016cb1f8dadb`.
Complete wall time was 1,790,267.803 ms, including 422.235 ms preprocessing,
206,814.484 ms prefill, 1,287,279.699 ms proposal, and 295,265.684 ms
verification. The measured modified-mode complete-path rate was 0.026253
tokens/s. The run read 1,635,650,719,744 process bytes for 1,633,855,114,496
logical source bytes and peaked at 3,959,439,360 resident bytes. Swap growth
and new throttled pages were zero.
