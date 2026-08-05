#!/bin/sh
set -eu

cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
python3 -m unittest discover -s tests -v

if command -v swiftc >/dev/null 2>&1 && [ "$(uname -s)" = "Darwin" ]; then
    prismwing_test_dir="$(mktemp -d)"
    trap 'rm -r "$prismwing_test_dir"' EXIT HUP INT TERM
    swiftc -O -framework Metal tools/metal_fp8_gemv.swift \
        -o "$prismwing_test_dir/metal_fp8_gemv"
    "$prismwing_test_dir/metal_fp8_gemv" \
        evals/fixtures/real/mtp-gate-fp8-gemv.json \
        kernels/block_fp8_gemv.metal
    "$prismwing_test_dir/metal_fp8_gemv" \
        evals/fixtures/real/mtp-gate-fp8-gemv.json \
        kernels/block_fp8_gemv.metal \
        block_fp8_gemv_parallel_lut_blocked 64
fi
