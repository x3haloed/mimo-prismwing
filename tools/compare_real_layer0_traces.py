#!/usr/bin/env python3
"""Compare PW-0056 independent and Rust real layer-0 captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


ORDER = ["embedding", "input_norm", "qkv", "query", "key", "value",
         "attention_scores", "attention_probabilities", "attention",
         "attention_projection", "post_attention", "post_attention_norm", "gate", "up",
         "swiglu", "down", "final"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path, manifest: dict, name: str) -> np.ndarray:
    record = manifest["captures"][name]
    path = root / record["file"]
    if record.get("dtype") != "BF16_widened_F32" or sha256(path) != record.get("sha256"):
        raise ValueError(f"{name}: capture authority mismatch")
    values = np.fromfile(path, dtype="<f4")
    shape = tuple(record["shape"])
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"{name}: capture shape or values mismatch")
    return values.reshape(shape)


def compare(oracle_path: Path, rust_path: Path) -> dict:
    oracle = json.loads(oracle_path.read_text())
    rust = json.loads(rust_path.read_text())
    if (oracle.get("semantic") != "mimo_real_layer0_bf16_dynamic_fp8_oracle"
            or rust.get("semantic") != "mimo_real_layer0_bf16_dynamic_fp8_rust_trace"
            or oracle.get("revision") != rust.get("revision")
            or oracle.get("prompt_token_ids") != rust.get("prompt_token_ids")
            or oracle.get("checkpoint_verification_sha256") != rust.get("checkpoint_verification_sha256")
            or oracle.get("numerics") != "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries"
            or rust.get("numerics") != oracle.get("numerics")
            or set(oracle.get("captures", {})) != set(ORDER)
            or set(rust.get("captures", {})) != set(ORDER)):
        raise ValueError("trace manifest identity mismatch")
    rows = []
    first_failure = None
    prior_l2 = 0.0
    for name in ORDER:
        expected = load(oracle_path.parent, oracle, name)
        actual = load(rust_path.parent, rust, name)
        if expected.shape != actual.shape:
            raise ValueError(f"{name}: trace shape disagreement")
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        denominator = float(np.sum(expected.astype(np.float64) ** 2))
        relative_l2 = float(np.sqrt(np.sum(difference ** 2) / denominator)) if denominator else 0.0
        maximum = float(np.max(np.abs(difference)))
        equality = float(np.mean(actual == expected))
        flat = np.abs(difference).reshape(-1)
        top = np.argsort(flat)[-8:][::-1]
        row = {"capture": name, "shape": list(expected.shape), "relative_l2": relative_l2,
               "maximum_absolute_error": maximum, "bf16_equality_rate": equality,
               "top_difference_flat_indices": top.tolist(),
               "top_absolute_differences": flat[top].tolist()}
        rows.append(row)
        amplification = relative_l2 > max(5e-4, prior_l2 * 10.0)
        failed = relative_l2 > 5e-4 or maximum > 2e-2 or equality < 0.99
        if first_failure is None and failed:
            first_failure = {"capture": name, "amplification_over_prior": amplification}
        prior_l2 = relative_l2
    return {"schema_version": 1, "semantic": "mimo_real_layer0_trace_comparison",
            "oracle_manifest_sha256": sha256(oracle_path), "rust_manifest_sha256": sha256(rust_path),
            "captures": rows, "first_failure": first_failure,
            "layer0_provisionally_cleared": first_failure is None,
            "acceptance_effect": "diagnostic_only_no_hosted_threshold_change"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    atomic_write_new(args.output, canonical_json(compare(args.oracle, args.rust)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
