#!/usr/bin/env python3
"""Compare PW-0060 independent and Rust layer-final traces."""

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


ORDER = ["embedding"] + [f"layer_{layer:02}_final" for layer in range(48)] + ["final_norm", "last_logits"]
NUMERICS = "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path, manifest: dict, name: str) -> np.ndarray:
    record = manifest["captures"][name]; path = root / record["file"]
    dtype = "F32" if name == "last_logits" else "BF16_widened_F32"
    if record.get("dtype") != dtype or sha256(path) != record.get("sha256"):
        raise ValueError(f"{name}: capture authority mismatch")
    values = np.fromfile(path, dtype="<f4"); shape = tuple(record["shape"])
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"{name}: capture shape or values mismatch")
    return values.reshape(shape)


def route_map(experts: list, weights: list) -> list[dict[int, float]]:
    if len(experts) != len(weights):
        raise ValueError("route row count mismatch")
    result = []
    for chosen, values in zip(experts, weights, strict=True):
        if len(chosen) != 8 or len(values) != 8 or len(set(chosen)) != 8:
            raise ValueError("route row mismatch")
        result.append(dict(zip(chosen, values, strict=True)))
    return result


def compare(oracle_path: Path, rust_path: Path) -> dict:
    oracle = json.loads(oracle_path.read_text()); rust = json.loads(rust_path.read_text())
    if (oracle.get("semantic") != "mimo_full_prefix_layer_final_oracle"
            or rust.get("semantic") != "mimo_full_prefix_layer_final_rust_trace"
            or oracle.get("revision") != rust.get("revision")
            or oracle.get("prompt_token_ids") != rust.get("prompt_token_ids")
            or oracle.get("checkpoint_verification_sha256") != rust.get("checkpoint_verification_sha256")
            or oracle.get("numerics") != NUMERICS or rust.get("numerics") != NUMERICS
            or set(oracle.get("captures", {})) != set(ORDER)
            or set(rust.get("captures", {})) != set(ORDER)
            or len(oracle.get("layer_traces", [])) != 48 or len(rust.get("layer_traces", [])) != 48):
        raise ValueError("trace manifest identity mismatch")
    route_rows = []; route_failure = None
    for layer, (expected, actual) in enumerate(zip(oracle["layer_traces"], rust["layer_traces"], strict=True)):
        if expected.get("layer") != layer or actual.get("layer") != layer or expected.get("attention") != actual.get("attention"):
            raise ValueError(f"layer {layer}: trace identity mismatch")
        if layer == 0:
            if expected["selected_experts_by_position"] or actual["selected_experts_by_position"]:
                raise ValueError("dense layer contains routes")
            continue
        expected_routes = route_map(expected["selected_experts_by_position"], expected["route_weights_by_position"])
        actual_routes = route_map(actual["selected_experts_by_position"], actual["route_weights_by_position"])
        mismatches = []; maximum = 0.0
        for position, (left, right) in enumerate(zip(expected_routes, actual_routes, strict=True)):
            if set(left) != set(right): mismatches.append(position); continue
            maximum = max(maximum, max(abs(right[key] - left[key]) for key in left))
        row = {"layer": layer, "expert_set_mismatch_positions": mismatches,
               "maximum_weight_error_by_expert": maximum}
        route_rows.append(row)
        if route_failure is None and (mismatches or maximum > 5e-7): route_failure = layer
    rows = []; first_failure = None
    for name in ORDER:
        expected = load(oracle_path.parent, oracle, name); actual = load(rust_path.parent, rust, name)
        if expected.shape != actual.shape: raise ValueError(f"{name}: shape disagreement")
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        denominator = float(np.sum(expected.astype(np.float64) ** 2))
        relative_l2 = float(np.sqrt(np.sum(difference ** 2) / denominator)) if denominator else 0.0
        maximum = float(np.max(np.abs(difference))); equality = float(np.mean(actual == expected))
        row = {"capture": name, "shape": list(expected.shape), "relative_l2": relative_l2,
               "maximum_absolute_error": maximum, "equality_rate": equality}
        rows.append(row)
        failed = relative_l2 > 5e-4 or maximum > 2e-2 or (name != "last_logits" and equality < 0.99)
        if name == "layer_47_final": failed = failed or relative_l2 > 4e-5 or maximum > 3e-6
        if first_failure is None and failed: first_failure = name
    if route_failure is not None and first_failure is None: first_failure = f"layer_{route_failure:02}_routes"
    oracle_logits = load(oracle_path.parent, oracle, "last_logits")
    rust_logits = load(rust_path.parent, rust, "last_logits")
    hosted = {str(token): {"oracle": float(oracle_logits[token]), "rust": float(rust_logits[token]),
                           "absolute_error": float(abs(rust_logits[token] - oracle_logits[token]))}
              for token in (9707, 0)}
    return {"schema_version": 1, "semantic": "mimo_full_prefix_trace_comparison",
            "oracle_manifest_sha256": sha256(oracle_path), "rust_manifest_sha256": sha256(rust_path),
            "captures": rows, "routes": route_rows, "first_failure": first_failure,
            "hosted_chosen_token_logits": hosted, "full_prefix_provisionally_cleared": first_failure is None,
            "acceptance_effect": "diagnostic_only_no_hosted_threshold_change"}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); atomic_write_new(args.output, canonical_json(compare(args.oracle, args.rust))); return 0


if __name__ == "__main__": raise SystemExit(main())
