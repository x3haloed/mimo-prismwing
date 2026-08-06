#!/usr/bin/env python3
"""Compare PW-0059 independent and Rust selected-expert traces."""

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


ORDER = ["moe_input", "expert_gate", "expert_up", "expert_swiglu",
         "expert_down", "routed_output", "final"]
NUMERICS = "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries"


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


def route_maps(manifest: dict) -> list[dict[int, float]]:
    selected = manifest.get("selected_experts_by_position")
    weights = manifest.get("route_weights_by_position")
    if not isinstance(selected, list) or not isinstance(weights, list) or len(selected) != len(weights):
        raise ValueError("routing result shape mismatch")
    result = []
    for experts, values in zip(selected, weights, strict=True):
        if (not isinstance(experts, list) or not isinstance(values, list)
                or len(experts) != 8 or len(values) != 8 or len(set(experts)) != 8
                or any(not isinstance(expert, int) or expert < 0 for expert in experts)
                or any(not np.isfinite(value) for value in values)):
            raise ValueError("routing row mismatch")
        result.append(dict(zip(experts, values, strict=True)))
    return result


def validate_schedule(manifest: dict) -> list[dict]:
    schedule = manifest.get("expert_schedule")
    if (not isinstance(schedule, list) or len(schedule) != 28
            or [entry.get("expert") for entry in schedule] != sorted(entry.get("expert") for entry in schedule)
            or any(not isinstance(entry.get("positions"), list)
                   or entry["positions"] != sorted(entry["positions"])
                   or len(set(entry["positions"])) != len(entry["positions"])
                   for entry in schedule)
            or sum(len(entry["positions"]) for entry in schedule) != 216):
        raise ValueError("expert schedule mismatch")
    expected: dict[int, list[int]] = {}
    for position, experts in enumerate(manifest.get("selected_experts_by_position", [])):
        for expert in experts:
            expected.setdefault(expert, []).append(position)
    if schedule != [{"expert": expert, "positions": positions}
                    for expert, positions in sorted(expected.items())]:
        raise ValueError("expert schedule disagrees with routes")
    return schedule


def compare(oracle_path: Path, rust_path: Path) -> dict:
    oracle = json.loads(oracle_path.read_text())
    rust = json.loads(rust_path.read_text())
    if (oracle.get("semantic") != "mimo_real_layer1_selected_experts_oracle"
            or rust.get("semantic") != "mimo_real_layer1_selected_experts_rust_trace"
            or oracle.get("revision") != rust.get("revision")
            or oracle.get("prompt_token_ids") != rust.get("prompt_token_ids")
            or oracle.get("checkpoint_verification_sha256") != rust.get("checkpoint_verification_sha256")
            or oracle.get("source_input_sha256") != rust.get("source_input_sha256")
            or oracle.get("numerics") != NUMERICS or rust.get("numerics") != NUMERICS
            or set(oracle.get("captures", {})) != set(ORDER)
            or set(rust.get("captures", {})) != set(ORDER)
            or validate_schedule(oracle) != validate_schedule(rust)):
        raise ValueError("trace manifest identity mismatch")

    oracle_routes, rust_routes = route_maps(oracle), route_maps(rust)
    if len(oracle_routes) != len(rust_routes):
        raise ValueError("routing position count mismatch")
    set_mismatches = []
    maximum_weight_error = 0.0
    for position, (expected, actual) in enumerate(zip(oracle_routes, rust_routes, strict=True)):
        if set(expected) != set(actual):
            set_mismatches.append(position)
            continue
        maximum_weight_error = max(maximum_weight_error,
                                   max(abs(actual[key] - expected[key]) for key in expected))
    if set_mismatches or maximum_weight_error > 5e-7:
        raise ValueError("routing authority exceeds PW-0058 gate")

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
        failed = relative_l2 > 5e-4 or maximum > 2e-2 or equality < 0.99
        if name == "final":
            failed = failed or relative_l2 > 4e-5 or maximum > 3e-6
        if first_failure is None and failed:
            first_failure = {"capture": name,
                             "amplification_over_prior": relative_l2 > max(5e-4, prior_l2 * 10.0)}
        prior_l2 = relative_l2
    return {"schema_version": 1, "semantic": "mimo_real_layer1_expert_trace_comparison",
            "oracle_manifest_sha256": sha256(oracle_path),
            "rust_manifest_sha256": sha256(rust_path), "captures": rows,
            "routing": {"expert_set_mismatch_positions": set_mismatches,
                        "maximum_weight_error_by_expert": maximum_weight_error},
            "first_failure": first_failure,
            "layer1_experts_provisionally_cleared": first_failure is None,
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
