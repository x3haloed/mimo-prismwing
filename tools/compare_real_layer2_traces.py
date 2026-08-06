#!/usr/bin/env python3
"""Compare independent and Rust complete routed-layer traces."""

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

ORDER = ["incoming", "input_norm", "qkv", "query", "key", "value", "sinks",
         "attention_scores", "attention_probabilities", "attention", "attention_projection",
         "post_attention", "moe_input", "router_logits", "router_scores", "expert_gate",
         "expert_up", "expert_swiglu", "expert_down", "routed_output", "final"]
NUMERICS = "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries"

def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def load(root: Path, manifest: dict, name: str) -> np.ndarray:
    record = manifest["captures"][name]; path = root / record["file"]
    dtype = "F32" if name.startswith("router_") else "BF16_widened_F32"
    if record.get("dtype") != dtype or sha256(path) != record.get("sha256"):
        raise ValueError(f"{name}: capture authority mismatch")
    values = np.fromfile(path, dtype="<f4"); shape = tuple(record["shape"])
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"{name}: capture shape or values mismatch")
    return values.reshape(shape)

def routes(manifest: dict) -> list[dict[int, float]]:
    selected = manifest["selected_experts_by_position"]; weights = manifest["route_weights_by_position"]
    if len(selected) != 27 or len(weights) != 27: raise ValueError("route row count mismatch")
    result = []
    for experts, values in zip(selected, weights, strict=True):
        if len(experts) != 8 or len(values) != 8 or len(set(experts)) != 8: raise ValueError("route row mismatch")
        result.append(dict(zip(experts, values, strict=True)))
    return result

def validate_schedule(manifest: dict) -> list[dict]:
    expected: dict[int, list[int]] = {}
    for position, experts in enumerate(manifest["selected_experts_by_position"]):
        for expert in experts: expected.setdefault(expert, []).append(position)
    wanted = [{"expert": expert, "positions": positions} for expert, positions in sorted(expected.items())]
    if manifest.get("expert_schedule") != wanted or sum(map(lambda x: len(x["positions"]), wanted)) != 216:
        raise ValueError("expert schedule disagreement")
    return wanted

def compare(oracle_path: Path, rust_path: Path, target_layer: int = 2) -> dict:
    if target_layer not in (2, 4, 7): raise ValueError(f"unsupported routed trace layer {target_layer}")
    oracle = json.loads(oracle_path.read_text()); rust = json.loads(rust_path.read_text())
    if (oracle.get("semantic") != f"mimo_real_layer{target_layer}_complete_oracle"
            or rust.get("semantic") != f"mimo_real_layer{target_layer}_complete_rust_trace"
            or oracle.get("revision") != rust.get("revision")
            or oracle.get("prompt_token_ids") != rust.get("prompt_token_ids")
            or oracle.get("checkpoint_verification_sha256") != rust.get("checkpoint_verification_sha256")
            or oracle.get("source_input_sha256") != rust.get("source_input_sha256")
            or oracle.get("numerics") != NUMERICS or rust.get("numerics") != NUMERICS
            or set(oracle.get("captures", {})) != set(ORDER) or set(rust.get("captures", {})) != set(ORDER)
            or validate_schedule(oracle) != validate_schedule(rust)):
        raise ValueError("trace manifest identity mismatch")
    oracle_routes = routes(oracle); rust_routes = routes(rust); mismatches = []; weight_error = 0.0
    for position, (expected, actual) in enumerate(zip(oracle_routes, rust_routes, strict=True)):
        if set(expected) != set(actual): mismatches.append(position); continue
        weight_error = max(weight_error, max(abs(actual[key] - expected[key]) for key in expected))
    rows = []; first_failure = None; prior_l2 = 0.0
    for name in ORDER:
        expected = load(oracle_path.parent, oracle, name); actual = load(rust_path.parent, rust, name)
        if expected.shape != actual.shape: raise ValueError(f"{name}: shape disagreement")
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        denominator = float(np.sum(expected.astype(np.float64) ** 2))
        relative_l2 = float(np.sqrt(np.sum(difference ** 2) / denominator)) if denominator else 0.0
        maximum = float(np.max(np.abs(difference))); equality = float(np.mean(actual == expected))
        rows.append({"capture": name, "shape": list(expected.shape), "relative_l2": relative_l2,
                     "maximum_absolute_error": maximum, "equality_rate": equality})
        failed = (relative_l2 > 1e-5 or maximum > 5e-5) if name.startswith("router_") else (
            relative_l2 > 5e-4 or maximum > 2e-2 or equality < 0.99)
        if name == "final": failed = failed or relative_l2 > 4e-5 or maximum > 3e-6
        if first_failure is None and failed:
            first_failure = {"capture": name, "amplification_over_prior": relative_l2 > max(5e-4, prior_l2 * 10)}
        prior_l2 = relative_l2
    if first_failure is None and (mismatches or weight_error > 5e-7):
        first_failure = {"capture": "selected_experts_and_route_weights", "amplification_over_prior": False}
    return {"schema_version": 1, "semantic": f"mimo_real_layer{target_layer}_trace_comparison",
            "oracle_manifest_sha256": sha256(oracle_path), "rust_manifest_sha256": sha256(rust_path),
            "captures": rows, "routing": {"expert_set_mismatch_positions": mismatches,
            "maximum_weight_error_by_expert": weight_error}, "first_failure": first_failure,
            f"layer{target_layer}_provisionally_cleared": first_failure is None,
            "acceptance_effect": "diagnostic_only_no_hosted_threshold_change"}

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--oracle", required=True, type=Path)
    p.add_argument("--rust", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    p.add_argument("--layer", choices=(2, 4, 7), default=2, type=int); a = p.parse_args()
    atomic_write_new(a.output, canonical_json(compare(a.oracle, a.rust, a.layer))); return 0

if __name__ == "__main__": raise SystemExit(main())
