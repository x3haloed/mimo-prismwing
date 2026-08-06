#!/usr/bin/env python3
"""Profile the two frozen PW-0092 retained-cache endpoint reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_projection(report: dict) -> dict:
    return {
        "identity": {key: report[key] for key in (
            "schema_version", "semantic", "revision", "fixture_sha256",
            "checkpoint_verification_sha256", "prompt_token_ids",
            "generated_token_ids", "generated_text", "batch_size", "concurrency",
            "accepted_tokens", "A", "exactness", "implementation",
        )},
        "steps": [{
            **{key: step[key] for key in (
                "input_token_id", "input_token_ids", "output_token_id",
                "output_token_text", "top_logits", "full_logits",
            )},
            "layer_traces": [
                {key: value for key, value in trace.items() if key != "wall_ms"}
                for trace in step["layer_traces"]
            ],
        } for step in report["steps"]],
    }


def validate(report: dict) -> None:
    if (report.get("schema_version") != 2
            or report.get("semantic") != "mimo_v2_5_target_faithful_slow_chat_endpoint"
            or report.get("revision") != REVISION
            or report.get("generated_token_ids") != [264, 13]
            or report.get("accepted_tokens") != 2
            or report.get("batch_size") != 1 or report.get("concurrency") != 1
            or len(report.get("steps", [])) != 2
            or report["steps"][1].get("input_token_ids") != [264]
            or len(report["steps"][1].get("full_logits", [])) != 152_576
            or len(report["steps"][1].get("layer_traces", [])) != 48
            or {trace.get("cache_length") for trace in report["steps"][0]["layer_traces"]} != {27}
            or {trace.get("cache_length") for trace in report["steps"][1]["layer_traces"]} != {28}):
        raise ValueError("PW-0092 endpoint report identity mismatch")


def run_profile(report: dict, incremental_bytes: int, expert_bytes: int) -> dict:
    step = report["steps"][1]
    layers = step["layer_traces"]
    routed = layers[1:]
    full = [trace for trace in routed if trace["attention"] == "full"]
    swa = [trace for trace in routed if trace["attention"] == "sliding_window_128"]
    expert_source_bytes = expert_bytes * 47 * 8
    expert_expansions = 47 * 8 * 3
    all_expansions = 1_179
    return {
        "complete_wall_ms": report["complete_wall_ms"],
        "prefill_and_first_token_wall_ms": report["steps"][0]["wall_ms"],
        "incremental_wall_ms": step["wall_ms"],
        "incremental_token_rate_diagnostic": 1000.0 / step["wall_ms"],
        "layer_zero_wall_ms": layers[0]["wall_ms"],
        "routed_layer_wall_ms": sum(trace["wall_ms"] for trace in routed),
        "routed_layer_wall_fraction": sum(trace["wall_ms"] for trace in routed) / step["wall_ms"],
        "routed_layer_mean_ms": statistics.mean(trace["wall_ms"] for trace in routed),
        "routed_layer_median_ms": statistics.median(trace["wall_ms"] for trace in routed),
        "full_routed_layer_mean_ms": statistics.mean(trace["wall_ms"] for trace in full),
        "swa_routed_layer_mean_ms": statistics.mean(trace["wall_ms"] for trace in swa),
        "non_layer_remainder_ms": step["wall_ms"] - sum(trace["wall_ms"] for trace in layers),
        "incremental_logical_source_bytes": incremental_bytes,
        "incremental_expert_source_bytes": expert_source_bytes,
        "incremental_shared_source_bytes": incremental_bytes - expert_source_bytes,
        "expert_source_byte_fraction": expert_source_bytes / incremental_bytes,
        "incremental_fp8_matrices_expanded": all_expansions,
        "incremental_expert_fp8_matrices_expanded": expert_expansions,
        "expert_fp8_expansion_fraction": expert_expansions / all_expansions,
        "incremental_routed_expert_executions": 47 * 8,
        "two_step_actual_process_disk_bytes_read": report["ledger"]["actual_process_disk_bytes_read"],
    }


def profile(run_paths: list[Path], expected_hashes: list[str], model_path: Path) -> dict:
    if len(run_paths) != 2 or len(expected_hashes) != 2:
        raise ValueError("profile requires exactly two clean reports")
    actual_hashes = [sha256(path) for path in run_paths]
    if actual_hashes != expected_hashes:
        raise ValueError("PW-0092 report hash mismatch")
    reports = [json.loads(path.read_text()) for path in run_paths]
    for report in reports:
        validate(report)
    if semantic_projection(reports[0]) != semantic_projection(reports[1]):
        raise ValueError("PW-0092 semantic projections differ")
    model = json.loads(model_path.read_text())["constants"]
    incremental_bytes = model["pw0092_source_exact_incremental_decode"]["incremental_logical_source_bytes"]
    expert_bytes = model["source_fp8"]["expert_bytes"]
    runs = [run_profile(report, incremental_bytes, expert_bytes) for report in reports]
    incremental_times = [run["incremental_wall_ms"] for run in runs]
    routed_times = [run["routed_layer_wall_ms"] for run in runs]
    return {"schema_version": 1, "semantic": "mimo_incremental_token_cost_profile",
        "revision": REVISION, "input_sha256": actual_hashes,
        "throughput_model_sha256": sha256(model_path), "runs": runs,
        "incremental_wall_relative_spread": (max(incremental_times) - min(incremental_times))
            / statistics.mean(incremental_times),
        "routed_wall_relative_spread": (max(routed_times) - min(routed_times))
            / statistics.mean(routed_times),
        "bottleneck": "routed_expert_fp8_to_f32_expansion_and_execution",
        "acceptance_effect": "diagnostic_only_no_performance_default_or_accepted_tps"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--expected-sha256", action="append", required=True)
    parser.add_argument("--throughput-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    atomic_write_new(args.output, canonical_json(profile(
        args.run, args.expected_sha256, args.throughput_model)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
