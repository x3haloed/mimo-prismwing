#!/usr/bin/env python3
"""Validate and summarize the immutable PW-0111 canonical report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


RAW_SHA256 = "47f764370172dff489629bb171d9dad7345f39e37f21622244a63b6f4edfcb14"
IMPLEMENTATION_COMMIT = "1334876bce437816dd4502b0486c39852f207226"
EXPECTED_INSTALLED_BYTES = 201_375_744


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("expected three positive trial walls")
    return statistics.median(values)


def select(trials: list[dict], state: str, field: str) -> list[float]:
    values = [trial[field] for trial in trials if trial["cache_state"] == state]
    if len(values) != 3:
        raise ValueError(f"expected three {state} {field} values")
    return values


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0111 raw report hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("semantic")
        != "mimo_v2_5_layer4_one_barrier_metal_native_l3_benchmark"
        or report.get("commit") != IMPLEMENTATION_COMMIT
        or report.get("metal_device") != "Apple M1"
        or report.get("primitive_probe_passed") is not True
        or report.get("A") != 0
        or report.get("U") != 8
        or report.get("accepted_tokens") != 0
        or report.get("performance_claim") is not None
    ):
        raise ValueError("PW-0111 report authority mismatch")
    controls = report.get("control_trials")
    candidates = report.get("candidate_trials")
    if not isinstance(controls, list) or len(controls) != 6:
        raise ValueError("PW-0111 requires six C2 controls")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("PW-0111 requires six C4 candidates")
    if len(report.get("trial_order", [])) != 12:
        raise ValueError("PW-0111 trial-order ledger mismatch")
    for trial in controls:
        if (
            trial.get("variant") != "C2_artifact_no_copy"
            or trial.get("installed_source_bytes") != EXPECTED_INSTALLED_BYTES
            or trial.get("sparse_repair_counts") != [6, 4, 3]
            or len(trial.get("expert_diagnostics", [])) != 8
            or len(trial.get("expert_tomography", [])) != 8
        ):
            raise ValueError("PW-0111 C2 accounting mismatch")
    for trial in candidates:
        transaction = trial.get("transaction", {})
        if (
            trial.get("variant") != "C4_one_barrier_metal_native"
            or trial.get("installed_source_bytes") != EXPECTED_INSTALLED_BYTES
            or trial.get("sparse_repair_counts") != [0, 0, 0]
            or len(trial.get("expert_diagnostics", [])) != 8
            or transaction.get("command_buffers") != 1
            or transaction.get("encoders") != 6
            or transaction.get("commits") != 1
            or transaction.get("waits") != 1
            or transaction.get("projection_dispatches") != 24
            or transaction.get("kernel_dispatches") != 28
            or transaction.get("final_residual_readbacks") != 1
            or transaction.get("error_flags") != 0
            or transaction.get("scratch_high_water_bytes", 1 << 60) >= 1 << 30
            or transaction.get("total_metal_resource_bytes", 1 << 60)
            > transaction.get("recommended_max_working_set_size", 0)
        ):
            raise ValueError("PW-0111 C4 accounting mismatch")
        adjusted = trial["raw_layer_wall_ms"] - trial["safety_observation_ms"]
        transaction_adjusted = (
            transaction["raw_wall_ms"] - transaction["safety_observation_ms"]
        )
        if abs(adjusted - trial["layer_wall_ms"]) > 1e-6:
            raise ValueError("PW-0111 layer safety-observation ledger mismatch")
        if abs(transaction_adjusted - transaction["wall_ms"]) > 1e-6:
            raise ValueError("PW-0111 transaction safety-observation ledger mismatch")

    control_routed = {trial["routed_sha256"] for trial in controls}
    control_final = {trial["final_residual_sha256"] for trial in controls}
    candidate_routed = {trial["routed_sha256"] for trial in candidates}
    candidate_final = {trial["final_residual_sha256"] for trial in candidates}
    exact_control_identity = (
        len(control_routed) == len(control_final) == 1
        and candidate_routed == control_routed
        and candidate_final == control_final
    )
    if not exact_control_identity:
        raise ValueError("PW-0111 candidate output is not deterministic C2 identity")

    cold_control = select(controls, "cold", "layer_wall_ms")
    cold_candidate = select(candidates, "cold", "layer_wall_ms")
    warm_control = select(controls, "warm", "layer_wall_ms")
    warm_candidate = select(candidates, "warm", "layer_wall_ms")
    cold_control_median = median(cold_control)
    cold_candidate_median = median(cold_candidate)
    warm_control_median = median(warm_control)
    warm_candidate_median = median(warm_candidate)
    cold_speedup = cold_control_median / cold_candidate_median
    warm_speedup = warm_control_median / warm_candidate_median
    paired_cold_nonregression = all(
        candidate < control
        for candidate, control in zip(cold_candidate, cold_control, strict=True)
    )
    cold_gate_passed = cold_speedup >= 2.0 and paired_cold_nonregression
    warm_gate_passed = warm_candidate_median <= warm_control_median

    safety = report.get("safety_snapshots", [])
    if not safety:
        raise ValueError("PW-0111 lacks Gate 8 snapshots")
    safety_summary = {
        "minimum_system_memory_free_percent": min(
            snapshot["system_memory_free_percent"] for snapshot in safety
        ),
        "maximum_peak_resident_bytes": max(
            snapshot["process_peak_resident_bytes"] for snapshot in safety
        ),
        "maximum_swap_growth_bytes": max(
            snapshot["swap_growth_bytes"] for snapshot in safety
        ),
        "maximum_new_throttled_pages": max(
            snapshot["new_throttled_pages"] for snapshot in safety
        ),
        "final_physical_footprint_bytes": safety[-1][
            "process_physical_footprint_bytes"
        ],
    }
    if (
        safety_summary["minimum_system_memory_free_percent"] < 20
        or safety_summary["maximum_peak_resident_bytes"] > 8 * 1024**3
        or safety_summary["maximum_swap_growth_bytes"] > 512 * 1024**2
        or safety_summary["maximum_new_throttled_pages"] != 0
        or safety_summary["final_physical_footprint_bytes"] > 4 * 1024**3
    ):
        raise ValueError("PW-0111 Gate 8 failed")

    analysis_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    analysis_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
        ).stdout.strip()
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0111_one_barrier_metal_native_analysis",
        "raw_report_sha256": RAW_SHA256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "cold_control_wall_ms": cold_control,
        "cold_candidate_wall_ms": cold_candidate,
        "cold_control_median_ms": cold_control_median,
        "cold_candidate_median_ms": cold_candidate_median,
        "cold_speedup": cold_speedup,
        "paired_cold_nonregression": paired_cold_nonregression,
        "cold_two_x_gate_passed": cold_gate_passed,
        "warm_control_wall_ms": warm_control,
        "warm_candidate_wall_ms": warm_candidate,
        "warm_control_median_ms": warm_control_median,
        "warm_candidate_median_ms": warm_candidate_median,
        "warm_speedup": warm_speedup,
        "warm_nonregression_gate_passed": warm_gate_passed,
        "candidate_cold_wait_ms": select(
            [
                {"cache_state": trial["cache_state"], "value": trial["transaction"]["synchronous_wait_ms"]}
                for trial in candidates
            ],
            "cold",
            "value",
        ),
        "candidate_cold_gpu_interval_ms": select(
            [
                {"cache_state": trial["cache_state"], "value": trial["transaction"]["gpu_interval_ms"]}
                for trial in candidates
            ],
            "cold",
            "value",
        ),
        "exact_c2_output_identity": exact_control_identity,
        "candidate_sparse_repair_counts": [0, 0, 0],
        "control_sparse_repair_counts": [6, 4, 3],
        "routed_parity": candidates[0]["routed_parity"],
        "final_residual_parity": candidates[0]["final_residual_parity"],
        "safety": safety_summary,
        "decision": "reject_full_bank_and_token_walk_retain_warm_layer_mechanism",
        "limitations": "one authenticated layer-local row; unchanged source-derived layer gate still fails; no endpoint TPS or hosted parity claim",
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.report)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

