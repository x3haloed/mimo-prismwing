#!/usr/bin/env python3
"""Validate and summarize PW-0134's immutable AWQ-style result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_train_only_awq_int4_expert_rescaling import ALPHAS, _gate, physical_ledger
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_train_only_awq_int4_expert_rescaling import ALPHAS, _gate, physical_ledger


SOURCE_SHA256 = "7d470bd5fa5541424c2b619afb49a2ebf493ce7a11b2498cf281b3d1c6f34490"
BASELINE_AGGREGATE = 0.09766098385443563
EXPECTED_BASELINE = {
    4: 0.04191861101443821,
    24: 0.11917400478835524,
    46: 0.1546056441497156,
}


def _safety_summary(snapshots: list[dict]) -> dict:
    if not snapshots:
        raise ValueError("PW-0134 has no Gate 8 snapshots")
    release = [row for row in snapshots if row["release_boundary"]]
    baseline_names = {
        name for name, pids in snapshots[0]["protected_service_pids"].items() if pids
    }
    services_remained = all(
        all(row["protected_service_pids"].get(name) for name in baseline_names)
        for row in snapshots
    )
    result = {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(row["system_memory_free_percent"] for row in snapshots),
        "maximum_process_peak_resident_bytes": max(row["process_peak_resident_bytes"] for row in snapshots),
        "maximum_process_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in snapshots),
        "maximum_release_boundary_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in release),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_services_remained_resident": services_remained,
        "protected_service_pid_sets_stable": all(
            row["protected_service_pids"] == snapshots[0]["protected_service_pids"]
            for row in snapshots
        ),
    }
    if (
        not release
        or result["minimum_system_memory_free_percent"] < 20
        or result["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or result["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or result["maximum_release_boundary_physical_footprint_bytes"] > 4 * 1024**3
        or result["maximum_swap_growth_bytes"] > 512 * 1024**2
        or result["maximum_new_throttled_pages"] != 0
        or not result["protected_services_remained_resident"]
    ):
        raise ValueError("PW-0134 Gate 8 failed")
    return result


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0134 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0134_train_only_awq_int4_expert_rescaling"
        or source.get("alphas") != list(ALPHAS)
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_awq_activation_mean_scale_family"
    ):
        raise ValueError("PW-0134 source authority mismatch")
    reports = source["layer_reports"]
    if [row["layer"] for row in reports] != [4, 24, 46]:
        raise ValueError("PW-0134 layer identity mismatch")
    for row in reports:
        if row["baseline_validation"]["relative_l2"] != EXPECTED_BASELINE[row["layer"]]:
            raise ValueError("PW-0134 PW-0129 baseline mismatch")
        if any(
            expert["input_alpha"] not in ALPHAS
            or expert["hidden_alpha"] not in ALPHAS
            or len(expert["scale_sha256"]) != 64
            or len(expert["packed_sha256"]) != 64
            or expert["packed_bytes"] != physical_ledger()["int4_bytes_per_expert"]
            or expert["transform_reconstruction_relative_l2"] > 1e-6
            for expert in row["experts"]
        ):
            raise ValueError("PW-0134 expert calibration identity mismatch")
        for expert in row["experts"]:
            calibration = expert["calibration"]
            if expert["fallback"] == "none":
                if (
                    calibration is None
                    or [item["alpha"] for item in calibration["input_curve"]] != list(ALPHAS)
                    or [item["alpha"] for item in calibration["hidden_curve"]] != list(ALPHAS)
                ):
                    raise ValueError("PW-0134 calibration curve mismatch")
            elif calibration is not None:
                raise ValueError("PW-0134 fallback unexpectedly has calibration")

    recomputed = _gate(reports)
    if source["validation_gate"] != recomputed or recomputed["physical"] != physical_ledger():
        raise ValueError("PW-0134 gate mismatch")
    if recomputed["strict_pass"] or recomputed["near_miss"]:
        raise ValueError("PW-0134 rejection conflicts with gate")
    if any(
        row["candidate_validation"]["relative_l2"] >= row["baseline_validation"]["relative_l2"]
        for row in reports
    ):
        raise ValueError("PW-0134 lacks monotonic layer improvement")

    return {
        "schema_version": 1,
        "evidence_class": "pw0134_validated_awq_activation_mean_scale_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "validation_gate": recomputed,
        "baseline_aggregate_relative_l2": BASELINE_AGGREGATE,
        "relative_error_reduction": 1.0 - recomputed["aggregate_relative_l2"] / BASELINE_AGGREGATE,
        "layer_results": [
            {
                "layer": row["layer"],
                "baseline_relative_l2": row["baseline_validation"]["relative_l2"],
                "candidate_relative_l2": row["candidate_validation"]["relative_l2"],
                "maximum_row_relative_l2": row["candidate_validation"]["maximum_row_relative_l2"],
                "median_input_alpha": row["median_input_alpha"],
                "median_hidden_alpha": row["median_hidden_alpha"],
                "fallback_experts": row["fallback_experts"],
                "fallback_validation_placements": row["fallback_validation_placements"],
            }
            for row in reports
        ],
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branches": ["gptq_second_order_updates", "quarot_rotation", "recovery_training"],
        "limitations": source["limitations"],
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.source)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
