#!/usr/bin/env python3
"""Validate and summarize PW-0125's immutable rank-512 capacity rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json


SOURCE_SHA256 = "916ab149169a518d68eace66f2a6d857679c8e6e5e1777f604c904f0179b08e0"
RANK768_VALIDATION = 0.19566708093351987
RANK768_HOLDOUT = 0.2881280111180422


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0125 source report hash mismatch")
    source = json.loads(source_path.read_text())
    configuration = source.get("configuration", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class")
        != "pw0125_rank512_activation_weighted_capacity_control"
        or source.get("layer") != 46
        or source.get("expert") != 28
        or source.get("partition_counts")
        != {"train": 100, "validation": 56, "pilot_holdout": 56}
        or configuration.get("rank") != 512
        or configuration.get("seed") != 260125
        or configuration.get("active_projection_parameter_values") != 3_145_728
        or configuration.get("active_projection_semantic_adam_bytes") != 50_331_648
        or configuration.get("balanced_initialization_relative_l2_tolerance") != 1e-5
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0125 source report authority mismatch")
    if source["source_oracle_parity"] != {
        "equality_fraction": 1.0,
        "maximum_absolute_error": 0.0,
        "relative_l2": 0.0,
    }:
        raise ValueError("PW-0125 source oracle gate failed")

    projection_summary = {}
    for projection in ("gate", "up", "down"):
        report = source["projection_training"][projection]
        initial = report["initial_validation_normalized_mse"]
        selected = report["selected_validation_normalized_mse"]
        if (
            not (0 <= selected < initial)
            or report["selected_step"]
            not in {row["step"] for row in report["validation_history"]}
            or report["release_memory"]["current_allocated_bytes"] != 0
        ):
            raise ValueError(f"PW-0125 {projection} training/release gate failed")
        projection_summary[projection] = {
            "initial_validation_normalized_mse": initial,
            "selected_validation_normalized_mse": selected,
            "relative_validation_loss_reduction": 1.0 - selected / initial,
            "selected_step": report["selected_step"],
        }

    control = source["rank512_svd_control"]
    candidate = source["activation_weighted_candidate"]
    comparisons = {}
    for name in ("train", "validation", "pilot_holdout"):
        baseline = control["partitions"][name]["metrics"]["relative_l2"]
        fitted = candidate["partitions"][name]["metrics"]["relative_l2"]
        comparisons[name] = {
            "svd_relative_l2": baseline,
            "activation_weighted_relative_l2": fitted,
            "relative_error_reduction": 1.0 - fitted / baseline,
        }
    comparisons["overall"] = {
        "svd_relative_l2": control["overall"]["relative_l2"],
        "activation_weighted_relative_l2": candidate["overall"]["relative_l2"],
        "relative_error_reduction": 1.0
        - candidate["overall"]["relative_l2"] / control["overall"]["relative_l2"],
    }
    capacity_ratios = {
        "validation_to_rank768_fitted": (
            comparisons["validation"]["activation_weighted_relative_l2"]
            / RANK768_VALIDATION
        ),
        "holdout_to_rank768_fitted": (
            comparisons["pilot_holdout"]["activation_weighted_relative_l2"]
            / RANK768_HOLDOUT
        ),
    }
    if (
        comparisons["validation"]["relative_error_reduction"] < 0.25
        or comparisons["pilot_holdout"]["relative_error_reduction"] < 0.25
        or source["validation_relative_l2_gate"]["passed"]
        or not source["pilot_holdout_relative_l2_gate"]["passed"]
        or capacity_ratios["validation_to_rank768_fitted"] <= 1.25
        or capacity_ratios["holdout_to_rank768_fitted"] > 1.25
        or source["gates_passed"]
        or source["decision"] != "reject_rank512_eight_basis_branch_on_independent_capacity"
    ):
        raise ValueError("PW-0125 expected capacity rejection topology mismatch")

    snapshots = source["safety_snapshots"]
    services = snapshots[0]["protected_service_pids"]
    safety = {
        "minimum_system_memory_free_percent": min(
            row["system_memory_free_percent"] for row in snapshots
        ),
        "maximum_process_peak_resident_bytes": max(
            row["process_peak_resident_bytes"] for row in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"] for row in snapshots
        ),
        "maximum_release_boundary_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"]
            for row in snapshots
            if row["release_boundary"]
        ),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_services_stable": all(
            row["protected_service_pids"] == services for row in snapshots
        ),
        "final_physical_footprint_bytes": snapshots[-1]["process_physical_footprint_bytes"],
    }
    if (
        safety["minimum_system_memory_free_percent"] < 20
        or safety["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or safety["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or safety["maximum_release_boundary_physical_footprint_bytes"] > 4 * 1024**3
        or safety["maximum_swap_growth_bytes"] > 512 * 1024**2
        or safety["maximum_new_throttled_pages"] != 0
        or not safety["protected_services_stable"]
        or source["final_mps_memory"]["current_allocated_bytes"] != 0
    ):
        raise ValueError("PW-0125 Gate 8 gate failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0125_validated_rank512_capacity_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "projection_summary": projection_summary,
        "expert_output_comparisons": comparisons,
        "capacity_ratios": capacity_ratios,
        "source_oracle_bit_exact": True,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": "reject_rank512_eight_basis_branch_on_independent_capacity",
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
