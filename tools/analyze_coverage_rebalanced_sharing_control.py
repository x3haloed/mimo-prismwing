#!/usr/bin/env python3
"""Validate and summarize PW-0124's immutable coverage control."""

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


SOURCE_SHA256 = "086cd06b66aa79117e44f3b17e3f1b18b751640d1696e3ce6f3045a769586077"
PARENT_SHA256 = "4d4469184eda8717a12643a58b111d0a4fd6ac72585eb6aaabcfc6c187ab6438"
EXPERTS = [28, 249, 213, 125, 57]
EXPECTED_COUNTS = {
    "28": {"train": 124, "validation": 32, "pilot_holdout": 56},
    "249": {"train": 116, "validation": 30, "pilot_holdout": 56},
    "213": {"train": 113, "validation": 29, "pilot_holdout": 46},
    "125": {"train": 84, "validation": 21, "pilot_holdout": 56},
    "57": {"train": 58, "validation": 15, "pilot_holdout": 56},
}


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0124 source report hash mismatch")
    source = json.loads(source_path.read_text())
    configuration = source.get("configuration", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0124_coverage_rebalanced_sharing_control"
        or source.get("parent_analysis_sha256") != PARENT_SHA256
        or source.get("layer") != 46
        or source.get("experts") != EXPERTS
        or source.get("basis_count") != 4
        or source.get("rank") != 768
        or source.get("expected_partition_counts") != EXPECTED_COUNTS
        or configuration.get("partition_mode") != "development_occurrence_mod5"
        or configuration.get("seed") != 260124
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0124 source report authority mismatch")
    if not source["all_independent_projection_validation_improvement_gate_passed"]:
        raise ValueError("PW-0124 independent controls did not close")
    for report in source["independent_activation_weighted_controls"]:
        if report["source_oracle_parity"] != {
            "equality_fraction": 1.0,
            "maximum_absolute_error": 0.0,
            "relative_l2": 0.0,
        }:
            raise ValueError("PW-0124 source oracle mismatch")

    projection_summary = {}
    projection_failures = []
    failed_projection_experts = set()
    for projection in ("gate", "up", "down"):
        gate = source["projection_sharing_gates"][projection]
        training = source["shared_projection_training"][projection]
        if (
            not training["improved_over_initial"]
            or training["coefficient_gradient_norm_at_first_backward"] <= 0
            or training["release_memory"]["current_allocated_bytes"] != 0
        ):
            raise ValueError(f"PW-0124 shared {projection} execution gate failed")
        failed_experts = [
            int(expert)
            for expert, ratio in gate["shared_to_independent_per_expert_ratio"].items()
            if ratio > 1.5
        ]
        failed_projection_experts.update(failed_experts)
        projection_summary[projection] = {
            "shared_to_independent_aggregate_ratio": gate[
                "shared_to_independent_aggregate_ratio"
            ],
            "failed_experts": failed_experts,
            "fifth_expert_ratio": gate["shared_to_independent_per_expert_ratio"]["57"],
            "selected_step": training["selected_step"],
        }
        if not gate["aggregate_gate_passed"] or not gate["per_expert_gate_passed"]:
            projection_failures.append(projection)

    complete = source["complete_expert_sharing_gates"]
    if not all(row["aggregate_gate_passed"] for row in complete.values()):
        raise ValueError("PW-0124 expected complete aggregate pass is absent")
    holdout = source["per_expert_holdout_improvement_gates"]
    if not all(row["passed"] for row in holdout.values()):
        raise ValueError("PW-0124 expected holdout improvement pass is absent")
    if projection_failures != ["gate", "up", "down"] or 57 not in failed_projection_experts:
        raise ValueError("PW-0124 expected sharing rejection topology mismatch")

    physical = source["full_bank_physical_ledger"]
    if not physical["byte_gate_passed"] or not physical["multiplication_gate_passed"]:
        raise ValueError("PW-0124 physical eligibility unexpectedly failed")
    if source["gates_passed"] or source["decision"] != (
        "reject_coverage_scarcity_as_four_basis_failure_explanation"
    ):
        raise ValueError("PW-0124 final disposition mismatch")

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
        raise ValueError("PW-0124 Gate 8 gate failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0124_validated_coverage_scarcity_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "parent_analysis_sha256": PARENT_SHA256,
        "partition_counts": EXPECTED_COUNTS,
        "independent_controls_passed": True,
        "projection_summary": projection_summary,
        "failed_projection_gates": projection_failures,
        "failed_projection_experts": sorted(failed_projection_experts),
        "complete_expert_equal_mean_ratios": {
            partition: row["shared_to_independent_ratio"]
            for partition, row in complete.items()
        },
        "all_holdout_improvement_gates_passed": True,
        "fifth_expert_holdout": holdout["57"],
        "physical_eligibility": physical,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": "reject_coverage_scarcity_as_four_basis_failure_explanation",
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
