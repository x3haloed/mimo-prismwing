#!/usr/bin/env python3
"""Validate and summarize PW-0123's immutable forced-sharing rejection."""

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


SOURCE_SHA256 = "e0f682e77d3f9ca79b762fae52534820af963b3a0478d5d4fa9944694ce5bbc2"
EXPERTS = [28, 249, 213, 125, 57]


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0123 source report hash mismatch")
    source = json.loads(source_path.read_text())
    configuration = source.get("configuration", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class")
        != "pw0123_five_expert_four_basis_sharing_pilot"
        or source.get("layer") != 46
        or source.get("experts") != EXPERTS
        or source.get("basis_count") != 4
        or source.get("rank") != 768
        or configuration.get("shared_projection_parameter_values") != 20_447_252
        or configuration.get("shared_projection_semantic_adam_bytes") != 327_156_032
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0123 source report authority mismatch")
    if not source["all_independent_projection_validation_improvement_gate_passed"]:
        raise ValueError("PW-0123 independent controls did not close")
    for report in source["independent_activation_weighted_controls"]:
        if report["source_oracle_parity"] != {
            "equality_fraction": 1.0,
            "maximum_absolute_error": 0.0,
            "relative_l2": 0.0,
        }:
            raise ValueError("PW-0123 source oracle mismatch")

    projection_summary = {}
    projection_failures = []
    for projection in ("gate", "up", "down"):
        gate = source["projection_sharing_gates"][projection]
        training = source["shared_projection_training"][projection]
        if (
            not training["improved_over_initial"]
            or training["coefficient_gradient_norm_at_first_backward"] <= 0
            or training["release_memory"]["current_allocated_bytes"] != 0
        ):
            raise ValueError(f"PW-0123 shared {projection} execution gate failed")
        failed_experts = [
            int(expert)
            for expert, ratio in gate["shared_to_independent_per_expert_ratio"].items()
            if ratio > 1.5
        ]
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
        raise ValueError("PW-0123 expected complete aggregate pass is absent")
    holdout = source["per_expert_holdout_improvement_gates"]
    failed_holdout_experts = [int(expert) for expert, row in holdout.items() if not row["passed"]]
    if projection_failures != ["gate", "up", "down"] or failed_holdout_experts != [57]:
        raise ValueError("PW-0123 expected sharing rejection topology mismatch")

    physical = source["full_bank_physical_ledger"]
    if not physical["byte_gate_passed"] or not physical["multiplication_gate_passed"]:
        raise ValueError("PW-0123 physical eligibility unexpectedly failed")
    if source["gates_passed"] or source["decision"] != (
        "reject_rank768_four_basis_sharing_under_current_objective"
    ):
        raise ValueError("PW-0123 final disposition mismatch")

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
        "maximum_new_throttled_pages": max(
            row["new_throttled_pages"] for row in snapshots
        ),
        "protected_services_stable": all(
            row["protected_service_pids"] == services for row in snapshots
        ),
        "final_physical_footprint_bytes": snapshots[-1][
            "process_physical_footprint_bytes"
        ],
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
        raise ValueError("PW-0123 Gate 8 gate failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0123_validated_five_expert_four_basis_sharing_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "independent_controls_passed": True,
        "projection_summary": projection_summary,
        "failed_projection_gates": projection_failures,
        "complete_expert_equal_mean_ratios": {
            partition: row["shared_to_independent_ratio"]
            for partition, row in complete.items()
        },
        "failed_holdout_experts": failed_holdout_experts,
        "fifth_expert_holdout": holdout["57"],
        "physical_eligibility": physical,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": "reject_rank768_four_basis_sharing_under_current_objective",
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
