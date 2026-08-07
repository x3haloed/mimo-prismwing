#!/usr/bin/env python3
"""Validate and summarize PW-0121's immutable activation-weighted pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json


@dataclass(frozen=True)
class PilotAnalysisSpec:
    source_sha256: str
    source_evidence_class: str
    analysis_evidence_class: str
    layer: int
    expert: int
    partition_counts: dict[str, int]
    decision: str


PW0121_ANALYSIS_SPEC = PilotAnalysisSpec(
    source_sha256="04388f2704607657fecd5304d2533585e7ee6389080f3e77e5658a9875da05fb",
    source_evidence_class="pw0121_rank768_activation_weighted_expert_pilot",
    analysis_evidence_class="pw0121_validated_rank768_activation_weighted_expert_pilot",
    layer=24,
    expert=23,
    partition_counts={"train": 65, "validation": 46, "pilot_holdout": 56},
    decision="authorize_layer46_activation_weighted_rank768_pilot",
)


PW0122_ANALYSIS_SPEC = PilotAnalysisSpec(
    source_sha256="e05a6a5551e1ef8cb2f5593e0aa44f05a16d1c667dc531a3942e56b20196b50d",
    source_evidence_class="pw0122_layer46_rank768_activation_weighted_expert_pilot",
    analysis_evidence_class="pw0122_validated_layer46_rank768_activation_weighted_pilot",
    layer=46,
    expert=28,
    partition_counts={"train": 100, "validation": 56, "pilot_holdout": 56},
    decision="authorize_multi_expert_shared_basis_pilot_contract",
)


def analyze(
    source_path: Path, spec: PilotAnalysisSpec = PW0121_ANALYSIS_SPEC
) -> dict:
    if sha256_file(source_path) != spec.source_sha256:
        raise ValueError("activation-weighted pilot source report hash mismatch")
    source = json.loads(source_path.read_text())
    configuration = source.get("configuration", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class")
        != spec.source_evidence_class
        or source.get("layer") != spec.layer
        or source.get("expert") != spec.expert
        or source.get("partition_counts") != spec.partition_counts
        or configuration.get("rank") != 768
        or configuration.get("active_projection_parameter_values") != 4_718_592
        or configuration.get("active_projection_semantic_adam_bytes") != 75_497_472
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("activation-weighted pilot source report authority mismatch")
    if source["source_oracle_parity"] != {
        "equality_fraction": 1.0,
        "maximum_absolute_error": 0.0,
        "relative_l2": 0.0,
    }:
        raise ValueError("activation-weighted pilot source oracle gate failed")

    projection_summary = {}
    for projection in ("gate", "up", "down"):
        report = source["projection_training"][projection]
        initial = report["initial_validation_normalized_mse"]
        selected = report["selected_validation_normalized_mse"]
        history = report["validation_history"]
        if (
            not (0 <= selected < initial)
            or report["selected_step"] not in {row["step"] for row in history}
            or len(report["initial_factor_sha256"]) != 64
            or len(report["selected_factor_sha256"]) != 64
            or report["release_memory"]["current_allocated_bytes"] != 0
        ):
            raise ValueError(f"activation-weighted {projection} training/release gate failed")
        projection_summary[projection] = {
            "initial_validation_normalized_mse": initial,
            "selected_validation_normalized_mse": selected,
            "relative_validation_loss_reduction": 1.0 - selected / initial,
            "selected_step": report["selected_step"],
            "wall_ms": report["wall_ms"],
            "maximum_mps_current_allocated_bytes": max(
                row["current_allocated_bytes"] for row in report["memory"]
            ),
            "maximum_mps_driver_allocated_bytes": max(
                row["driver_allocated_bytes"] for row in report["memory"]
            ),
        }

    control = source["rank768_svd_control"]
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
    if (
        comparisons["validation"]["relative_error_reduction"] < 0.25
        or comparisons["pilot_holdout"]["relative_error_reduction"] < 0.25
        or not source["validation_relative_l2_gate"]["passed"]
        or not source["pilot_holdout_relative_l2_gate"]["passed"]
        or not source["gates_passed"]
    ):
        raise ValueError("activation-weighted pilot fidelity gate failed")

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
        raise ValueError("activation-weighted pilot Gate 8 gate failed")
    return {
        "schema_version": 1,
        "evidence_class": spec.analysis_evidence_class,
        "source_report_sha256": spec.source_sha256,
        "projection_summary": projection_summary,
        "expert_output_comparisons": comparisons,
        "source_oracle_bit_exact": True,
        "safety": safety,
        "gates_passed": True,
        "decision": spec.decision,
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
