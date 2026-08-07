#!/usr/bin/env python3
"""Validate and summarize PW-0135's immutable three-expert GPTQ result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_group_local_gptq_three_expert_control import (
        DAMPINGS,
        ORDERS,
        SAMPLES,
        _gate,
        physical_ledger,
    )
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_group_local_gptq_three_expert_control import (
        DAMPINGS,
        ORDERS,
        SAMPLES,
        _gate,
        physical_ledger,
    )


SOURCE_SHA256 = "56b9d38c3c630359b8d5b1a911627882df06a2e2fc374751fde2fddaeb3888db"


def _safety_summary(snapshots: list[dict]) -> dict:
    if not snapshots:
        raise ValueError("PW-0135 has no Gate 8 snapshots")
    release = [row for row in snapshots if row["release_boundary"]]
    baseline_names = {
        name for name, pids in snapshots[0]["protected_service_pids"].items() if pids
    }
    result = {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(row["system_memory_free_percent"] for row in snapshots),
        "maximum_process_peak_resident_bytes": max(row["process_peak_resident_bytes"] for row in snapshots),
        "maximum_process_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in snapshots),
        "maximum_release_boundary_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"] for row in release
        ) if release else 0,
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_services_remained_resident": all(
            all(row["protected_service_pids"].get(name) for name in baseline_names)
            for row in snapshots
        ),
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
        raise ValueError("PW-0135 Gate 8 failed")
    return result


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0135 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0135_group_local_gptq_three_expert_control"
        or source.get("samples") != [list(row) for row in SAMPLES]
        or source.get("dampings") != list(DAMPINGS)
        or source.get("orders") != list(ORDERS)
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_group_local_fixed_grid_gptq"
    ):
        raise ValueError("PW-0135 source authority mismatch")
    reports = source["reports"]
    if [(row["layer"], row["expert"]) for row in reports] != [
        (layer, expert) for layer, expert, _, _ in SAMPLES
    ]:
        raise ValueError("PW-0135 expert identity mismatch")
    for row, (_, _, train_count, validation_count) in zip(reports, SAMPLES):
        if row["train_placements"] != train_count or row["validation_placements"] != validation_count:
            raise ValueError("PW-0135 placement count mismatch")
        if row["dense_vs_packed_control_validation"]["relative_l2"] > 0.001:
            raise ValueError("PW-0135 dense oracle diverges from packed control")
        if set(row["projection_selections"]) != {"gate", "up", "down"}:
            raise ValueError("PW-0135 projection identity mismatch")
        for selection in row["projection_selections"].values():
            if (
                selection["selected_damping"] not in DAMPINGS
                or selection["selected_order"] not in ORDERS
                or len(selection["selected_grid_sha256"]) != 64
                or selection["selected_train_metrics"]["relative_l2"]
                >= selection["baseline_train_metrics"]["relative_l2"]
            ):
                raise ValueError("PW-0135 projection selection mismatch")

    recomputed = _gate(reports)
    if source["continuation_gate"] != recomputed or recomputed["physical"] != physical_ledger():
        raise ValueError("PW-0135 gate mismatch")
    if recomputed["passes"] or [row["passes"] for row in recomputed["experts"]] != [True, True, False]:
        raise ValueError("PW-0135 rejection conflicts with gate")

    return {
        "schema_version": 1,
        "evidence_class": "pw0135_validated_group_local_fixed_grid_gptq_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "continuation_gate": recomputed,
        "expert_results": [
            {
                "layer": report["layer"],
                "expert": report["expert"],
                "baseline_validation_relative_l2": report["dense_control_validation"]["relative_l2"],
                "candidate_validation_relative_l2": report["gptq_validation"]["relative_l2"],
                "candidate_maximum_row_relative_l2": report["gptq_validation"]["maximum_row_relative_l2"],
                "validation_relative_error_reduction": gate["validation_relative_error_reduction"],
                "passes": gate["passes"],
            }
            for report, gate in zip(reports, recomputed["experts"])
        ],
        "closest_failed_margin": {
            "layer": 46,
            "expert": 28,
            "criterion": "maximum_validation_relative_l2",
            "observed": reports[2]["gptq_validation"]["relative_l2"],
            "threshold": recomputed["thresholds"]["maximum_validation_relative_l2"],
            "absolute_excess": reports[2]["gptq_validation"]["relative_l2"]
            - recomputed["thresholds"]["maximum_validation_relative_l2"],
        },
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branches": ["global_hessian_gptq", "function_preserving_rotation", "recovery_training"],
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
