#!/usr/bin/env python3
"""Validate and summarize PW-0132's immutable train-only repair result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_int4_low_rank_repair_oracle import physical_ledger
    from tools.run_train_only_int4_low_rank_repair import RANK, _gate
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_int4_low_rank_repair_oracle import physical_ledger
    from run_train_only_int4_low_rank_repair import RANK, _gate


SOURCE_SHA256 = "0499a40645452eab646276e1619fb2e94b74439ef4263a71f036fae61fd8a9fe"
EXPECTED_LAYERS = (4, 24, 46)


def _safety_summary(snapshots: list[dict]) -> dict:
    if not snapshots:
        raise ValueError("PW-0132 has no Gate 8 snapshots")
    services = snapshots[0]["protected_service_pids"]
    release_snapshots = [row for row in snapshots if row["release_boundary"]]
    if not release_snapshots:
        raise ValueError("PW-0132 has no Gate 8 release boundary")
    result = {
        "snapshot_count": len(snapshots),
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
            row["process_physical_footprint_bytes"] for row in release_snapshots
        ),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_services_stable": all(
            row["protected_service_pids"] == services for row in snapshots
        ),
    }
    if (
        result["minimum_system_memory_free_percent"] < 20
        or result["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or result["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or result["maximum_release_boundary_physical_footprint_bytes"] > 4 * 1024**3
        or result["maximum_swap_growth_bytes"] > 512 * 1024**2
        or result["maximum_new_throttled_pages"] != 0
        or not result["protected_services_stable"]
    ):
        raise ValueError("PW-0132 Gate 8 failed")
    return result


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0132 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0132_train_only_int4_rank32_repair"
        or source.get("rank") != RANK
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_pilot_train_only_rank32_int4_repair"
    ):
        raise ValueError("PW-0132 source report authority mismatch")
    reports = source["layer_reports"]
    if tuple(row["layer"] for row in reports) != EXPECTED_LAYERS:
        raise ValueError("PW-0132 layer identity mismatch")
    if any(
        len(row["affine_parameter_sha256"]) != 64
        or len(row["rank_factor_sha256"]) != 64
        for row in reports
    ):
        raise ValueError("PW-0132 fitted parameter identity mismatch")

    recomputed = _gate(reports)
    if source["validation_gate"] != recomputed:
        raise ValueError("PW-0132 validation gate mismatch")
    if recomputed["physical"] != physical_ledger(RANK):
        raise ValueError("PW-0132 physical ledger mismatch")
    if recomputed["strict_pass"] or recomputed["near_miss"]:
        raise ValueError("PW-0132 rejection conflicts with validation gate")
    if not any(
        row["rank32"]["validation"]["relative_l2"]
        > row["baseline"]["validation"]["relative_l2"]
        for row in reports
    ):
        raise ValueError("PW-0132 lacks recorded overfit regression")
    if all(row["coverage"]["validation_identity_fallback_placements"] == 0 for row in reports):
        raise ValueError("PW-0132 lacks recorded unseen-expert fallback")

    return {
        "schema_version": 1,
        "evidence_class": "pw0132_validated_train_only_int4_rank32_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "validation_gate": recomputed,
        "layer_results": [
            {
                "layer": row["layer"],
                "baseline_validation_relative_l2": row["baseline"]["validation"]["relative_l2"],
                "affine_validation_relative_l2": row["affine"]["validation"]["relative_l2"],
                "rank32_train_relative_l2": row["rank32"]["train"]["relative_l2"],
                "rank32_validation_relative_l2": row["rank32"]["validation"]["relative_l2"],
                "maximum_validation_row_relative_l2": row["rank32"]["validation"]["maximum_row_relative_l2"],
                "validation_coverage_fraction": row["coverage"]["validation_coverage_fraction"],
                "validation_identity_fallback_placements": row["coverage"]["validation_identity_fallback_placements"],
            }
            for row in reports
        ],
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branch": "weight_domain_calibration_or_mixed_precision",
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
