#!/usr/bin/env python3
"""Validate and summarize PW-0133's immutable exception-store result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_train_only_int4_source_fp8_exception_store import (
        FRACTIONS,
        PW0129_SHA256,
        PW0132_SHA256,
        _gate,
        physical_ledger,
    )
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_train_only_int4_source_fp8_exception_store import (
        FRACTIONS,
        PW0129_SHA256,
        PW0132_SHA256,
        _gate,
        physical_ledger,
    )


SOURCE_SHA256 = "a0226e42058a04ea1009a6c00a6b44fdc85728bf36e383166a589b1d3e28b0d8"
EXPECTED_BASELINE = {
    4: 0.04191861101443821,
    24: 0.11917400478835524,
    46: 0.1546056441497156,
}
BASELINE_AGGREGATE = 0.09766098385443563


def _safety_summary(snapshots: list[dict]) -> dict:
    if not snapshots:
        raise ValueError("PW-0133 has no Gate 8 snapshots")
    release = [row for row in snapshots if row["release_boundary"]]
    if not release:
        raise ValueError("PW-0133 has no release snapshots")
    services = snapshots[0]["protected_service_pids"]
    result = {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(row["system_memory_free_percent"] for row in snapshots),
        "maximum_process_peak_resident_bytes": max(row["process_peak_resident_bytes"] for row in snapshots),
        "maximum_process_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in snapshots),
        "maximum_release_boundary_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in release),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_services_stable": all(row["protected_service_pids"] == services for row in snapshots),
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
        raise ValueError("PW-0133 Gate 8 failed")
    return result


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0133 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0133_train_only_int4_source_fp8_exception_store"
        or source.get("fractions") != list(FRACTIONS)
        or source.get("source_hashes", {}).get("pw0129_report_sha256") != PW0129_SHA256
        or source.get("source_hashes", {}).get("pw0132_report_sha256") != PW0132_SHA256
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_diagonal_sensitivity_source_fp8_exception_store"
    ):
        raise ValueError("PW-0133 source authority mismatch")
    reports = source["layer_reports"]
    if [row["layer"] for row in reports] != [4, 24, 46]:
        raise ValueError("PW-0133 layer identity mismatch")
    for row in reports:
        if row["baseline_validation"]["relative_l2"] != EXPECTED_BASELINE[row["layer"]]:
            raise ValueError("PW-0133 PW-0129 baseline mismatch")
        for fraction in FRACTIONS:
            if row["fractions"][str(fraction)]["physical"] != physical_ledger(fraction):
                raise ValueError("PW-0133 physical ledger mismatch")
        for expert in row["experts"]:
            if len(expert["packed_sha256"]) != 64:
                raise ValueError("PW-0133 packed identity mismatch")
            for fraction in FRACTIONS:
                projections = expert["fractions"][str(fraction)]["projections"]
                if set(projections) != {"gate", "up", "down"} or any(
                    len(record["selection_sha256"]) != 64 for record in projections.values()
                ):
                    raise ValueError("PW-0133 selection identity mismatch")

    recomputed = _gate(reports)
    if source["validation_gate"] != recomputed:
        raise ValueError("PW-0133 validation gate mismatch")
    if recomputed["smallest_strict_fraction"] is not None or recomputed["smallest_near_miss_fraction"] is not None:
        raise ValueError("PW-0133 rejection conflicts with gate")
    curve = [row["aggregate_relative_l2"] for row in recomputed["candidates"]]
    if any(later > earlier for earlier, later in zip(curve, curve[1:])):
        raise ValueError("PW-0133 aggregate curve is not monotonic")
    maximum = recomputed["candidates"][-1]
    if maximum["physical"]["combined_to_source_ratio"] > 0.60:
        raise ValueError("PW-0133 maximum candidate exceeds byte contract")

    return {
        "schema_version": 1,
        "evidence_class": "pw0133_validated_source_fp8_exception_store_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "validation_gate": recomputed,
        "maximum_admissible_candidate": maximum,
        "baseline_aggregate_relative_l2": BASELINE_AGGREGATE,
        "relative_error_reduction_at_maximum": 1.0 - maximum["aggregate_relative_l2"] / BASELINE_AGGREGATE,
        "fallback_experts": sum(row["fallback_experts"] for row in reports),
        "fallback_validation_placements": sum(row["fallback_validation_placements"] for row in reports),
        "holdout_unsealed": False,
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
        "next_branches": ["awq_weight_scaling", "gptq_second_order_updates", "quarot_rotation", "recovery_training"],
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
