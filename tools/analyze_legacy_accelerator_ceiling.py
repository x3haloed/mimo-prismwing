#!/usr/bin/env python3
"""Validate and summarize PW-0128's immutable legacy-accelerator ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_legacy_accelerator_ceiling import (
        EXPECTED_MACS,
        configuration_ceiling,
        route_window_ceiling,
    )
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_legacy_accelerator_ceiling import (
        EXPECTED_MACS,
        configuration_ceiling,
        route_window_ceiling,
    )


SOURCE_SHA256 = "12a177721d520864bd628ad99b9388cfe9c467bb7ad3706a1329536ce293611a"


def summarize_windows(windows: list[dict]) -> dict:
    if not windows:
        raise ValueError("route-window summary requires at least one window")
    values = sorted(row["impossible_perfect_acceptance_tps"] for row in windows)
    return {
        "window_count": len(windows),
        "minimum_impossible_tps": values[0],
        "median_impossible_tps": values[len(values) // 2],
        "maximum_impossible_tps": values[-1],
        "minimum_transfer_bytes": min(row["source_expert_transfer_bytes"] for row in windows),
        "maximum_transfer_bytes": max(row["source_expert_transfer_bytes"] for row in windows),
        "maximum_layer_expert_records": max(row["maximum_layer_expert_records"] for row in windows),
        "maximum_three_arena_bytes": max(row["arena_residency_bytes"]["3"] for row in windows),
        "all_three_arena_capacity_gates_pass": all(
            row["three_arenas_fit_24_decimal_gb"] for row in windows
        ),
    }


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0128 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0128_legacy_24gib_accelerator_ceiling"
        or source.get("mandatory_macs_by_category") != EXPECTED_MACS
        or source.get("mandatory_macs_per_position") != sum(EXPECTED_MACS.values())
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("gates_passed")
        or source.get("decision") != "reject_named_legacy_direct_fp32_configs_on_8k_ttft"
    ):
        raise ValueError("PW-0128 source report authority mismatch")

    macs = sum(EXPECTED_MACS.values())
    expected_configurations = [
        configuration_ceiling("one_tesla_m40_24gb", 1, 7e12, macs),
        configuration_ceiling("two_tesla_m40_24gb", 2, 7e12, macs),
        configuration_ceiling("one_tesla_p40_24gb", 1, 12e12, macs),
    ]
    if source["configurations"] != expected_configurations:
        raise ValueError("PW-0128 configuration arithmetic mismatch")
    if any(row["passes_impossible_prefill_floor"] for row in expected_configurations):
        raise ValueError("PW-0128 expected prefill rejection absent")

    q137_source = source["route_ceilings"]["1"]["137"]
    if len(q137_source) != 1:
        raise ValueError("PW-0128 q=137 window authority mismatch")
    q137 = q137_source[0]
    if (
        q137["total_layer_expert_records"] != 903
        or q137["source_expert_transfer_bytes"] != 22_730_287_104
        or q137["maximum_layer_expert_records"] != 31
        or q137["arena_residency_bytes"]["3"] != 2_340_993_024
    ):
        raise ValueError("PW-0128 q=137 physical ledger mismatch")
    reconstructed = route_window_ceiling(
        137,
        0,
        [31] + [19] * 45 + [17],
        25_171_968,
        1,
    )
    for field in (
        "total_layer_expert_records",
        "source_expert_transfer_bytes",
        "maximum_layer_expert_records",
        "maximum_single_layer_expert_bytes",
        "arena_residency_bytes",
        "impossible_expert_only_transfer_seconds",
        "impossible_perfect_acceptance_tps",
    ):
        if q137[field] != reconstructed[field]:
            raise ValueError(f"PW-0128 q=137 recomputation mismatch: {field}")

    route_summary = {
        count: {q: summarize_windows(windows) for q, windows in widths.items()}
        for count, widths in source["route_ceilings"].items()
    }
    market = source["market_and_platform"]
    if (
        market.get("complete_bom_proven")
        or market.get("one_m40_subtotal_usd") != 453.75
        or market.get("two_m40_subtotal_usd") != 603.75
        or market.get("last_supported_cuda_toolkit_family") != "12.x"
    ):
        raise ValueError("PW-0128 market/platform ledger mismatch")

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
    }
    if (
        safety["minimum_system_memory_free_percent"] < 20
        or safety["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or safety["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or safety["maximum_release_boundary_physical_footprint_bytes"] > 4 * 1024**3
        or safety["maximum_swap_growth_bytes"] > 512 * 1024**2
        or safety["maximum_new_throttled_pages"] != 0
        or not safety["protected_services_stable"]
    ):
        raise ValueError("PW-0128 Gate 8 failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0128_validated_legacy_accelerator_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "configurations": expected_configurations,
        "route_summary": route_summary,
        "market_and_platform": market,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": source["decision"],
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
