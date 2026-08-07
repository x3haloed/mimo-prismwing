#!/usr/bin/env python3
"""Validate and summarize PW-0127's immutable CPU-only rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_r720_cpu_arithmetic_ceiling import ceiling, peak_flops
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_r720_cpu_arithmetic_ceiling import ceiling, peak_flops


SOURCE_SHA256 = "6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140"
EXPECTED_MACS = {
    "attention_projections": 4_482_662_400,
    "dense_layer0_mlp": 201_326_592,
    "routers": 49_283_072,
    "selected_experts": 9_462_349_824,
    "lm_head": 624_951_296,
}


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0127 source report hash mismatch")
    source = json.loads(source_path.read_text())
    candidate = source.get("candidate", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0127_r720_cpu_arithmetic_ceiling"
        or candidate.get("sockets") != 2
        or candidate.get("cores_per_socket") != 10
        or candidate.get("granted_all_core_ghz") != 3.60
        or candidate.get("granted_sp_operations_per_cycle_per_core") != 16
        or candidate.get("memory_gib") != 512
        or candidate.get("candidate_is_owned_or_measured")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0127 source report authority mismatch")
    if source["mandatory_macs_by_category"] != EXPECTED_MACS:
        raise ValueError("PW-0127 mandatory MAC ledger mismatch")
    if sum(EXPECTED_MACS.values()) != 14_820_573_184:
        raise ValueError("PW-0127 analyzer MAC constant does not close")
    recomputed = ceiling(peak_flops(2, 10, 3.60, 16), sum(EXPECTED_MACS.values()))
    if source["arithmetic_ceiling"] != recomputed:
        raise ValueError("PW-0127 arithmetic ceiling mismatch")
    bandwidth = source["ordinary_token_bandwidth_ceiling"]
    expected_bandwidth_tps = 119.4e9 / 9_464_659_968
    if (
        bandwidth["per_socket_official_maximum_bytes_per_second"] != 59.7e9
        or bandwidth["impossible_dual_socket_bytes_per_second"] != 119.4e9
        or bandwidth["selected_expert_bytes_per_ordinary_token"] != 9_464_659_968
        or bandwidth["expert_only_ordinary_token_tps_ceiling"] != expected_bandwidth_tps
    ):
        raise ValueError("PW-0127 bandwidth ceiling mismatch")
    if (
        recomputed["impossible_maximum_tps"] >= 50
        or recomputed["targets"]["50.0"]["arithmetically_possible_at_impossible_peak"]
        or not recomputed["targets"]["34.3"]["arithmetically_possible_at_impossible_peak"]
        or recomputed["targets"]["34.3"]["required_fraction_of_impossible_peak"] < 0.8
        or source["gates_passed"]
        or source["decision"] != "reject_cpu_only_dual_e5_2680v2_for_prismwing_50"
    ):
        raise ValueError("PW-0127 expected disposition mismatch")

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
    ):
        raise ValueError("PW-0127 Gate 8 gate failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0127_validated_r720_cpu_arithmetic_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "mandatory_macs_by_category": EXPECTED_MACS,
        "arithmetic_ceiling": recomputed,
        "ordinary_token_bandwidth_ceiling": bandwidth,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": "reject_cpu_only_dual_e5_2680v2_for_prismwing_50",
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
