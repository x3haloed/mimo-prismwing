#!/usr/bin/env python3
"""Validate and summarize PW-0131's immutable low-rank capacity result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_int4_low_rank_repair_oracle import RANKS, _capacity_gate, physical_ledger
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_int4_low_rank_repair_oracle import RANKS, _capacity_gate, physical_ledger


SOURCE_SHA256 = "e0cf60d13b3e55fd805b480bf834baa55e87f7cf5de6b49623f722c094c0d876"
EXPECTED_AFFINE = {
    4: 0.011530384904770237,
    24: 0.024850293417194323,
    46: 0.048155094657548025,
}


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0131 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0131_int4_input_conditioned_low_rank_repair_oracle"
        or source.get("ranks") != list(RANKS)
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "authorize_train_only_int4_low_rank_repair"
    ):
        raise ValueError("PW-0131 source report authority mismatch")
    for reproduction in source["affine_reproductions"]:
        if reproduction["metrics"]["relative_l2"] != EXPECTED_AFFINE[reproduction["layer"]]:
            raise ValueError("PW-0131 affine authority mismatch")
    reports = source["rank_reports"]
    if len(reports) != len(RANKS) * 3 or any(len(row["factor_sha256"]) != 64 for row in reports):
        raise ValueError("PW-0131 factor report mismatch")
    recomputed = _capacity_gate(reports)
    if source["capacity_gate"] != recomputed or not recomputed["passes"]:
        raise ValueError("PW-0131 capacity gate mismatch")
    if recomputed["smallest_passing_rank"] != 32 or not recomputed["rank_monotonic_at_every_layer"]:
        raise ValueError("PW-0131 selected rank mismatch")
    for row in recomputed["ranks"]:
        if row["physical"] != physical_ledger(row["rank"]):
            raise ValueError("PW-0131 physical ledger mismatch")

    selected = next(row for row in recomputed["ranks"] if row["rank"] == 32)
    if (
        selected["aggregate_relative_l2"] > 0.01
        or selected["maximum_layer_relative_l2"] > 0.02
        or selected["maximum_row_relative_l2"] > 0.05
        or selected["physical"]["combined_to_source_layer_bank_ratio"] > 0.60
        or selected["physical"]["repair_to_source_expert_mac_ratio"] > 0.05
    ):
        raise ValueError("PW-0131 selected capacity result misses gate")

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
        raise ValueError("PW-0131 Gate 8 failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0131_validated_low_rank_repair_capacity_pass",
        "source_report_sha256": SOURCE_SHA256,
        "capacity_gate": recomputed,
        "selected_rank": 32,
        "selected_rank_result": selected,
        "holdout_unsealed": False,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": True,
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
