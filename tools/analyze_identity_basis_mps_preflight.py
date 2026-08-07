#!/usr/bin/env python3
"""Validate and summarize the immutable PW-0118 MPS preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.analyze_pw0116_corpus import sha256_file
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json
    from analyze_pw0116_corpus import sha256_file


SOURCE_SHA256 = "9d96d71f21f68c249b10422ec0fb479ec905874a93a5631c2488b3fc90e53c9c"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0118 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0118_identity_basis_mps_optimizer_preflight"
        or source.get("parameter_values") != 83_894_272
        or source.get("configuration", {}).get("rank") != 128
        or source.get("configuration", {}).get("basis_count") != 32
        or source.get("configuration", {}).get("activation") != "identity"
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0118 source report authority mismatch")
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
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(
            row["new_throttled_pages"] for row in snapshots
        ),
        "protected_services_stable": all(
            row["protected_service_pids"] == services for row in snapshots
        ),
        "post_release_physical_footprint_bytes": snapshots[-1][
            "process_physical_footprint_bytes"
        ],
    }
    losses = source["losses"]
    if (
        len(losses) != 5
        or losses[-1] >= losses[0]
        or safety["minimum_system_memory_free_percent"] < 20
        or safety["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or safety["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or safety["maximum_swap_growth_bytes"] > 512 * 1024**2
        or safety["maximum_new_throttled_pages"] != 0
        or not safety["protected_services_stable"]
        or safety["post_release_physical_footprint_bytes"] > 4 * 1024**3
        or source["memory_after_release"]["current_allocated_bytes"] != 0
    ):
        raise ValueError("PW-0118 optimizer or Gate 8 gate failed")
    return {
        "schema_version": 1,
        "evidence_class": "pw0118_validated_identity_basis_mps_optimizer_preflight",
        "source_report_sha256": SOURCE_SHA256,
        "parameter_values": source["parameter_values"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "relative_loss_reduction": 1.0 - losses[-1] / losses[0],
        "step_wall_ms": source["step_wall_ms"],
        "maximum_mps_current_allocated_bytes": max(
            row["current_allocated_bytes"] for row in source["memory_after_steps"]
        ),
        "maximum_mps_driver_allocated_bytes": max(
            row["driver_allocated_bytes"] for row in source["memory_after_steps"]
        ),
        "safety": safety,
        "gates_passed": True,
        "decision": "authorize_streamed_identity_basis_weight_fitting_contract",
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
