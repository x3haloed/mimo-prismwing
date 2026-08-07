#!/usr/bin/env python3
"""Validate and summarize PW-0120's immutable rejected preflight."""

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


SOURCE_SHA256 = "8e1a597fc5f15e98fffe2afb0e14964777b7fc5251e5bcb8bf60ae8923d5b2db"


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0120 source report hash mismatch")
    source = json.loads(source_path.read_text())
    configuration = source.get("configuration", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0120_rank768_mps_optimizer_preflight"
        or configuration.get("rank") != 768
        or configuration.get("basis_count") != 4
        or configuration.get("mps_memory_fraction") != 0.6
        or source.get("parameter_values") != 415_237_120
        or source.get("parameter_bytes") != 1_660_948_480
        or source.get("semantic_parameter_gradient_adam_bytes") != 6_643_793_920
        or source.get("succeeded") is not False
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0120 source report authority mismatch")
    failure = source["failure"]
    if (
        failure.get("type") != "RuntimeError"
        or "MPS backend out of memory" not in failure.get("message", "")
        or "Tried to allocate 1.50 GiB" not in failure.get("message", "")
    ):
        raise ValueError("PW-0120 expected allocator rejection is absent")
    memory = source["memory_by_phase"]
    required_phases = {
        "production_parameters_allocated",
        "optimizer_and_source_ready",
        "forward_loss_complete",
        "backward_gradients_complete",
        "optimizer_and_parameters_released",
    }
    if set(memory) != required_phases:
        raise ValueError("PW-0120 phase ledger mismatch")
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
        "release_physical_footprint_bytes": next(
            row["process_physical_footprint_bytes"]
            for row in snapshots
            if row["release_boundary"]
        ),
    }
    live_safety_preserved = (
        safety["minimum_system_memory_free_percent"] >= 20
        and safety["maximum_process_peak_resident_bytes"] <= 8 * 1024**3
        and safety["maximum_process_physical_footprint_bytes"] <= 8 * 1024**3
        and safety["maximum_swap_growth_bytes"] <= 512 * 1024**2
        and safety["maximum_new_throttled_pages"] == 0
        and safety["protected_services_stable"]
    )
    release_gate_passed = safety["release_physical_footprint_bytes"] <= 4 * 1024**3
    if (
        not live_safety_preserved
        or release_gate_passed
        or memory["optimizer_and_parameters_released"]["current_allocated_bytes"] != 0
    ):
        raise ValueError("PW-0120 rejection/safety interpretation mismatch")
    return {
        "schema_version": 1,
        "evidence_class": "pw0120_validated_rank768_mps_optimizer_preflight_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "parameter_values": source["parameter_values"],
        "semantic_parameter_gradient_adam_bytes": source[
            "semantic_parameter_gradient_adam_bytes"
        ],
        "loss_before_rejected_step": source["loss"],
        "memory_by_phase": memory,
        "allocator_failure": failure,
        "live_safety_preserved": live_safety_preserved,
        "release_gate_passed": release_gate_passed,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": "reject_direct_full_state_rank768_mps_adam_and_contract_low_memory_optimizer",
        "limitations": "one source tile and one attempted Adam step; rejects this direct full-state MPS Adam embodiment, not rank-768 activation-weighted fitting with a lower-memory optimizer",
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
    except (OSError, ValueError, KeyError, TypeError, StopIteration, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
