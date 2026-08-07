#!/usr/bin/env python3
"""Validate and summarize PW-0130's immutable INT4 repair rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_int4_output_affine_repair_oracle import (
        BIAS_REPAIR_BYTES_PER_LAYER,
        FULL_AFFINE_REPAIR_BYTES_PER_LAYER,
        PW0129_SHA256,
        _gate,
    )
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_int4_output_affine_repair_oracle import (
        BIAS_REPAIR_BYTES_PER_LAYER,
        FULL_AFFINE_REPAIR_BYTES_PER_LAYER,
        PW0129_SHA256,
        _gate,
    )


SOURCE_SHA256 = "b011bd5ced8787df62f4380aeeccab9a35aef8b8ab15541207bcd99e35727994"
EXPECTED_BASELINE = {
    4: 0.04191861101443821,
    24: 0.11917400478835524,
    46: 0.1546056441497156,
}


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0130 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0130_int4_output_affine_repair_oracle"
        or source.get("pw0129_report_sha256") != PW0129_SHA256
        or source.get("holdout_unsealed")
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("decision") != "reject_int4_diagonal_output_affine_repair"
    ):
        raise ValueError("PW-0130 source report authority mismatch")
    reports = source["layer_reports"]
    if [row["layer"] for row in reports] != [4, 24, 46]:
        raise ValueError("PW-0130 layer report mismatch")
    summary = []
    for row in reports:
        layer = row["layer"]
        baseline = row["baseline_validation"]["relative_l2"]
        bias = row["bias"]
        affine = row["affine"]
        if (
            baseline != EXPECTED_BASELINE[layer]
            or bias["mode"] != "bias"
            or affine["mode"] != "affine"
            or bias["full_layer_parameter_bytes"] != BIAS_REPAIR_BYTES_PER_LAYER
            or affine["full_layer_parameter_bytes"] != FULL_AFFINE_REPAIR_BYTES_PER_LAYER
            or len(bias["fitted_parameter_sha256"]) != 64
            or len(affine["fitted_parameter_sha256"]) != 64
            or not (
                affine["validation_metrics"]["relative_l2"]
                <= bias["validation_metrics"]["relative_l2"]
                <= baseline
            )
        ):
            raise ValueError("PW-0130 nested repair ledger mismatch")
        summary.append(
            {
                "layer": layer,
                "baseline_relative_l2": baseline,
                "bias_relative_l2": bias["validation_metrics"]["relative_l2"],
                "affine_relative_l2": affine["validation_metrics"]["relative_l2"],
                "affine_maximum_row_relative_l2": affine["validation_metrics"]["maximum_row_relative_l2"],
                "validation_touched_experts": affine["validation_touched_experts"],
                "affine_parameter_bytes": FULL_AFFINE_REPAIR_BYTES_PER_LAYER,
                "affine_parameter_to_source_layer_bank_ratio": affine["parameter_to_source_layer_bank_ratio"],
            }
        )
    recomputed_gate = _gate(reports)
    if source["capacity_gate"] != recomputed_gate or recomputed_gate["passes"]:
        raise ValueError("PW-0130 capacity gate mismatch")
    if (
        recomputed_gate["aggregate_relative_l2"] <= 0.029
        or recomputed_gate["maximum_layer_relative_l2"] <= 0.048
        or recomputed_gate["maximum_row_relative_l2"] <= 0.069
        or not recomputed_gate["nested_oracles_monotonic"]
        or recomputed_gate["maximum_parameter_to_source_layer_bank_ratio"] > 0.002
    ):
        raise ValueError("PW-0130 expected repair rejection absent")

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
        raise ValueError("PW-0130 Gate 8 failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0130_validated_int4_output_repair_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "layer_summary": summary,
        "capacity_gate": recomputed_gate,
        "holdout_unsealed": False,
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
