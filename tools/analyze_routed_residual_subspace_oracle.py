#!/usr/bin/env python3
"""Validate and summarize PW-0126's immutable validation rejection."""

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


SOURCE_SHA256 = "7a36bba9d8e6fc24cce802341ecfd56933aa05f7f4c07471004662ac414a5ffe"
LAYERS = [4, 24, 46]
RANKS = [16, 32, 64, 96, 111]


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0126 source report hash mismatch")
    source = json.loads(source_path.read_text())
    configuration = source.get("configuration", {})
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0126_routed_residual_subspace_oracle"
        or configuration.get("layers") != LAYERS
        or configuration.get("ranks") != RANKS
        or configuration.get("train_positions") != [0, 112]
        or configuration.get("validation_positions") != [112, 168]
        or configuration.get("holdout_positions") != [168, 224]
        or configuration.get("aggregate_relative_l2_maximum") != 0.01
        or configuration.get("nonempty_slice_relative_l2_maximum") != 0.02
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0126 source report authority mismatch")

    summaries = []
    if [row["layer"] for row in source["layers"]] != LAYERS:
        raise ValueError("PW-0126 layer identity mismatch")
    for row in source["layers"]:
        singular = row["singular_values"]
        training = row["training_relative_l2_by_rank"]
        validation = row["validation_by_rank"]
        if (
            len(singular) != 112
            or any(singular[index] < singular[index + 1] for index in range(111))
            or any(
                training[str(RANKS[index])] + 1e-12 < training[str(RANKS[index + 1])]
                for index in range(len(RANKS) - 1)
            )
            or training["111"] > 1e-10
            or row["selected_rank"] is not None
            or row["holdout"] is not None
        ):
            raise ValueError(f"PW-0126 layer {row['layer']} capacity authority mismatch")
        maximum_rank = validation["111"]
        if maximum_rank["aggregate_relative_l2"] <= 0.01:
            raise ValueError(f"PW-0126 layer {row['layer']} expected validation failure absent")
        for rank in RANKS:
            report = validation[str(rank)]
            if report["aggregate_relative_l2"] < 0 or any(
                slice_report["relative_l2"] is not None
                and slice_report["relative_l2"] < 0
                for slice_report in report["slices"].values()
            ):
                raise ValueError(f"PW-0126 layer {row['layer']} invalid metric")
        ledger = row["physical_ledger"]
        if (
            ledger["rank"] != 111
            or ledger["f32_mean_and_basis_bytes"] != 1_835_008
            or ledger["source_layer_bank_bytes"] != 6_444_023_808
            or ledger["oracle_output_synthesis_multiplications"] != 454_656
            or ledger["source_selected_mixture_multiplications"] != 201_326_592
            or not ledger["byte_gate_passed"]
            or not ledger["multiplication_gate_passed"]
        ):
            raise ValueError(f"PW-0126 layer {row['layer']} physical ledger mismatch")
        summaries.append(
            {
                "layer": row["layer"],
                "training_rank111_relative_l2": training["111"],
                "validation_rank111": maximum_rank,
                "artifact_to_source_bank_ratio": ledger["artifact_to_source_bank_ratio"],
                "synthesis_to_source_multiplication_ratio": ledger[
                    "synthesis_to_source_multiplication_ratio"
                ],
            }
        )

    if (
        source["validation_passed"]
        or source["holdout_unsealed"]
        or source["holdout_passed"]
        or source["gates_passed"]
        or source["decision"]
        != "reject_fixed_linear_routed_residual_dictionary_on_validation"
    ):
        raise ValueError("PW-0126 final disposition mismatch")

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
        raise ValueError("PW-0126 Gate 8 gate failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0126_validated_linear_residual_dictionary_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "layer_summaries": summaries,
        "holdout_remained_sealed": True,
        "safety": safety,
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": "reject_fixed_linear_routed_residual_dictionary_on_validation",
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
