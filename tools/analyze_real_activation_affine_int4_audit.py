#!/usr/bin/env python3
"""Validate and summarize PW-0129's immutable real-activation INT4 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
    from tools.run_real_activation_affine_int4_audit import (
        BITS,
        CORPUS_SHA256,
        LAYERS,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _validation_gate,
    )
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json
    from run_real_activation_affine_int4_audit import (
        BITS,
        CORPUS_SHA256,
        LAYERS,
        SOURCE_EXPERT_BYTES,
        VERIFICATION_SHA256,
        _validation_gate,
    )


SOURCE_SHA256 = "1deb9dd85f0b598f31bc2d8bc1d41bf52cfabcda43de63a2ae5b3fdfad400306"
EXPECTED_PACKED_BYTES = {4: 13_369_344, 8: 25_952_256}


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0129 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class") != "pw0129_real_activation_affine_int4_layer_audit"
        or source.get("checkpoint_verification_sha256") != VERIFICATION_SHA256
        or source.get("corpus_manifest_sha256") != CORPUS_SHA256
        or source.get("layers") != list(LAYERS)
        or source.get("bits") != list(BITS)
        or source.get("group_size") != 128
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
        or source.get("candidate_passed")
        or source.get("holdout_unsealed")
        or source.get("holdout_reports") != []
        or source.get("holdout_gate") is not None
        or source.get("decision") != "reject_naive_affine_int4_on_real_validation"
    ):
        raise ValueError("PW-0129 source report authority mismatch")
    if len(source["source_replays"]) != len(LAYERS):
        raise ValueError("PW-0129 source replay count mismatch")
    for replay, layer in zip(source["source_replays"], LAYERS, strict=True):
        metrics = replay["metrics"]
        if (
            replay["layer"] != layer
            or len(replay["positions"]) < 2
            or metrics["relative_l2"] != 0.0
            or metrics["maximum_absolute_error"] != 0.0
            or metrics["equality_fraction"] != 1.0
        ):
            raise ValueError("PW-0129 source replay mismatch")

    reports = source["reports"]
    if len(reports) != len(LAYERS) * len(BITS):
        raise ValueError("PW-0129 report cardinality mismatch")
    validation_rows = []
    summary = []
    for layer in LAYERS:
        for bits in BITS:
            matches = [row for row in reports if row["layer"] == layer and row["bits"] == bits]
            if len(matches) != 1:
                raise ValueError("PW-0129 layer/precision report mismatch")
            report = matches[0]
            for partition_name, expected_positions in (("train", 112), ("validation", 56)):
                partition = report[partition_name]
                if (
                    partition["bits"] != bits
                    or partition["positions"] != expected_positions
                    or partition["source_bytes_per_expert"] != SOURCE_EXPERT_BYTES
                    or partition["packed_bytes_per_expert"] != EXPECTED_PACKED_BYTES[bits]
                    or partition["packed_to_source_ratio"]
                    != EXPECTED_PACKED_BYTES[bits] / SOURCE_EXPERT_BYTES
                    or partition["routed_output_metrics"]["rows"] != expected_positions
                    or not partition["expert_reports"]
                    or any(
                        row["packed_bytes"] != EXPECTED_PACKED_BYTES[bits]
                        or len(row["packed_sha256"]) != 64
                        for row in partition["expert_reports"]
                    )
                ):
                    raise ValueError("PW-0129 partition physical ledger mismatch")
            validation_rows.append({"layer": layer, **report["validation"]})
            summary.append(
                {
                    "layer": layer,
                    "bits": bits,
                    "train_relative_l2": report["train"]["routed_output_metrics"]["relative_l2"],
                    "validation_relative_l2": report["validation"]["routed_output_metrics"]["relative_l2"],
                    "validation_maximum_row_relative_l2": report["validation"]["routed_output_metrics"]["maximum_row_relative_l2"],
                    "packed_bytes_per_expert": EXPECTED_PACKED_BYTES[bits],
                    "packed_to_source_ratio": EXPECTED_PACKED_BYTES[bits] / SOURCE_EXPERT_BYTES,
                    "diagnostic_setup_wall_ms": report["validation"]["setup_wall_ms"],
                    "diagnostic_execution_wall_ms": report["validation"]["execution_wall_ms"],
                }
            )
    recomputed_gate = _validation_gate(validation_rows)
    if source["validation_gate"] != recomputed_gate or recomputed_gate["passes"]:
        raise ValueError("PW-0129 validation gate mismatch")
    if (
        recomputed_gate["aggregate_relative_l2"] <= 0.09
        or recomputed_gate["maximum_layer_relative_l2"] <= 0.15
        or recomputed_gate["maximum_row_relative_l2"] <= 0.17
        or recomputed_gate["maximum_packed_to_source_ratio"] > 0.60
    ):
        raise ValueError("PW-0129 expected decisive INT4 rejection absent")

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
        raise ValueError("PW-0129 Gate 8 failed")

    return {
        "schema_version": 1,
        "evidence_class": "pw0129_validated_real_activation_int4_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "source_replays_exact": True,
        "precision_layer_summary": summary,
        "validation_gate": recomputed_gate,
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
