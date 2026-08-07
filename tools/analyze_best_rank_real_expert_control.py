#!/usr/bin/env python3
"""Validate and summarize PW-0119's immutable best-rank control."""

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


SOURCE_SHA256 = "3e7729dfff3d9ab6793d8e74d29ad20bb3c877bea328ae53d9325737c717c8fb"
EXPECTED_SAMPLES = {(4, 64), (4, 10), (24, 23), (24, 101), (46, 28), (46, 0)}
RANKS = [128, 512, 768]


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0119 source report hash mismatch")
    source = json.loads(source_path.read_text())
    samples = source.get("samples", [])
    if (
        source.get("schema_version") != 1
        or source.get("evidence_class")
        != "pw0119_best_rank_real_expert_activation_control"
        or source.get("ranks") != RANKS
        or {(row.get("layer"), row.get("expert")) for row in samples}
        != EXPECTED_SAMPLES
        or len(samples) != len(EXPECTED_SAMPLES)
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0119 source report authority mismatch")

    sample_summaries = []
    for sample in samples:
        source_parity = sample["source_oracle_parity"]
        if (
            source_parity["relative_l2"] > 1e-3
            or source_parity["maximum_absolute_error"] > 0.02
        ):
            raise ValueError("PW-0119 source oracle parity gate failed")
        for projection in ("gate", "up", "down"):
            residuals = sample["projections"][projection][
                "relative_frobenius_residual_by_rank"
            ]
            ordered = [residuals[str(rank)] for rank in RANKS]
            if not all(0 <= value <= 1 for value in ordered) or not (
                ordered[0] >= ordered[1] >= ordered[2]
            ):
                raise ValueError("PW-0119 projection residual gate failed")
        rank_errors = {
            str(rank): sample["rank_controls"][str(rank)]["overall"]["relative_l2"]
            for rank in RANKS
        }
        if not all(
            rank_errors[str(RANKS[index])] >= rank_errors[str(RANKS[index + 1])]
            for index in range(len(RANKS) - 1)
        ):
            raise ValueError("PW-0119 activation error monotonicity gate failed")
        for rank in RANKS:
            control = sample["rank_controls"][str(rank)]
            if not (0 <= control["overall"]["equality_fraction"] <= 1):
                raise ValueError("PW-0119 equality fraction gate failed")
            for partition in control["partitions"].values():
                if partition["positions"] and partition["metrics"] is None:
                    raise ValueError("PW-0119 partition coverage gate failed")
        sample_summaries.append(
            {
                "layer": sample["layer"],
                "expert": sample["expert"],
                "frequency_class": sample["frequency_class"],
                "placements": sample["placements"],
                "source_oracle_parity": source_parity,
                "expert_output_relative_l2_by_rank": rank_errors,
                "rank_128_to_768_error_ratio": rank_errors["128"]
                / max(rank_errors["768"], 1e-30),
            }
        )

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
        "maximum_release_boundary_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"]
            for row in snapshots
            if row["release_boundary"]
        ),
    }
    if (
        safety["minimum_system_memory_free_percent"] < 20
        or safety["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or safety["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or safety["maximum_swap_growth_bytes"] > 512 * 1024**2
        or safety["maximum_new_throttled_pages"] != 0
        or not safety["protected_services_stable"]
        or safety["maximum_release_boundary_physical_footprint_bytes"] > 4 * 1024**3
    ):
        raise ValueError("PW-0119 Gate 8 gate failed")

    by_layer = {}
    for layer in (4, 24, 46):
        layer_samples = [row for row in sample_summaries if row["layer"] == layer]
        by_layer[str(layer)] = {
            str(rank): {
                "minimum_relative_l2": min(
                    row["expert_output_relative_l2_by_rank"][str(rank)]
                    for row in layer_samples
                ),
                "maximum_relative_l2": max(
                    row["expert_output_relative_l2_by_rank"][str(rank)]
                    for row in layer_samples
                ),
            }
            for rank in RANKS
        }
    return {
        "schema_version": 1,
        "evidence_class": "pw0119_validated_best_rank_real_expert_control",
        "source_report_sha256": SOURCE_SHA256,
        "complete_wall_ms": source["complete_wall_ms"],
        "samples": sample_summaries,
        "expert_output_relative_l2_range_by_layer_and_rank": by_layer,
        "all_source_oracles_bit_exact": all(
            row["source_oracle_parity"]["equality_fraction"] == 1.0
            for row in sample_summaries
        ),
        "all_activation_errors_improve_monotonically_with_rank": True,
        "safety": safety,
        "gates_passed": True,
        "decision": "begin_rank_768_activation_weighted_pilot_before_shared_bank",
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
