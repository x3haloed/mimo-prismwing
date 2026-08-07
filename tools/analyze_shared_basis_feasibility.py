#!/usr/bin/env python3
"""Compute the immutable PW-0115 shared-basis physical envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


SOURCE_MANIFEST_SHA256 = "f6cb7d8510d2076b35db074a5c6a0511fff7c047effa0dcbb6fe7a146f7aea6a"
CONTRACT_COMMIT = "2e4599ca09326e6ff1d1b079b09211b9c0271bcc"
RANKS = [16, 32, 64, 128, 256, 512, 768, 1024]
BASIS_COUNTS = [1, 2, 4, 8, 16, 32]
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 256
SELECTED = 8
ROUTED_LAYERS = 47
PROJECTIONS = 3
SOURCE_PROJECTION_BYTES = 8_390_656
SOURCE_EXPERT_BYTES = 25_171_968
SOURCE_SELECTED_LAYER_BYTES = 201_375_744
SOURCE_ROUTED_BANK_BYTES = 302_869_118_976
FOUR_GIB = 4 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration(rank: int, bases: int) -> dict:
    source_values_per_layer = PROJECTIONS * EXPERTS * INTERMEDIATE * HIDDEN
    factor_bytes_per_layer = PROJECTIONS * EXPERTS * INTERMEDIATE * rank
    basis_bytes_per_layer = PROJECTIONS * bases * rank * HIDDEN
    coefficient_bytes_per_layer = PROJECTIONS * EXPERTS * bases * 2
    bank_bytes = ROUTED_LAYERS * (
        factor_bytes_per_layer + basis_bytes_per_layer + coefficient_bytes_per_layer
    )
    selected_factor_bytes_per_layer = PROJECTIONS * SELECTED * INTERMEDIATE * rank
    source_selected_values_per_layer = PROJECTIONS * SELECTED * INTERMEDIATE * HIDDEN
    compute_values_per_layer = selected_factor_bytes_per_layer + basis_bytes_per_layer
    resident_basis_bytes = ROUTED_LAYERS * basis_bytes_per_layer
    bank_ratio = bank_bytes / (ROUTED_LAYERS * source_values_per_layer)
    selected_stream_ratio = selected_factor_bytes_per_layer / source_selected_values_per_layer
    compute_ratio = compute_values_per_layer / source_selected_values_per_layer
    gates = {
        "bank_ratio_at_most_25_percent": bank_ratio <= 0.25,
        "selected_stream_ratio_at_most_25_percent": selected_stream_ratio <= 0.25,
        "compute_ratio_at_most_50_percent": compute_ratio <= 0.50,
        "resident_basis_bytes_at_most_4_gib": resident_basis_bytes <= FOUR_GIB,
    }
    return {
        "rank": rank,
        "basis_count": bases,
        "factor_bytes_per_layer": factor_bytes_per_layer,
        "basis_bytes_per_layer": basis_bytes_per_layer,
        "coefficient_bytes_per_layer": coefficient_bytes_per_layer,
        "idealized_routed_bank_bytes": bank_bytes,
        "idealized_routed_bank_ratio": bank_ratio,
        "selected_factor_bytes_per_layer": selected_factor_bytes_per_layer,
        "selected_stream_ratio": selected_stream_ratio,
        "optimistic_compute_ratio": compute_ratio,
        "resident_basis_bytes_all_layers": resident_basis_bytes,
        "gates": gates,
        "eligible": all(gates.values()),
    }


def select_audit_configurations(eligible: list[dict]) -> list[dict]:
    if not eligible:
        return []
    rank_heavy = max(eligible, key=lambda row: (row["rank"], row["basis_count"]))
    basis_heavy = max(eligible, key=lambda row: (row["basis_count"], row["rank"]))
    excluded = {(rank_heavy["rank"], rank_heavy["basis_count"]), (basis_heavy["rank"], basis_heavy["basis_count"])}
    remaining = [
        row for row in eligible if (row["rank"], row["basis_count"]) not in excluded
    ]
    balanced = max(
        remaining,
        key=lambda row: (
            row["rank"] * row["basis_count"],
            min(row["rank"] / 512, row["basis_count"] / 8),
            row["rank"],
        ),
    )
    return [
        {"selection_role": "rank_heavy", **rank_heavy},
        {"selection_role": "balanced", **balanced},
        {"selection_role": "basis_heavy", **basis_heavy},
    ]


def analyze(source_manifest_path: Path) -> dict:
    if sha256_file(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("PW-0115 source manifest hash mismatch")
    source = json.loads(source_manifest_path.read_text())
    if (
        source.get("evidence_class")
        != "pw0113_exact_selected_expert_neuron_canonicalization"
        or source.get("source_expert_bytes") != SOURCE_EXPERT_BYTES
        or source.get("source_logical_bytes") != SOURCE_SELECTED_LAYER_BYTES
        or source.get("experts") != [9, 31, 64, 88, 96, 130, 232, 245]
        or source.get("neurons_per_expert") != INTERMEDIATE
        or source.get("artifact_sha256")
        != "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
    ):
        raise ValueError("PW-0115 source shape/byte authority mismatch")
    if (
        SOURCE_PROJECTION_BYTES * PROJECTIONS != SOURCE_EXPERT_BYTES
        or SOURCE_EXPERT_BYTES * SELECTED != SOURCE_SELECTED_LAYER_BYTES
        or SOURCE_EXPERT_BYTES * EXPERTS * ROUTED_LAYERS != SOURCE_ROUTED_BANK_BYTES
    ):
        raise ValueError("PW-0115 source byte constants do not close")
    unchanged_down_floor_ratio = (
        SOURCE_PROJECTION_BYTES * EXPERTS * ROUTED_LAYERS / SOURCE_ROUTED_BANK_BYTES
    )
    configurations = [
        configuration(rank, bases) for rank in RANKS for bases in BASIS_COUNTS
    ]
    eligible = [row for row in configurations if row["eligible"]]
    frontier = []
    for bases in BASIS_COUNTS:
        rows = [row for row in eligible if row["basis_count"] == bases]
        if rows:
            frontier.append(max(rows, key=lambda row: row["rank"]))
    selected = select_audit_configurations(eligible)
    analysis_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    analysis_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
        ).stdout.strip()
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0115_shared_basis_physical_feasibility",
        "contract_commit": CONTRACT_COMMIT,
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts_per_layer": EXPERTS,
            "selected_experts": SELECTED,
            "routed_layers": ROUTED_LAYERS,
            "projection_bytes": SOURCE_PROJECTION_BYTES,
            "expert_bytes": SOURCE_EXPERT_BYTES,
            "selected_layer_bytes": SOURCE_SELECTED_LAYER_BYTES,
            "routed_bank_bytes": SOURCE_ROUTED_BANK_BYTES,
        },
        "published_mobe_applicability": {
            "unchanged_down_floor_ratio": unchanged_down_floor_ratio,
            "pw0045_bank_ratio_limit": 0.25,
            "physical_gate_passed": unchanged_down_floor_ratio <= 0.25,
            "decision": "reject_direct_published_mobe_shape_for_pw0045_byte_gate",
            "scope": "published form keeps down projections unchanged; does not reject learned shared bases",
        },
        "optimistic_model": {
            "factor_and_basis_bytes_per_value": 1,
            "coefficient_bytes": 2,
            "candidate_scale_metadata_included": False,
            "shared_bases_resident": True,
            "shared_basis_evaluations_per_mixture": "once_per_basis",
        },
        "configurations": configurations,
        "eligible_configuration_count": len(eligible),
        "basis_count_frontier": frontier,
        "selected_activation_audit_configurations": selected,
        "all_projection_family_has_physically_eligible_configuration": bool(eligible),
        "decision": (
            "continue_all_projection_shared_basis_to_activation_weighted_audit"
            if eligible
            else "reject_current_shared_basis_parameter_family"
        ),
        "limitations": (
            "optimistic value/compute count only; no learned factors, quantization metadata, kernel, "
            "activation-weighted error, route stability, wall time, endpoint output, or accepted TPS"
        ),
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.source_manifest)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
