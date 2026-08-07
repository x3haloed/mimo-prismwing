#!/usr/bin/env python3
"""Independently validate and summarize the immutable PW-0116 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


SOURCE_SHA256 = "b9df976876d63c1ffbbe0c70507aea8b939a749ce5b1db27cbca0b5d82cf802e"
LAYERS = [4, 24, 46]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0116 source manifest hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("semantic")
        != "mimo_pw0116_real_routed_mixture_activation_pilot_corpus"
        or source.get("target_layers") != LAYERS
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0116 source authority mismatch")
    root = source_path.parent
    summaries = []
    payload_bytes = 0
    for layer in source["layers"]:
        if layer["layer"] not in LAYERS or len(layer["selected_experts_by_position"]) != 224:
            raise ValueError("PW-0116 layer identity or position mismatch")
        if sum(layer["expert_access_counts"].values()) != 1792:
            raise ValueError("PW-0116 access count mismatch")
        if sum(len(row["positions"]) for row in layer["expert_schedule"]) != 1792:
            raise ValueError("PW-0116 schedule count mismatch")
        if sum(row["positions"] for row in layer["partition_coverage"]) != 224:
            raise ValueError("PW-0116 partition position mismatch")
        if sum(row["placements"] for row in layer["partition_coverage"]) != 1792:
            raise ValueError("PW-0116 partition placement mismatch")
        if (
            layer["routed_reconstruction_sha256"]
            != layer["captures"]["routed_output"]["sha256"]
            or layer["final_reconstruction_sha256"]
            != layer["captures"]["final"]["sha256"]
        ):
            raise ValueError("PW-0116 reconstruction identity mismatch")
        for capture in layer["captures"].values():
            path = root / capture["file"]
            if path.stat().st_size != capture["bytes"] or sha256_file(path) != capture["sha256"]:
                raise ValueError("PW-0116 payload byte or hash mismatch")
            payload_bytes += capture["bytes"]
        summaries.append(
            {
                "layer": layer["layer"],
                "distinct_experts": layer["distinct_experts"],
                "experts_with_at_most_two_placements": len(
                    layer["experts_with_at_most_two_placements"]
                ),
                "partition_coverage": layer["partition_coverage"],
            }
        )
    safety = source["safety_snapshots"]
    result = {
        "schema_version": 1,
        "evidence_class": "pw0116_validated_real_routed_mixture_activation_pilot_corpus",
        "source_manifest_sha256": SOURCE_SHA256,
        "route_semantics_sha256": source["route_semantics_sha256"],
        "payload_bytes": payload_bytes,
        "layers": summaries,
        "wall_ms": source["complete_wall_ms"],
        "ledger": source["ledger"],
        "safety": {
            "minimum_system_memory_free_percent": min(
                row["system_memory_free_percent"] for row in safety
            ),
            "maximum_process_peak_resident_bytes": max(
                row["process_peak_resident_bytes"] for row in safety
            ),
            "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in safety),
            "maximum_new_throttled_pages": max(
                row["new_throttled_pages"] for row in safety
            ),
            "final_process_physical_footprint_bytes": safety[-1][
                "process_physical_footprint_bytes"
            ],
        },
        "gates_passed": True,
        "decision": "authorize_frozen_pw0115_activation_weighted_pilot_audit_only",
        "limitations": "one correlated English trace; not representative or promotion evidence",
        "performance_claim": None,
    }
    if (
        payload_bytes != 132_120_576
        or result["safety"]["minimum_system_memory_free_percent"] < 20
        or result["safety"]["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or result["safety"]["maximum_swap_growth_bytes"] > 512 * 1024**2
        or result["safety"]["maximum_new_throttled_pages"] != 0
    ):
        raise ValueError("PW-0116 payload or Gate 8 boundary mismatch")
    return result


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
