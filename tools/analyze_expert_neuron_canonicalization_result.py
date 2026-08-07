#!/usr/bin/env python3
"""Validate and decide the immutable PW-0113 neuron result."""

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


RAW_SHA256 = "f6cb7d8510d2076b35db074a5c6a0511fff7c047effa0dcbb6fe7a146f7aea6a"
IMPLEMENTATION_COMMIT = "a0ef5c4c6b95197d921cae52db5a474637f6550b"
CONTRACT_COMMIT = "e6cd914bee4b448d04864e2473e4e573698756d3"
SOURCE_BYTES = 201_375_744
EXPANDED_BYTES = 207_618_048
EXPERT_IDS = [9, 31, 64, 88, 96, 130, 232, 245]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def codec(stream: dict, level: int) -> dict:
    matches = [record for record in stream.get("codec", []) if record.get("level") == level]
    if len(matches) != 1:
        raise ValueError(f"codec level {level} is missing or ambiguous")
    return matches[0]


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0113 raw manifest hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("evidence_class")
        != "pw0113_exact_selected_expert_neuron_canonicalization"
        or report.get("contract_commit") != CONTRACT_COMMIT
        or report.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or report.get("artifact_manifest_sha256")
        != "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
        or report.get("artifact_sha256")
        != "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        or report.get("experts") != EXPERT_IDS
        or report.get("reference_expert") != 9
        or report.get("neurons_per_expert") != 2048
        or report.get("neuron_bytes") != 12_672
        or report.get("source_expert_bytes") != 25_171_968
        or report.get("expanded_expert_bytes") != 25_952_256
        or report.get("expanded_overhead_bytes_per_expert") != 780_288
        or report.get("permutation_bytes_per_expert") != 4096
        or report.get("source_logical_bytes") != SOURCE_BYTES
        or report.get("expanded_logical_bytes") != EXPANDED_BYTES
        or report.get("accepted_tokens") != 0
        or report.get("performance_claim") is not None
    ):
        raise ValueError("PW-0113 authority mismatch")
    if report["representation_overhead_fraction"] > 0.10:
        raise ValueError("PW-0113 representation overhead gate failed")
    permutations = report.get("permutations_reference_slot_to_source_neuron")
    if not isinstance(permutations, dict) or any(
        sorted(permutations.get(str(expert), [])) != list(range(2048))
        for expert in EXPERT_IDS
    ):
        raise ValueError("PW-0113 permutation is not bijective")
    assignments = report.get("assignment_evidence")
    if not isinstance(assignments, dict) or set(assignments) != {str(expert) for expert in EXPERT_IDS[1:]} or any(
        evidence.get("feature_dimensions") != 384
        or evidence.get("primary_quantization_levels") != 1_000_000_000
        or len(evidence.get("combined_cost_sha256", "")) != 64
        for evidence in assignments.values()
    ):
        raise ValueError("PW-0113 assignment evidence mismatch")
    streams = report.get("streams")
    expected_streams = {"expanded_expert_major", "identity_delta", "aligned_delta"}
    if not isinstance(streams, dict) or set(streams) != expected_streams or any(
        stream.get("bytes") != EXPANDED_BYTES
        or len(stream.get("sha256", "")) != 64
        or sorted(record.get("level") for record in stream.get("codec", [])) != [1, 19]
        for stream in streams.values()
    ):
        raise ValueError("PW-0113 stream accounting mismatch")
    fast = {name: codec(stream, 1) for name, stream in streams.items()}
    high = {name: codec(stream, 19) for name, stream in streams.items()}
    aligned_fast = fast["aligned_delta"]
    aligned_gain_vs_expanded = 1.0 - (
        aligned_fast["compressed_bytes"] / fast["expanded_expert_major"]["compressed_bytes"]
    )
    aligned_gain_vs_identity = 1.0 - (
        aligned_fast["compressed_bytes"] / fast["identity_delta"]["compressed_bytes"]
    )
    source_byte_reduction = 1.0 - aligned_fast["compressed_bytes"] / SOURCE_BYTES
    signal_passed = aligned_gain_vs_expanded >= 0.10 and aligned_gain_vs_identity >= 0.10
    physical_passed = (
        source_byte_reduction >= 0.25
        and aligned_fast["optimistic_transformed_bound_ms"] <= 47.7
    )
    safety = report.get("safety_snapshots")
    if not isinstance(safety, list) or len(safety) != 13:
        raise ValueError("PW-0113 safety ledger mismatch")
    baseline_services = safety[0]["protected_service_pids"]
    safety_summary = {
        "minimum_system_memory_free_percent": min(
            snapshot["system_memory_free_percent"] for snapshot in safety
        ),
        "maximum_peak_resident_bytes": max(
            snapshot["process_peak_resident_bytes"] for snapshot in safety
        ),
        "maximum_process_physical_footprint_bytes": max(
            snapshot["process_physical_footprint_bytes"] for snapshot in safety
        ),
        "post_release_physical_footprint_bytes": safety[-1]["process_physical_footprint_bytes"],
        "maximum_swap_growth_bytes": max(snapshot["swap_growth_bytes"] for snapshot in safety),
        "maximum_new_throttled_pages": max(
            snapshot["new_throttled_pages"] for snapshot in safety
        ),
        "protected_services_stable": all(
            snapshot["protected_service_pids"] == baseline_services for snapshot in safety
        ),
    }
    if (
        safety_summary["minimum_system_memory_free_percent"] < 20
        or safety_summary["maximum_peak_resident_bytes"] > 8 * 1024**3
        or safety_summary["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or safety_summary["post_release_physical_footprint_bytes"] > 4 * 1024**3
        or safety_summary["maximum_swap_growth_bytes"] > 512 * 1024**2
        or safety_summary["maximum_new_throttled_pages"] != 0
        or not safety_summary["protected_services_stable"]
    ):
        raise ValueError("PW-0113 safety gate failed")
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
        "evidence_class": "pw0113_exact_neuron_canonicalization_analysis",
        "raw_manifest_sha256": RAW_SHA256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "representation_overhead_fraction": report["representation_overhead_fraction"],
        "fast_codec": fast,
        "high_codec": high,
        "aligned_fast_gain_vs_expanded_control": aligned_gain_vs_expanded,
        "aligned_fast_gain_vs_identity_delta": aligned_gain_vs_identity,
        "aligned_fast_source_byte_reduction": source_byte_reduction,
        "canonicalization_signal_gate_passed": signal_passed,
        "physical_continuation_gate_passed": physical_passed,
        "exactness_and_accounting_passed": True,
        "safety": safety_summary,
        "decision": "reject_exact_fine_grained_neuron_canonicalization",
        "limitations": (
            "eight selected real layer-4 experts; deterministic statistical-feature assignment; "
            "zstd is diagnostic; does not test learned bases, sign symmetry, modified weights, or endpoint TPS"
        ),
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.manifest)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
