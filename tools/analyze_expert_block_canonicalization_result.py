#!/usr/bin/env python3
"""Validate and decide the immutable PW-0109 canonicalization result."""

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


RAW_SHA256 = "9e0f15f65269d1b5c53536f18cda62df039d13ed19f48242f3eef91966b43bab"
IMPLEMENTATION_COMMIT = "f91121f0a3491ab41733a1e3dddc6f82e18538ee"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def codec(stream: dict, level: int) -> dict:
    matches = [record for record in stream["codec"] if record["level"] == level]
    if len(matches) != 1:
        raise ValueError(f"codec level {level} is missing or ambiguous")
    return matches[0]


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0109 raw manifest hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("evidence_class")
        != "pw0109_exact_selected_expert_block_canonicalization"
        or report.get("contract_commit") != "e3d47f1cdbc866cf70a056ba0dfe87b643ee4e82"
        or report.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or report.get("artifact_manifest_sha256")
        != "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
        or report.get("artifact_sha256")
        != "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        or report.get("experts") != [9, 31, 64, 88, 96, 130, 232, 245]
        or report.get("reference_expert") != 9
        or report.get("block_neurons") != 128
        or report.get("blocks_per_expert") != 16
        or report.get("block_bytes") != 1_573_248
        or report.get("logical_bytes") != 201_375_744
        or report.get("accepted_tokens") != 0
        or report.get("A") != 0
        or report.get("U") != 8
    ):
        raise ValueError("PW-0109 authority mismatch")
    permutations = report.get("permutations_reference_slot_to_source_block")
    if not isinstance(permutations, dict) or any(
        sorted(permutations.get(str(expert), [])) != list(range(16))
        for expert in report["experts"]
    ):
        raise ValueError("PW-0109 permutation is not bijective")
    streams = report.get("streams")
    expected_streams = {"unmodified_expert_major", "identity_delta", "aligned_delta"}
    if not isinstance(streams, dict) or set(streams) != expected_streams or any(
        stream.get("bytes") != 201_375_744
        or len(stream.get("sha256", "")) != 64
        or sorted(record.get("level") for record in stream.get("codec", [])) != [1, 19]
        for stream in streams.values()
    ):
        raise ValueError("PW-0109 stream or codec accounting mismatch")
    fast = {name: codec(stream, 1) for name, stream in streams.items()}
    high = {name: codec(stream, 19) for name, stream in streams.items()}
    aligned_fast = fast["aligned_delta"]
    identity_fast = fast["identity_delta"]
    unmodified_fast = fast["unmodified_expert_major"]
    aligned_gain_vs_identity = 1.0 - (
        aligned_fast["compressed_bytes"] / identity_fast["compressed_bytes"]
    )
    aligned_gain_vs_unmodified = 1.0 - (
        aligned_fast["compressed_bytes"] / unmodified_fast["compressed_bytes"]
    )
    canonicalization_signal_passed = (
        aligned_gain_vs_identity >= 0.10 and aligned_gain_vs_unmodified >= 0.10
    )
    physical_gate_passed = (
        aligned_fast["compressed_ratio"] <= 0.75
        and aligned_fast["optimistic_transformed_bound_ms"] <= 47.7
    )
    safety = report["safety_snapshots"]
    baseline_services = safety[0]["protected_service_pids"]
    safety_summary = {
        "minimum_system_memory_free_percent": min(
            snapshot["system_memory_free_percent"] for snapshot in safety
        ),
        "maximum_peak_resident_bytes": max(
            snapshot["process_peak_resident_bytes"] for snapshot in safety
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
        or safety_summary["maximum_swap_growth_bytes"] != 0
        or safety_summary["maximum_new_throttled_pages"] != 0
        or not safety_summary["protected_services_stable"]
    ):
        raise ValueError("PW-0109 safety gate failed")
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
        "evidence_class": "pw0109_exact_expert_block_canonicalization_analysis",
        "raw_manifest_sha256": RAW_SHA256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "fast_codec": fast,
        "high_codec": high,
        "aligned_fast_gain_vs_identity_delta": aligned_gain_vs_identity,
        "aligned_fast_gain_vs_unmodified": aligned_gain_vs_unmodified,
        "canonicalization_signal_gate_passed": canonicalization_signal_passed,
        "physical_continuation_gate_passed": physical_gate_passed,
        "exactness_and_accounting_passed": True,
        "safety": safety_summary,
        "decision": "reject_128_neuron_block_canonicalization",
        "limitations": (
            "eight selected real layer-4 experts only; zstd is a diagnostic codec; "
            "does not test arbitrary-neuron scale expansion, learned bases, or endpoint TPS"
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
