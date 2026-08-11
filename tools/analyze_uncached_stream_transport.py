#!/usr/bin/env python3
"""Authenticate and summarize PW-0213's isolated F_NOCACHE transport run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json


SOURCE_SHA256 = "50873d1925ac9ef3c7152a4d3ed44cfe465a4f59ea3681532ffe9780d002ca5f"
COMMIT = "5326f0dbdddcf52f57f918e55ba49c9a90a3217f"
MANIFEST_SHA256 = "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
ARTIFACT_SHA256 = "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
SCOPES = {
    "one_expert_object": {
        "records": 6,
        "logical_bytes": 25_171_968,
        "widened_bytes": 25_280_512,
        "stream_sha256": "5f9844a25f1de4c965c2a6812e8cabba7f9bd329aedfb2b6db52c05c00ee92d0",
    },
    "complete_routed_layer": {
        "records": 48,
        "logical_bytes": 201_375_744,
        "widened_bytes": 202_162_176,
        "stream_sha256": "f31aed3aa8b65ae938968d48a61134cc0b6013a51f9798bf31c9460166094927",
    },
}
TRANSPORTS = ("cacheable_pread_control", "f_nocache_pread")


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("PW-0213 median requires values")
    return float(statistics.median(values))


def _safety(snapshots: list[dict]) -> dict:
    if len(snapshots) != 15 or snapshots[-1].get("phase") != "buffer_release":
        raise ValueError("PW-0213 safety boundaries are incomplete")
    baseline = snapshots[0]["protected_service_pids"]
    summary = {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(row["system_memory_free_percent"] for row in snapshots),
        "maximum_process_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in snapshots),
        "final_process_physical_footprint_bytes": snapshots[-1]["process_physical_footprint_bytes"],
        "maximum_process_peak_resident_bytes": max(row["process_peak_resident_bytes"] for row in snapshots),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_service_pid_sets_stable": all(row["protected_service_pids"] == baseline for row in snapshots),
    }
    if (
        summary["minimum_system_memory_free_percent"] < 10
        or summary["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or summary["final_process_physical_footprint_bytes"] > 4 * 1024**3
        or summary["maximum_swap_growth_bytes"] != 0
        or summary["maximum_new_throttled_pages"] != 0
        or not summary["protected_service_pid_sets_stable"]
    ):
        raise ValueError("PW-0213 Gate 8 failed")
    return summary


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0213 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("semantic") != "mimo_v2_5_real_checkpoint_page_aligned_f_nocache_transport"
        or source.get("commit") != COMMIT
        or source.get("artifact_manifest_sha256") != MANIFEST_SHA256
        or source.get("artifact_sha256") != ARTIFACT_SHA256
        or source.get("page_bytes") != 16 * 1024
        or source.get("f_nocache_value") != 48
        or source.get("f_rdahead_value") != 45
        or source.get("nocache_enabled") is not True
        or source.get("automatic_readahead_disabled") is not True
        or source.get("maximum_buffer_bytes") != 8_404_992
        or source.get("batch_size") != 1
        or source.get("concurrency") != 1
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("U") != 8
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0213 source authority mismatch")
    trials = source.get("trials")
    if not isinstance(trials, list) or len(trials) != 12:
        raise ValueError("PW-0213 trial count mismatch")

    distributions = {}
    for scope, expected in SCOPES.items():
        distributions[scope] = {}
        for transport in TRANSPORTS:
            rows = [row for row in trials if row.get("scope") == scope and row.get("transport") == transport]
            if sorted(row.get("repetition") for row in rows) != [0, 1, 2]:
                raise ValueError("PW-0213 interleaved identity mismatch")
            for row in rows:
                if (
                    row.get("records") != expected["records"]
                    or row.get("logical_bytes") != expected["logical_bytes"]
                    or row.get("widened_bytes") != expected["widened_bytes"]
                    or row.get("pread_calls") != expected["records"]
                    or row.get("logical_stream_sha256") != expected["stream_sha256"]
                    or row.get("read_amplification", 2.0) > 1.05
                    or row.get("activity", {}).get("disk_bytes_read", 0) < 0.95 * expected["logical_bytes"]
                    or row.get("source_pages_probed", 0) <= 0
                ):
                    raise ValueError("PW-0213 transfer integrity or cold-validity gate failed")
            walls = [row["transfer_wall_ms"] for row in rows]
            resident = [row["source_resident_fraction"] for row in rows]
            distributions[scope][transport] = {
                "transfer_wall_ms": walls,
                "median_transfer_wall_ms": median(walls),
                "source_resident_fractions": resident,
                "median_source_resident_fraction": median(resident),
                "read_amplification": rows[0]["read_amplification"],
                "physical_read_bytes": [row["activity"]["disk_bytes_read"] for row in rows],
            }

    gates = {}
    for scope in SCOPES:
        control = distributions[scope]["cacheable_pread_control"]
        candidate = distributions[scope]["f_nocache_pread"]
        residency_reduction = 1.0 - (
            candidate["median_source_resident_fraction"]
            / control["median_source_resident_fraction"]
        )
        wall_reduction = 1.0 - (
            candidate["median_transfer_wall_ms"] / control["median_transfer_wall_ms"]
        )
        gates[scope] = {
            "source_file_backed_residency_reduction": residency_reduction,
            "transfer_wall_reduction": wall_reduction,
            "file_backed_gate_passed": residency_reduction >= 0.5,
            "transfer_wall_gate_passed": wall_reduction >= 0.1,
            "read_amplification_gate_passed": candidate["read_amplification"] <= 1.05,
        }
        gates[scope]["continuation_gate_passed"] = (
            gates[scope]["read_amplification_gate_passed"]
            and (gates[scope]["file_backed_gate_passed"] or gates[scope]["transfer_wall_gate_passed"])
        )
    if not all(gate["continuation_gate_passed"] for gate in gates.values()):
        raise ValueError("PW-0213 unexpectedly fails its frozen continuation gate")

    return {
        "schema_version": 1,
        "evidence_class": "pw0213_validated_uncached_real_checkpoint_transport",
        "source_report_sha256": SOURCE_SHA256,
        "implementation_commit": COMMIT,
        "distributions": distributions,
        "gates": gates,
        "safety": _safety(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": True,
        "decision": "authorize_isolated_two_buffer_overlap_candidate",
        "limitations": "one authenticated source-FP8 expert and layer on Apple M1 internal SSD; acquisition and source-page residency only; transfer wall regresses; no arithmetic, verifier, endpoint, or TPS",
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
