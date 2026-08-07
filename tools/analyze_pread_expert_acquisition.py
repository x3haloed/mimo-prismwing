#!/usr/bin/env python3
"""Validate and summarize PW-0136's immutable explicit-pread result."""

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


SOURCE_SHA256 = "e6ab84cada19c6036ee7b83f318c3920631141b9ea5e882cc88eb9784d0b5a56"
ARTIFACT_SHA256 = "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
MANIFEST_SHA256 = "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
ARTIFACT_BYTES = 201_719_808
WORKERS = [1, 2, 4, 8]


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("PW-0136 median requires values")
    return float(statistics.median(values))


def _safety_summary(snapshots: list[dict]) -> dict:
    if not snapshots or snapshots[-1]["phase"] != "buffer_release":
        raise ValueError("PW-0136 lacks final Gate 8 release")
    baseline_names = {
        name for name, pids in snapshots[0]["protected_service_pids"].items() if pids
    }
    result = {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(row["system_memory_free_percent"] for row in snapshots),
        "maximum_process_peak_resident_bytes": max(row["process_peak_resident_bytes"] for row in snapshots),
        "maximum_process_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in snapshots),
        "final_process_physical_footprint_bytes": snapshots[-1]["process_physical_footprint_bytes"],
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_services_remained_resident": all(
            all(row["protected_service_pids"].get(name) for name in baseline_names)
            for row in snapshots
        ),
        "protected_service_pid_sets_stable": all(
            row["protected_service_pids"] == snapshots[0]["protected_service_pids"]
            for row in snapshots
        ),
    }
    if (
        result["minimum_system_memory_free_percent"] < 20
        or result["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or result["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or result["final_process_physical_footprint_bytes"] > 4 * 1024**3
        or result["maximum_swap_growth_bytes"] > 512 * 1024**2
        or result["maximum_new_throttled_pages"] != 0
        or not result["protected_services_remained_resident"]
    ):
        raise ValueError("PW-0136 Gate 8 failed")
    return result


def analyze(source_path: Path) -> dict:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("PW-0136 source report hash mismatch")
    source = json.loads(source_path.read_text())
    if (
        source.get("schema_version") != 1
        or source.get("semantic") != "mimo_v2_5_layer4_page_aligned_pread_expert_slot_acquisition"
        or source.get("artifact_manifest_sha256") != MANIFEST_SHA256
        or source.get("artifact_sha256") != ARTIFACT_SHA256
        or source.get("artifact_bytes") != ARTIFACT_BYTES
        or source.get("expert_stride_bytes") != 25_214_976
        or source.get("expert_count") != 8
        or source.get("worker_counts") != WORKERS
        or source.get("slot_capacity_bytes") != ARTIFACT_BYTES
        or source.get("slot_alignment_bytes") != 2 * 1024 * 1024
        or source.get("slot_buffer_pointer_identity") != [True] * 8
        or source.get("slot_buffer_lengths") != [25_214_976] * 8
        or source.get("accepted_tokens") != 0
        or source.get("A") != 0
        or source.get("U") != 8
        or source.get("performance_claim") is not None
    ):
        raise ValueError("PW-0136 source authority mismatch")
    trials = source["trials"]
    if len(trials) != 24:
        raise ValueError("PW-0136 trial count mismatch")
    for state in ("cold", "warm"):
        for workers in WORKERS:
            rows = [row for row in trials if row["cache_state"] == state and row["workers"] == workers]
            if sorted(row["repetition"] for row in rows) != [0, 1, 2]:
                raise ValueError("PW-0136 interleaved trial identity mismatch")
            for row in rows:
                if (
                    row["requested_bytes"] != ARTIFACT_BYTES
                    or row["returned_bytes"] != ARTIFACT_BYTES
                    or row["pread_calls"] != 8
                    or row["slot_stream_sha256"] != ARTIFACT_SHA256
                    or len(row["expert_reads"]) != 8
                    or any(
                        expert["requested_bytes"] != 25_214_976
                        or expert["returned_bytes"] != 25_214_976
                        or expert["pread_calls"] != 1
                        for expert in row["expert_reads"]
                    )
                ):
                    raise ValueError("PW-0136 transfer integrity mismatch")
                if state == "cold" and row["activity"]["disk_bytes_read"] < 0.95 * ARTIFACT_BYTES:
                    raise ValueError("PW-0136 cold physical-read gate failed")
                if state == "warm" and row["activity"]["disk_bytes_read"] != 0:
                    raise ValueError("PW-0136 warm trial unexpectedly read disk")

    distributions = {
        state: {
            str(workers): {
                "trial_walls_ms": [
                    row["transfer_wall_ms"]
                    for row in trials
                    if row["cache_state"] == state and row["workers"] == workers
                ],
                "median_wall_ms": median([
                    row["transfer_wall_ms"]
                    for row in trials
                    if row["cache_state"] == state and row["workers"] == workers
                ]),
            }
            for workers in WORKERS
        }
        for state in ("cold", "warm")
    }
    selected_workers = min(
        WORKERS,
        key=lambda workers: (distributions["cold"][str(workers)]["median_wall_ms"], workers),
    )
    selected_cold = distributions["cold"][str(selected_workers)]
    selected_warm = distributions["warm"][str(selected_workers)]
    serial_warm = distributions["warm"]["1"]["median_wall_ms"]
    gate = {
        "selected_workers": selected_workers,
        "cold_median_ms": selected_cold["median_wall_ms"],
        "cold_maximum_ms": max(selected_cold["trial_walls_ms"]),
        "warm_median_ms": selected_warm["median_wall_ms"],
        "serial_warm_median_ms": serial_warm,
        "cold_median_limit_ms": 47.7,
        "cold_trial_maximum_limit_ms": 57.723,
        "warm_no_regression": selected_warm["median_wall_ms"] <= serial_warm,
    }
    gate["passes"] = (
        gate["cold_median_ms"] <= gate["cold_median_limit_ms"]
        and gate["cold_maximum_ms"] <= gate["cold_trial_maximum_limit_ms"]
        and gate["warm_no_regression"]
    )
    if gate["passes"]:
        raise ValueError("PW-0136 unexpectedly passes frozen rejection interpretation")

    return {
        "schema_version": 1,
        "evidence_class": "pw0136_validated_page_aligned_pread_acquisition_rejection",
        "source_report_sha256": SOURCE_SHA256,
        "distributions": distributions,
        "physical_continuation_gate": gate,
        "metal_io_cold_median_ms": 58.034,
        "metal_io_warm_three_buffer_median_ms": 14.782,
        "best_pread_warm_workers": min(
            WORKERS,
            key=lambda workers: (distributions["warm"][str(workers)]["median_wall_ms"], workers),
        ),
        "safety": _safety_summary(source["safety_snapshots"]),
        "evidence_valid": True,
        "experiment_passed": False,
        "decision": "reject_parallel_pread_for_internal_ssd_source_fp8",
        "next_branch": "fidelity_qualified_int4_then_pread_slots",
        "limitations": "one authenticated source-FP8 layer and internal SSD; acquisition only; no GPU ownership, cache, scheduler, compute overlap, endpoint, or TPS",
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
