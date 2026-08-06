#!/usr/bin/env python3
"""Validate and summarize the immutable PW-0108 Metal-I/O report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


RAW_SHA256 = "6f7d816b4f39c00b967642bdf300e7baea8563a5fca593ab5d0943b5df047d68"
COMMIT = "ea83a9b4d9a46379ad4d6b4a9932934b03a0ab8f"
STATES = ("cold", "warm")
CONFIGURATIONS = (1, 2, 3)
PHYSICAL_BOUND_MS = 47.7
PW0107_C3_COLD_MS = (115.446542, 117.193083, 114.811708)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("PW-0108 distributions require exactly three trials")
    return statistics.median(values)


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0108 raw report hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("semantic") != "mimo_v2_5_layer4_metal_io_acquisition_bound"
        or report.get("commit") != COMMIT
        or report.get("artifact_manifest_sha256")
        != "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
        or report.get("artifact_sha256")
        != "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        or report.get("artifact_bytes") != 201_719_808
        or report.get("encoded_bytes_per_trial") != 201_375_744
        or report.get("record_count") != 48
        or report.get("metal_device") != "Apple M1"
        or report.get("selector_probe_passed") is not True
        or report.get("exact_offset_probe_passed") is not True
        or report.get("accepted_tokens") != 0
        or report.get("A") != 0
        or report.get("U") != 8
    ):
        raise ValueError("PW-0108 report authority mismatch")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != 18:
        raise ValueError("PW-0108 requires exactly 18 trials")
    grouped: dict[tuple[str, int], list[dict]] = {}
    for state in STATES:
        for command_buffers in CONFIGURATIONS:
            records = sorted(
                [
                    trial
                    for trial in trials
                    if trial.get("cache_state") == state
                    and trial.get("command_buffers") == command_buffers
                ],
                key=lambda trial: trial["repetition"],
            )
            if [trial.get("repetition") for trial in records] != [0, 1, 2]:
                raise ValueError(f"missing or duplicate {state}/{command_buffers} repetition")
            grouped[(state, command_buffers)] = records
    if any(
        trial.get("encoded_records") != 48
        or trial.get("encoded_bytes") != 201_375_744
        or trial.get("destination_capacity_bytes") != 201_719_808
        or trial.get("complete_statuses") != [3] * trial["command_buffers"]
        for trial in trials
    ):
        raise ValueError("PW-0108 transfer accounting mismatch")
    hashes = {trial["destination_records_sha256"] for trial in trials}
    if len(hashes) != 1:
        raise ValueError("PW-0108 transfer configurations differ in bytes")
    if any(
        trial["activity"]["disk_bytes_read"] != 201_719_808
        for command_buffers in CONFIGURATIONS
        for trial in grouped[("cold", command_buffers)]
    ):
        raise ValueError("PW-0108 cold physical-read ledger mismatch")
    if any(
        trial["activity"]["disk_bytes_read"] != 0 or trial["activity"]["pageins"] != 0
        for command_buffers in CONFIGURATIONS
        for trial in grouped[("warm", command_buffers)]
    ):
        raise ValueError("PW-0108 warm state performed physical reads")

    summaries = {}
    for state in STATES:
        for command_buffers in CONFIGURATIONS:
            records = grouped[(state, command_buffers)]
            summaries[f"{state}/{command_buffers}_command_buffers"] = {
                "transfer_wall_ms": [trial["transfer_wall_ms"] for trial in records],
                "median_transfer_wall_ms": median(
                    [trial["transfer_wall_ms"] for trial in records]
                ),
                "destination_initialize_ms": [
                    trial["destination_initialize_ms"] for trial in records
                ],
                "median_destination_initialize_ms": median(
                    [trial["destination_initialize_ms"] for trial in records]
                ),
                "integrity_ms": [trial["integrity_ms"] for trial in records],
                "disk_bytes_read": [
                    trial["activity"]["disk_bytes_read"] for trial in records
                ],
                "pageins": [trial["activity"]["pageins"] for trial in records],
            }
    cold_serial = summaries["cold/1_command_buffers"]["median_transfer_wall_ms"]
    cold_two = summaries["cold/2_command_buffers"]["median_transfer_wall_ms"]
    cold_three = summaries["cold/3_command_buffers"]["median_transfer_wall_ms"]
    warm_serial = summaries["warm/1_command_buffers"]["median_transfer_wall_ms"]
    warm_three = summaries["warm/3_command_buffers"]["median_transfer_wall_ms"]
    no_cold_trial_exceeds_pw0107 = all(
        trial["transfer_wall_ms"] <= PW0107_C3_COLD_MS[trial["repetition"]]
        for command_buffers in (2, 3)
        for trial in grouped[("cold", command_buffers)]
    )
    warm_concurrent_no_regression = all(
        summaries[f"warm/{command_buffers}_command_buffers"]["median_transfer_wall_ms"]
        <= warm_serial
        for command_buffers in (2, 3)
    )
    physical_bound_passed = (
        min(cold_two, cold_three) <= PHYSICAL_BOUND_MS
        and no_cold_trial_exceeds_pw0107
        and warm_concurrent_no_regression
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
        raise ValueError("PW-0108 safety gate failed")
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
        "evidence_class": "pw0108_metal_io_routed_layer_acquisition_bound_analysis",
        "raw_report_sha256": RAW_SHA256,
        "commit": report["commit"],
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "artifact_manifest_sha256": report["artifact_manifest_sha256"],
        "artifact_sha256": report["artifact_sha256"],
        "exact_offset_probe_ms": report["exact_offset_probe_ms"],
        "trial_summaries": summaries,
        "cold_two_buffer_speedup_vs_serial": cold_serial / cold_two,
        "cold_three_buffer_speedup_vs_serial": cold_serial / cold_three,
        "warm_three_buffer_speedup_vs_serial": warm_serial / warm_three,
        "physical_continuation_bound_ms": PHYSICAL_BOUND_MS,
        "best_cold_concurrent_median_ms": min(cold_two, cold_three),
        "gap_to_physical_continuation_bound_ms": min(cold_two, cold_three)
        - PHYSICAL_BOUND_MS,
        "no_cold_concurrent_trial_exceeds_pw0107_c3": no_cold_trial_exceeds_pw0107,
        "warm_concurrent_no_regression": warm_concurrent_no_regression,
        "physical_continuation_gate_passed": physical_bound_passed,
        "all_records_identical_all_trials": True,
        "destination_records_sha256": next(iter(hashes)),
        "safety": safety_summary,
        "decision": "reject_metal_io_overlap_on_internal_ssd_before_phase_b",
        "limitations": (
            "real layer-4 selected-byte acquisition bound only; post-timing full integrity scan; "
            "no compute overlap or endpoint TPS; a faster named storage device changes the premise"
        ),
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.report)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
