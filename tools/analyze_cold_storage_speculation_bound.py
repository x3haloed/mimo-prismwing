#!/usr/bin/env python3
"""Derive PW-0110's optimistic speculation width bound from PW-0108."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


RAW_SHA256 = "6f7d816b4f39c00b967642bdf300e7baea8563a5fca593ab5d0943b5df047d68"
ROUTED_LAYERS = 47


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_tps_ceiling(accepted_over_union: float, acquisition_seconds: float) -> float:
    if accepted_over_union <= 0 or acquisition_seconds <= 0:
        raise ValueError("speculation bound inputs must be positive")
    return accepted_over_union / acquisition_seconds


def required_accepted_over_union(target_tps: float, acquisition_seconds: float) -> float:
    if target_tps <= 0 or acquisition_seconds <= 0:
        raise ValueError("speculation requirement inputs must be positive")
    return target_tps * acquisition_seconds


def minimum_width(
    target_tps: float, acquisition_seconds: float, union: float = 1.0, acceptance_rate: float = 1.0
) -> int:
    if union < 1.0 or not 0 < acceptance_rate <= 1.0:
        raise ValueError("union or acceptance rate is outside its physical domain")
    return math.ceil(required_accepted_over_union(target_tps, acquisition_seconds) * union / acceptance_rate)


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0108 raw report hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("semantic") != "mimo_v2_5_layer4_metal_io_acquisition_bound"
        or report.get("commit") != "ea83a9b4d9a46379ad4d6b4a9932934b03a0ab8f"
        or report.get("artifact_sha256")
        != "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        or report.get("encoded_bytes_per_trial") != 201_375_744
        or report.get("record_count") != 48
        or report.get("metal_device") != "Apple M1"
        or report.get("selector_probe_passed") is not True
        or report.get("exact_offset_probe_passed") is not True
    ):
        raise ValueError("PW-0110 source authority mismatch")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != 18:
        raise ValueError("PW-0110 requires all eighteen PW-0108 trials")
    cold_medians = {}
    for command_buffers in (1, 2, 3):
        records = sorted(
            [
                trial
                for trial in trials
                if trial.get("cache_state") == "cold"
                and trial.get("command_buffers") == command_buffers
            ],
            key=lambda trial: trial["repetition"],
        )
        if [trial.get("repetition") for trial in records] != [0, 1, 2] or any(
            trial.get("encoded_records") != 48
            or trial.get("encoded_bytes") != 201_375_744
            or trial.get("activity", {}).get("disk_bytes_read") != 201_719_808
            or trial.get("complete_statuses") != [3] * command_buffers
            for trial in records
        ):
            raise ValueError("PW-0110 cold trial accounting mismatch")
        cold_medians[command_buffers] = statistics.median(
            trial["transfer_wall_ms"] for trial in records
        )
    hashes = {trial["destination_records_sha256"] for trial in trials}
    if len(hashes) != 1:
        raise ValueError("PW-0110 source integrity mismatch")
    safety = report["safety_snapshots"]
    baseline_services = safety[0]["protected_service_pids"]
    if (
        min(snapshot["system_memory_free_percent"] for snapshot in safety) < 20
        or max(snapshot["swap_growth_bytes"] for snapshot in safety) != 0
        or max(snapshot["new_throttled_pages"] for snapshot in safety) != 0
        or any(snapshot["protected_service_pids"] != baseline_services for snapshot in safety)
    ):
        raise ValueError("PW-0110 source safety mismatch")
    best_command_buffers = min(cold_medians, key=cold_medians.get)
    best_layer_ms = cold_medians[best_command_buffers]
    acquisition_ms = best_layer_ms * ROUTED_LAYERS
    acquisition_seconds = acquisition_ms / 1000.0
    widths = (8, 16, 32, 64, 128, 137)
    targets = (10.0, 25.0, 34.3, 50.0)
    ideal_width_ceilings = {
        str(width): accepted_tps_ceiling(float(width), acquisition_seconds) for width in widths
    }
    target_requirements = {
        str(target): {
            "required_A_over_U": required_accepted_over_union(target, acquisition_seconds),
            "minimum_q_at_A_equals_q_U_equals_1": minimum_width(target, acquisition_seconds),
        }
        for target in targets
    }
    sensitivity = {}
    for union in (1.0, 1.5, 2.0, 2.3351063829787235):
        for acceptance_rate in (1.0, 0.9, 0.75):
            sensitivity[f"U={union:.12g}/acceptance={acceptance_rate:.2f}"] = {
                "minimum_q_for_34_3": minimum_width(
                    34.3, acquisition_seconds, union, acceptance_rate
                ),
                "minimum_q_for_50": minimum_width(
                    50.0, acquisition_seconds, union, acceptance_rate
                ),
            }
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
        "evidence_class": "pw0110_cold_storage_speculation_width_bound",
        "source_report_sha256": RAW_SHA256,
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "best_command_buffers": best_command_buffers,
        "cold_layer_medians_ms": {str(key): value for key, value in cold_medians.items()},
        "routed_layers": ROUTED_LAYERS,
        "ideal_minimum_union": 1.0,
        "ideal_complete_acceptance": True,
        "routed_acquisition_floor_ms_per_verifier_block": acquisition_ms,
        "ideal_width_accepted_tps_ceilings": ideal_width_ceilings,
        "target_requirements": target_requirements,
        "sensitivity": sensitivity,
        "q16_rejected_for_50": ideal_width_ceilings["16"] < 50.0,
        "q32_rejected_for_50": ideal_width_ceilings["32"] < 50.0,
        "q32_rejected_for_34_3": ideal_width_ceilings["32"] < 34.3,
        "destination_records_sha256": next(iter(hashes)),
        "decision": "reject_q16_q32_source_fp8_internal_ssd_widths_raise_minimum_to_q137_for_50",
        "omitted_costs": [
            "dense and attention weight acquisition",
            "all target arithmetic",
            "draft execution",
            "I/O and compute barriers",
            "KV work and memory",
            "exact correction and rollback",
            "sampling and control overhead",
            "thermal and sustained-run effects",
        ],
        "limitations": (
            "necessary optimistic source-FP8 internal-SSD bound only; does not prove q137 "
            "acceptance, route union, memory fitness, fidelity, or endpoint TPS"
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
