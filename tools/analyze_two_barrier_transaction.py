#!/usr/bin/env python3
"""Validate and summarize the immutable PW-0107 interleaved layer report."""

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


RAW_SHA256 = "39d2a678212a7d98aee33396119928c0e9c2baa7aa4e9f5a19c63ce0fd005bd2"
COMMIT = "09d6efd64418c9eebdea1b6dbcb053adba03feab"
CONTROL = "C2_serial_no_copy_24_barriers"
CANDIDATE = "C3_two_barrier_no_copy"
STATES = ("cold", "warm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("PW-0107 distributions require exactly three trials")
    return statistics.median(values)


def candidate_stage_totals(trial: dict) -> dict[str, float]:
    transaction = trial["transaction"]
    phases = transaction["phases"]
    return {
        "transaction_wall_ms": transaction["wall_ms"],
        "source_buffer_bind_ms": sum(phase["source_buffer_bind_ms"] for phase in phases),
        "small_buffer_install_ms": sum(phase["small_buffer_install_ms"] for phase in phases),
        "synchronous_wait_ms": sum(phase["synchronous_wait_ms"] for phase in phases),
        "gpu_interval_ms": sum(phase["gpu_interval_ms"] for phase in phases),
        "dynamic_input_ms": transaction["dynamic_input_ms"],
        "gate_up_cpu_stage_ms": transaction["gate_up_cpu_stage_ms"],
        "dynamic_hidden_ms_subset": transaction["dynamic_hidden_ms"],
        "down_cpu_stage_ms": transaction["down_cpu_stage_ms"],
        "weighted_scatter_ms": trial["weighted_scatter_ms"],
    }


def serial_stage_totals(trial: dict) -> dict[str, float]:
    experts = trial["serial_expert_tomography"]
    projections = [projection for expert in experts for projection in expert["projections"]]
    return {
        "synchronous_wait_ms": sum(item["synchronous_wait_ms"] for item in projections),
        "gpu_interval_ms": sum(item["gpu_interval_ms"] for item in projections),
        "source_buffer_bind_ms": sum(item["source_buffer_install_ms"] for item in projections),
    }


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0107 raw report hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("semantic")
        != "mimo_v2_5_layer4_two_barrier_no_copy_transaction_benchmark"
        or report.get("commit") != COMMIT
        or report.get("artifact_manifest_sha256")
        != "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
        or report.get("artifact_sha256")
        != "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        or report.get("selected_experts") != [232, 31, 64, 96, 9, 88, 245, 130]
        or report.get("maximum_route_weight_absolute_error") != 0.0
        or report.get("accepted_tokens") != 0
        or report.get("A") != 0
        or report.get("U") != 8
    ):
        raise ValueError("PW-0107 report authority mismatch")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != 12:
        raise ValueError("PW-0107 requires exactly 12 trials")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for state in STATES:
        for variant in (CONTROL, CANDIDATE):
            records = [
                trial
                for trial in trials
                if trial.get("cache_state") == state and trial.get("variant") == variant
            ]
            if sorted(trial.get("repetition") for trial in records) != [0, 1, 2]:
                raise ValueError(f"missing or duplicate {state}/{variant} repetition")
            grouped[(state, variant)] = records
    if any(
        trial.get("installed_source_bytes") != 201_375_744
        or trial.get("sparse_repair_counts") != [6, 4, 3]
        or len(trial.get("expert_diagnostics", [])) != 8
        for trial in trials
    ):
        raise ValueError("PW-0107 trial accounting mismatch")
    controls = [trial for trial in trials if trial["variant"] == CONTROL]
    candidates = [trial for trial in trials if trial["variant"] == CANDIDATE]
    if any(
        len(trial.get("serial_expert_tomography", [])) != 8
        or trial.get("transaction") is not None
        for trial in controls
    ):
        raise ValueError("PW-0107 serial-control topology mismatch")
    for trial in candidates:
        transaction = trial.get("transaction")
        if (
            trial.get("serial_expert_tomography", []) != []
            or not isinstance(transaction, dict)
            or transaction.get("command_buffers") != 2
            or transaction.get("commits") != 2
            or transaction.get("waits") != 2
            or transaction.get("projection_dispatches") != 24
            or [phase.get("projection_dispatches") for phase in transaction.get("phases", [])]
            != [16, 8]
            or any(
                phase["gpu_interval_ms"] > phase["synchronous_wait_ms"] + 1.0
                for phase in transaction.get("phases", [])
            )
        ):
            raise ValueError("PW-0107 candidate command topology mismatch")
    routed_hashes = {trial["routed_sha256"] for trial in trials}
    final_hashes = {trial["final_residual_sha256"] for trial in trials}
    diagnostic_payloads = {
        json.dumps(trial["expert_diagnostics"], sort_keys=True, separators=(",", ":"))
        for trial in trials
    }
    if len(routed_hashes) != 1 or len(final_hashes) != 1 or len(diagnostic_payloads) != 1:
        raise ValueError("PW-0107 variants do not reproduce identical outputs")

    summaries = {}
    for state in STATES:
        for variant in (CONTROL, CANDIDATE):
            records = grouped[(state, variant)]
            totals = [
                candidate_stage_totals(trial)
                if variant == CANDIDATE
                else serial_stage_totals(trial)
                for trial in records
            ]
            summaries[f"{state}/{variant}"] = {
                "layer_wall_ms": [trial["layer_wall_ms"] for trial in records],
                "median_layer_wall_ms": median([trial["layer_wall_ms"] for trial in records]),
                "disk_bytes_read": [trial["activity"]["disk_bytes_read"] for trial in records],
                "pageins": [trial["activity"]["pageins"] for trial in records],
                "median_stage_totals_ms": {
                    field: median([record[field] for record in totals]) for field in totals[0]
                },
            }
    cold_control = summaries[f"cold/{CONTROL}"]["median_layer_wall_ms"]
    cold_candidate = summaries[f"cold/{CANDIDATE}"]["median_layer_wall_ms"]
    warm_control = summaries[f"warm/{CONTROL}"]["median_layer_wall_ms"]
    warm_candidate = summaries[f"warm/{CANDIDATE}"]["median_layer_wall_ms"]
    paired_cold_regressions = [
        repetition
        for repetition in range(3)
        if grouped[("cold", CANDIDATE)][repetition]["layer_wall_ms"]
        > grouped[("cold", CONTROL)][repetition]["layer_wall_ms"]
    ]
    cold_speedup = cold_control / cold_candidate
    warm_speedup = warm_control / warm_candidate
    performance_gate_passed = (
        cold_speedup >= 2.0 and not paired_cold_regressions and warm_speedup >= 1.0
    )
    warm_zero_reads = all(
        trial["activity"]["disk_bytes_read"] == 0 and trial["activity"]["pageins"] == 0
        for trial in grouped[("warm", CANDIDATE)]
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
        not warm_zero_reads
        or safety_summary["minimum_system_memory_free_percent"] < 20
        or safety_summary["maximum_swap_growth_bytes"] != 0
        or safety_summary["maximum_new_throttled_pages"] != 0
        or not safety_summary["protected_services_stable"]
    ):
        raise ValueError("PW-0107 cache-state or safety gate failed")
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
        "evidence_class": "pw0107_two_barrier_routed_layer_transaction_analysis",
        "raw_report_sha256": RAW_SHA256,
        "commit": report["commit"],
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "artifact_manifest_sha256": report["artifact_manifest_sha256"],
        "artifact_sha256": report["artifact_sha256"],
        "trial_summaries": summaries,
        "cold_speedup_candidate_vs_control": cold_speedup,
        "warm_speedup_candidate_vs_control": warm_speedup,
        "paired_cold_candidate_regression_repetitions": paired_cold_regressions,
        "performance_gate_passed": performance_gate_passed,
        "warm_candidate_zero_physical_reads_and_pageins": warm_zero_reads,
        "identical_expert_diagnostics_all_trials": True,
        "identical_routed_and_final_hashes_all_trials": True,
        "routed_sha256": next(iter(routed_hashes)),
        "final_residual_sha256": next(iter(final_hashes)),
        "safety": safety_summary,
        "decision": "reject_command_aggregation_promote_metal_io_compute_overlap",
        "limitations": (
            "real layer-4 one-row component only; unchanged rejected L3 arithmetic; "
            "cold invalidation produced variable physical-read counts; no endpoint TPS"
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
