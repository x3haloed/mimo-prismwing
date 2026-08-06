#!/usr/bin/env python3
"""Validate and summarize the immutable PW-0106 interleaved layer report."""

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


RAW_SHA256 = "fb0a1cf0e9dba0d3941a5d9786e4867fe04ea21dcd81469d986928fdaada9232"
VARIANTS = (
    "C0_copied_global_release",
    "C1_artifact_copied",
    "C2_artifact_no_copy",
)
STATES = ("cold", "warm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("PW-0106 distributions require exactly three trials")
    return statistics.median(values)


def stage_totals(trial: dict) -> dict[str, float]:
    experts = trial["expert_tomography"]
    projections = [projection for expert in experts for projection in expert["projections"]]
    return {
        "tensor_lookup_validation_ms": sum(
            expert["tensor_lookup_validation_ms"] for expert in experts
        ),
        "expert_release_ms": sum(
            expert["matrix_transient_release_ms"] for expert in experts
        ),
        "source_buffer_install_ms": sum(
            projection["source_buffer_install_ms"] for projection in projections
        ),
        "synchronous_wait_ms": sum(
            projection["synchronous_wait_ms"] for projection in projections
        ),
        "gpu_interval_ms": sum(
            projection["gpu_interval_ms"] for projection in projections
        ),
        "dynamic_fp8_ms": sum(
            expert["dynamic_input_ms"] + expert["dynamic_hidden_ms"]
            for expert in experts
        ),
        "sparse_repair_ms": sum(
            expert["gate_up_sparse_repair_ms"] + expert["down_sparse_repair_ms"]
            for expert in experts
        ),
        "swiglu_ms": sum(expert["swiglu_ms"] for expert in experts),
        "trusted_tensor_bind_ms": trial["trusted_tensor_bind_ms"],
        "final_release_ms": trial["final_release_ms"],
    }


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0106 raw report hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("semantic")
        != "mimo_v2_5_layer4_page_stable_copied_no_copy_benchmark"
        or report.get("commit") != "ec1d2f2b42532a3e870218d0b47021a8525a45b6"
        or report.get("artifact_manifest_sha256")
        != "40179385a571a19b135a4740122744ae3d8ea2c97ef265ac20968296e98822b8"
        or report.get("artifact_sha256")
        != "fac61c2cfad4b00248c96a52b68360fecd39e2c912e6ffd6643e3f06ade00d21"
        or report.get("selected_experts") != [232, 31, 64, 96, 9, 88, 245, 130]
        or report.get("maximum_route_weight_absolute_error") != 0.0
        or report.get("no_copy_probe_passed") is not True
        or report.get("accepted_tokens") != 0
        or report.get("A") != 0
        or report.get("U") != 8
    ):
        raise ValueError("PW-0106 report authority mismatch")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != 18:
        raise ValueError("PW-0106 requires exactly 18 trials")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for state in STATES:
        for variant in VARIANTS:
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
        or len(trial.get("expert_tomography", [])) != 8
        or len(trial.get("expert_diagnostics", [])) != 8
        or any(len(expert.get("projections", [])) != 3 for expert in trial["expert_tomography"])
        for trial in trials
    ):
        raise ValueError("PW-0106 trial accounting mismatch")
    routed_hashes = {trial["routed_sha256"] for trial in trials}
    final_hashes = {trial["final_residual_sha256"] for trial in trials}
    diagnostic_payloads = {
        json.dumps(trial["expert_diagnostics"], sort_keys=True, separators=(",", ":"))
        for trial in trials
    }
    if len(routed_hashes) != 1 or len(final_hashes) != 1 or len(diagnostic_payloads) != 1:
        raise ValueError("PW-0106 variants do not reproduce identical outputs")

    summaries = {}
    for state in STATES:
        for variant in VARIANTS:
            records = grouped[(state, variant)]
            totals = [stage_totals(trial) for trial in records]
            summaries[f"{state}/{variant}"] = {
                "layer_wall_ms": [trial["layer_wall_ms"] for trial in records],
                "median_layer_wall_ms": median(
                    [trial["layer_wall_ms"] for trial in records]
                ),
                "disk_bytes_read": [trial["activity"]["disk_bytes_read"] for trial in records],
                "pageins": [trial["activity"]["pageins"] for trial in records],
                "minor_fault_deltas": [
                    trial["activity"]["minor_faults"] for trial in records
                ],
                "median_stage_totals_ms": {
                    field: median([record[field] for record in totals])
                    for field in totals[0]
                },
            }
    cold_control = summaries["cold/C0_copied_global_release"]["median_layer_wall_ms"]
    cold_copied = summaries["cold/C1_artifact_copied"]["median_layer_wall_ms"]
    cold_no_copy = summaries["cold/C2_artifact_no_copy"]["median_layer_wall_ms"]
    c1_speedup = cold_control / cold_copied
    c2_speedup = cold_control / cold_no_copy
    no_copy_gain = cold_copied / cold_no_copy
    no_candidate_regression = all(
        trial["layer_wall_ms"] < min(
            control["layer_wall_ms"]
            for control in grouped[("cold", "C0_copied_global_release")]
        )
        for variant in VARIANTS[1:]
        for trial in grouped[("cold", variant)]
    )
    warm_candidates_have_zero_reads = all(
        trial["activity"]["disk_bytes_read"] == 0
        and trial["activity"]["pageins"] == 0
        for variant in VARIANTS[1:]
        for trial in grouped[("warm", variant)]
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
        "post_release_physical_footprint_bytes": safety[-1][
            "process_physical_footprint_bytes"
        ],
        "maximum_swap_growth_bytes": max(snapshot["swap_growth_bytes"] for snapshot in safety),
        "maximum_new_throttled_pages": max(
            snapshot["new_throttled_pages"] for snapshot in safety
        ),
        "protected_services_stable": all(
            snapshot["protected_service_pids"] == baseline_services for snapshot in safety
        ),
    }
    if (
        c1_speedup < 2.0
        or c2_speedup < 2.0
        or not no_candidate_regression
        or not warm_candidates_have_zero_reads
        or safety_summary["minimum_system_memory_free_percent"] < 20
        or safety_summary["maximum_swap_growth_bytes"] != 0
        or safety_summary["maximum_new_throttled_pages"] != 0
        or not safety_summary["protected_services_stable"]
    ):
        raise ValueError("PW-0106 promotion or safety gate failed")
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
        "evidence_class": "pw0106_page_stable_metal_ready_routed_layer_analysis",
        "raw_report_sha256": RAW_SHA256,
        "commit": report["commit"],
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "artifact_manifest_sha256": report["artifact_manifest_sha256"],
        "artifact_sha256": report["artifact_sha256"],
        "no_copy_probe_ms": report["no_copy_probe_ms"],
        "warm_prefault_ms": report["warm_prefault_ms"],
        "trial_summaries": summaries,
        "cold_speedup_c1_vs_c0": c1_speedup,
        "cold_speedup_c2_vs_c0": c2_speedup,
        "cold_speedup_c2_vs_c1": no_copy_gain,
        "no_cold_candidate_regression": no_candidate_regression,
        "warm_candidates_have_zero_physical_reads_and_pageins": warm_candidates_have_zero_reads,
        "identical_expert_diagnostics_all_trials": len(diagnostic_payloads) == 1,
        "identical_routed_and_final_hashes_all_trials": True,
        "routed_sha256": next(iter(routed_hashes)),
        "final_residual_sha256": next(iter(final_hashes)),
        "safety": safety_summary,
        "decision": "promote_page_stable_no_copy_to_routed_layer_transaction",
        "limitations": (
            "real layer-4 one-row component only; C0 necessarily defeats warm cache via its "
            "per-expert global release; unchanged rejected L3 arithmetic; no endpoint TPS"
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
