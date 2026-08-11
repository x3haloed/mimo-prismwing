#!/usr/bin/env python3
"""Authenticate PW-0213's cacheable/uncached/cacheable verifier pilot."""

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


COMMIT = "f096b8e1a5aefa9515b69ce209c11fe96c22ae2d"
REPORTS = {
    "control_1": "74b8d15186aeff9af97393643fe11bf39c11e0d0d2d2da7777493723b538ec0d",
    "candidate": "f05c8b8be0794aa1aa1a2e790a96ac3c1a967439e4c0fda1ad8fb6a4eb37eee5",
    "control_2": "be004a835f59c99eb2539ae54dd434ce05748840eb48caffc1dba8e83c88baca",
}
TOKENS = [30092, 4145, 5610, 678, 7987, 315, 279, 19745]


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("PW-0213 verifier median requires values")
    return float(statistics.median(values))


def _safety(report: dict) -> dict:
    snapshots = report["safety_snapshots"]
    baseline = snapshots[0]["protected_service_pids"]
    summary = {
        "minimum_system_memory_free_percent": min(row["system_memory_free_percent"] for row in snapshots),
        "maximum_process_peak_resident_bytes": max(row["process_peak_resident_bytes"] for row in snapshots),
        "maximum_process_physical_footprint_bytes": max(row["process_physical_footprint_bytes"] for row in snapshots),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "protected_service_pid_sets_stable": all(row["protected_service_pids"] == baseline for row in snapshots),
    }
    if (
        summary["minimum_system_memory_free_percent"] < 10
        or summary["maximum_process_peak_resident_bytes"] > 8 * 1024**3
        or summary["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or summary["maximum_swap_growth_bytes"] != 0
        or summary["maximum_new_throttled_pages"] != 0
        or not summary["protected_service_pid_sets_stable"]
        or report["residency"]["final_resident_bytes"] != 0
    ):
        raise ValueError("PW-0213 verifier Gate 8 failed")
    return summary


def _load(path: Path, expected_sha256: str, candidate: bool) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError("PW-0213 verifier report hash mismatch")
    report = json.loads(path.read_text())
    transaction = report.get("transactions", [None])[0]
    residency = report.get("residency", {})
    if (
        report.get("commit") != COMMIT
        or report.get("git_dirty") is not False
        or report.get("generated_token_ids") != TOKENS
        or report.get("accepted_tokens") != 8
        or report.get("requested_output_tokens") != 8
        or report.get("batch_size") != 1
        or report.get("concurrency") != 1
        or report.get("verifier_width") != 8
        or not report.get("route_trace_captured")
        or not isinstance(transaction, dict)
        or transaction.get("verifier_authorized_token_ids") != TOKENS[1:]
        or transaction.get("retained_proposal_rows") != 7
        or transaction.get("proposal_converged") is not True
        or residency.get("installed_identities") != ["expert:14:162"]
        or residency.get("installed_resident_bytes") != 25_182_208
        or residency.get("resident_source_bytes") != 201_375_744
        or residency.get("install_logical_source_bytes") != 25_171_968
        or residency.get("install_pread_calls") != 6
    ):
        raise ValueError("PW-0213 verifier authority mismatch")
    if candidate:
        if (
            report.get("schema_version") != 8
            or report.get("evidence_class") != "pw0213_uncached_two_buffer_single_object_verifier_pilot"
            or residency.get("source_transport") != "page_aligned_f_nocache_f_rdahead_zero_two_buffer_pread"
            or residency.get("install_widened_read_bytes") != 25_264_128
            or residency["install_widened_read_bytes"] / residency["install_logical_source_bytes"] > 1.05
        ):
            raise ValueError("PW-0213 uncached candidate identity mismatch")
    else:
        if (
            report.get("schema_version") != 4
            or report.get("evidence_class") != "pw0207_single_object_pressure_resident_route_trace"
            or residency.get("source_transport") != "cacheable_exact_pread"
            or residency.get("install_widened_read_bytes") != 25_171_968
        ):
            raise ValueError("PW-0213 cacheable control identity mismatch")
    return report


def analyze(control_1_path: Path, candidate_path: Path, control_2_path: Path) -> dict:
    control_1 = _load(control_1_path, REPORTS["control_1"], False)
    candidate = _load(candidate_path, REPORTS["candidate"], True)
    control_2 = _load(control_2_path, REPORTS["control_2"], False)
    controls = [control_1, control_2]

    control_metrics = {
        "complete_wall_ms": median([row["complete_wall_ms"] for row in controls]),
        "prefill_wall_ms": median([row["prefill_wall_ms"] for row in controls]),
        "post_prefill_wall_ms": median([row["complete_wall_ms"] - row["prefill_wall_ms"] for row in controls]),
        "proposal_wall_ms": median([row["proposal_wall_ms"] for row in controls]),
        "verification_wall_ms": median([row["verification_wall_ms"] for row in controls]),
        "install_transfer_wall_ms": median([row["residency"]["install_transfer_wall_ms"] for row in controls]),
    }
    candidate_metrics = {
        "complete_wall_ms": candidate["complete_wall_ms"],
        "prefill_wall_ms": candidate["prefill_wall_ms"],
        "post_prefill_wall_ms": candidate["complete_wall_ms"] - candidate["prefill_wall_ms"],
        "proposal_wall_ms": candidate["proposal_wall_ms"],
        "verification_wall_ms": candidate["verification_wall_ms"],
        "install_transfer_wall_ms": candidate["residency"]["install_transfer_wall_ms"],
    }
    comparisons = {
        key.replace("_wall_ms", "_wall_reduction"): 1.0 - candidate_metrics[key] / control_metrics[key]
        for key in control_metrics
    }
    comparisons["complete_accepted_tps_gain"] = control_metrics["complete_wall_ms"] / candidate_metrics["complete_wall_ms"] - 1.0
    comparisons["post_prefill_accepted_tps_gain"] = control_metrics["post_prefill_wall_ms"] / candidate_metrics["post_prefill_wall_ms"] - 1.0
    runtime_promotion_passed = comparisons["post_prefill_accepted_tps_gain"] > 0.0
    if runtime_promotion_passed:
        raise ValueError("PW-0213 verifier unexpectedly passes causal promotion")

    return {
        "schema_version": 1,
        "evidence_class": "pw0213_validated_uncached_single_object_verifier_rejection",
        "implementation_commit": COMMIT,
        "report_sha256": REPORTS,
        "control_observations": [
            {
                "complete_wall_ms": row["complete_wall_ms"],
                "prefill_wall_ms": row["prefill_wall_ms"],
                "post_prefill_wall_ms": row["complete_wall_ms"] - row["prefill_wall_ms"],
            }
            for row in controls
        ],
        "control_median": control_metrics,
        "candidate": candidate_metrics,
        "comparisons": comparisons,
        "exact_output_and_acceptance_match": True,
        "safety": {
            "control_1": _safety(control_1),
            "candidate": _safety(candidate),
            "control_2": _safety(control_2),
        },
        "runtime_promotion_passed": False,
        "experiment_complete": True,
        "decision": "reject_uncached_runtime_transport; preserve_isolated_file_backed_and_install_gains",
        "limitations": "one ordinary q8 one-object verifier pilot on Apple M1; complete-wall apparent gain is prefill-only and precedes the mechanism; no general endpoint TPS",
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-1", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--control-2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.control_1, arguments.candidate, arguments.control_2)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
