#!/usr/bin/env python3
"""Validate and decide the immutable PW-0114 paired distribution probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

try:
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from openrouter_reference import atomic_write_new, canonical_json


CONTROL_SHA256 = "24622adf564d840880ea44163edbb3c98905d4914b2bc4153cb21151fc58281e"
CANDIDATE_SHA256 = "16aaaded5cb082e5672f1a18b132fa52665375117dadb07c0c628fcc76b3b43f"
ORACLE_SHA256 = "75b4a5799bcc7dc898643c266d42a00b52c75be0f1fe1682ef253ce8fe4287a8"
IMPLEMENTATION_COMMIT = "a25236b7d5c0a65200934954665024b40b7890fb"
CONTRACT_COMMIT = "4238b4d523ebcc36e0760f971b2f84664eeffe56"
REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CHECKPOINT_VERIFICATION_SHA256 = (
    "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
)
VOCAB_SIZE = 152_576


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path, expected_hash: str, semantic: str) -> dict:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"PW-0114 {semantic} report hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("semantic") != semantic
        or report.get("revision") != REVISION
        or report.get("commit") != IMPLEMENTATION_COMMIT
        or report.get("checkpoint_verification_sha256")
        != CHECKPOINT_VERIFICATION_SHA256
        or report.get("oracle_manifest_sha256") != ORACLE_SHA256
        or report.get("status") != "diagnostic_complete_not_accepted"
        or report.get("diagnostic_only") is not True
        or report.get("output_committed") is not False
        or report.get("accepted_tokens_in_timed_interval") != 0
        or report.get("A") != 0
        or report.get("promotion_gates_passed") is not False
        or report.get("performance_claim") is not None
        or report.get("generated_token_ids") != [264, 13]
        or len(report.get("layer_parity", [])) != 48
        or len(report.get("steps", [])) != 2
    ):
        raise ValueError(f"PW-0114 {semantic} authority mismatch")
    ledger = report.get("metal_ledger", {})
    if (
        ledger.get("expert_executions") != 376
        or ledger.get("projection_dispatches") != 1_128
        or ledger.get("released_projection_buffers") != 1_128
        or ledger.get("installed_source_bytes") != 9_464_659_968
    ):
        raise ValueError(f"PW-0114 {semantic} causal ledger mismatch")
    return report


def read_oracle_logits(manifest_path: Path) -> list[float]:
    if sha256_file(manifest_path) != ORACLE_SHA256:
        raise ValueError("PW-0114 oracle manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get("captures", {}).get("incremental_last_logits", {})
    if (
        manifest.get("semantic") != "mimo_pytorch_incremental_cache_oracle"
        or manifest.get("revision") != REVISION
        or capture.get("shape") != [VOCAB_SIZE]
        or capture.get("dtype") != "F32"
        or Path(capture.get("file", "")).name != capture.get("file")
    ):
        raise ValueError("PW-0114 oracle logit authority mismatch")
    path = manifest_path.with_name(capture["file"])
    payload = path.read_bytes()
    if len(payload) != VOCAB_SIZE * 4 or hashlib.sha256(payload).hexdigest() != capture.get(
        "sha256"
    ):
        raise ValueError("PW-0114 oracle logit payload mismatch")
    return list(struct.unpack(f"<{VOCAB_SIZE}f", payload))


def top_indices(values: list[float], count: int) -> list[int]:
    if len(values) != VOCAB_SIZE or any(not math.isfinite(value) for value in values):
        raise ValueError("PW-0114 logit vector is invalid")
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def log_probability(values: list[float], token: int) -> float:
    maximum = max(values)
    normalizer = sum(math.exp(value - maximum) for value in values)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("PW-0114 logit normalization failed")
    return values[token] - maximum - math.log(normalizer)


def distribution_metrics(reference: list[float], candidate: list[float]) -> dict:
    source_top = top_indices(reference, 20)
    candidate_top = top_indices(candidate, 20)
    source_maximum = max(reference)
    candidate_maximum = max(candidate)
    source_normalizer = sum(math.exp(value - source_maximum) for value in reference)
    candidate_normalizer = sum(math.exp(value - candidate_maximum) for value in candidate)
    source_projection = [
        math.exp(reference[token] - source_maximum) / source_normalizer
        for token in source_top
    ]
    candidate_projection = [
        math.exp(candidate[token] - candidate_maximum) / candidate_normalizer
        for token in source_top
    ]
    source_projection.append(max(0.0, 1.0 - sum(source_projection)))
    candidate_projection.append(max(0.0, 1.0 - sum(candidate_projection)))
    jsd = 0.0
    for source, local in zip(source_projection, candidate_projection, strict=True):
        midpoint = 0.5 * (source + local)
        for value in (source, local):
            if value:
                jsd += 0.5 * value * math.log(value / midpoint)
    source_argmax = source_top[0]
    return {
        "source_argmax_token_id": source_argmax,
        "candidate_argmax_token_id": candidate_top[0],
        "source_chosen_token_absolute_logprob_error_nats": abs(
            log_probability(reference, source_argmax)
            - log_probability(candidate, source_argmax)
        ),
        "source_top20_candidate_overlap": len(set(source_top) & set(candidate_top)),
        "top20_token_identity": source_top == candidate_top,
        "projected_top20_jsd_nats": jsd,
    }


def safety_summary(report: dict) -> dict:
    snapshots = report.get("safety_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("PW-0114 safety ledger is missing")
    services = snapshots[0]["protected_service_pids"]
    summary = {
        "minimum_system_memory_free_percent": min(
            snapshot["system_memory_free_percent"] for snapshot in snapshots
        ),
        "maximum_peak_resident_bytes": max(
            snapshot["process_peak_resident_bytes"] for snapshot in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            snapshot["process_physical_footprint_bytes"] for snapshot in snapshots
        ),
        "post_release_physical_footprint_bytes": snapshots[-1][
            "process_physical_footprint_bytes"
        ],
        "maximum_swap_growth_bytes": max(
            snapshot["swap_growth_bytes"] for snapshot in snapshots
        ),
        "maximum_new_throttled_pages": max(
            snapshot["new_throttled_pages"] for snapshot in snapshots
        ),
        "protected_services_stable": all(
            snapshot["protected_service_pids"] == services for snapshot in snapshots
        ),
    }
    if (
        summary["minimum_system_memory_free_percent"] < 20
        or summary["maximum_peak_resident_bytes"] > 8 * 1024**3
        or summary["maximum_process_physical_footprint_bytes"] > 8 * 1024**3
        or summary["post_release_physical_footprint_bytes"] > 4 * 1024**3
        or summary["maximum_swap_growth_bytes"] > 512 * 1024**2
        or summary["maximum_new_throttled_pages"] != 0
        or not summary["protected_services_stable"]
    ):
        raise ValueError("PW-0114 safety gate failed")
    return summary


def layer_summary(report: dict) -> dict:
    rows = report["layer_parity"]
    return {
        "first_failed_layer": next(
            (row["layer"] for row in rows if not row["final_state"]["passed"]), None
        ),
        "source_route_mismatch_layers": [
            row["layer"] for row in rows if not row["selected_experts_exact"]
        ],
        "maximum_source_route_weight_absolute_error": max(
            row["maximum_route_weight_absolute_error"] for row in rows
        ),
        "worst_layer_final_relative_l2": max(
            row["final_state"]["relative_l2"] for row in rows
        ),
        "final_layer_relative_l2": rows[-1]["final_state"]["relative_l2"],
    }


def analyze(control_path: Path, candidate_path: Path, oracle_path: Path) -> dict:
    control = load_report(
        control_path, CONTROL_SHA256, "mimo_v2_5_pw0114_sparse_repaired_distribution_control"
    )
    candidate = load_report(
        candidate_path,
        CANDIDATE_SHA256,
        "mimo_v2_5_pw0114_repair_free_metal_native_l3_distribution_probe",
    )
    if (
        control.get("repair_mode") != "value_derived_sparse_repair"
        or control["metal_ledger"].get("sparse_repair_counts") != [129, 170, 250]
        or control["metal_ledger"].get("sparse_decoded_weight_bytes") != 1_736_704
        or candidate.get("repair_mode") != "disabled"
        or candidate["metal_ledger"].get("sparse_repair_counts") != [0, 0, 0]
        or candidate["metal_ledger"].get("sparse_decoded_weight_bytes") != 0
    ):
        raise ValueError("PW-0114 repair attribution mismatch")
    oracle_logits = read_oracle_logits(oracle_path)
    metrics = {}
    for name, report in (("control", control), ("candidate", candidate)):
        logits = report["steps"][1].get("full_logits")
        observed = distribution_metrics(oracle_logits, logits)
        for field, value in observed.items():
            recorded = report[field]
            if isinstance(value, float):
                if not math.isclose(value, recorded, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"PW-0114 {name} {field} does not recompute")
            elif value != recorded:
                raise ValueError(f"PW-0114 {name} {field} does not recompute")
        observed["gate_passed"] = (
            observed["source_argmax_token_id"] == observed["candidate_argmax_token_id"]
            and observed["source_chosen_token_absolute_logprob_error_nats"] <= 0.08
            and observed["source_top20_candidate_overlap"] >= 18
            and observed["projected_top20_jsd_nats"] <= 0.01
        )
        if observed["gate_passed"] is not report.get("distribution_probe_passed"):
            raise ValueError(f"PW-0114 {name} gate result mismatch")
        metrics[name] = observed
    control_routes = [
        row["selected_experts_by_position"] for row in control["steps"][1]["layer_traces"]
    ]
    candidate_routes = [
        row["selected_experts_by_position"] for row in candidate["steps"][1]["layer_traces"]
    ]
    control_weights = [
        row["route_weights_by_position"] for row in control["steps"][1]["layer_traces"]
    ]
    candidate_weights = [
        row["route_weights_by_position"] for row in candidate["steps"][1]["layer_traces"]
    ]
    candidate_vs_control = {
        "selected_expert_mismatch_layers": [
            layer
            for layer, (left, right) in enumerate(zip(control_routes, candidate_routes, strict=True))
            if left != right
        ],
        "route_weight_mismatch_layers": [
            layer
            for layer, (left, right) in enumerate(zip(control_weights, candidate_weights, strict=True))
            if left != right
        ],
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
        "evidence_class": "pw0114_repair_free_metal_native_distribution_analysis",
        "contract_commit": CONTRACT_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "control_report_sha256": CONTROL_SHA256,
        "candidate_report_sha256": CANDIDATE_SHA256,
        "oracle_manifest_sha256": ORACLE_SHA256,
        "distribution": metrics,
        "layer_behavior": {
            "control": layer_summary(control),
            "candidate": layer_summary(candidate),
            "candidate_vs_control": candidate_vs_control,
        },
        "repair_attribution": {
            "control_counts": control["metal_ledger"]["sparse_repair_counts"],
            "control_decoded_weight_bytes": control["metal_ledger"][
                "sparse_decoded_weight_bytes"
            ],
            "candidate_counts": candidate["metal_ledger"]["sparse_repair_counts"],
            "candidate_decoded_weight_bytes": candidate["metal_ledger"][
                "sparse_decoded_weight_bytes"
            ],
        },
        "timing_diagnostic_only": {
            "control_prefill_wall_ms": control["prefill_wall_ms"],
            "control_incremental_wall_ms": control["incremental_wall_ms"],
            "candidate_prefill_wall_ms": candidate["prefill_wall_ms"],
            "candidate_incremental_wall_ms": candidate["incremental_wall_ms"],
        },
        "safety": {
            "control": safety_summary(control),
            "candidate": safety_summary(candidate),
        },
        "candidate_numerical_continuation_gate_passed": metrics["candidate"][
            "gate_passed"
        ],
        "decision": "retain_repair_free_metal_native_l3_numerical_premise_conditionally",
        "limitations": (
            "one frozen source-derived text position; no hosted comparison; source routes diverge "
            "after accumulated hidden drift; candidate and repaired control select different experts "
            "at three late layers; no accepted token or TPS; projection-at-a-time physical vehicle remains rejected"
        ),
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.control, arguments.candidate, arguments.oracle)
        atomic_write_new(arguments.output, canonical_json(result))
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
