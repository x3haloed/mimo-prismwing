#!/usr/bin/env python3
"""Validate and summarize the immutable PW-0105 tomography report."""

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


RAW_SHA256 = "49c1f85b24e8864d43a3a901de9c7c40e8745a4427599248bd937abba4ce3e11"
PW0100_INCREMENTAL_WALL_MS = 75_725.919


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty timing distribution")
    ordered = sorted(values)
    percentile = lambda fraction: ordered[round(fraction * (len(ordered) - 1))]
    return {
        "count": len(values),
        "sum_ms": sum(values),
        "p10_ms": percentile(0.10),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(0.90),
        "maximum_ms": ordered[-1],
    }


def analyze(path: Path) -> dict:
    if sha256_file(path) != RAW_SHA256:
        raise ValueError("PW-0105 raw report hash mismatch")
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("semantic")
        != "mimo_v2_5_bounded_metal_incremental_weight_install_tomography"
        or report.get("commit") != "e67fe4b3927bf027b5fa91f176435989576715e8"
        or report.get("generated_token_ids") != [264, 13]
        or report.get("promotion_gates_passed") is not False
        or report.get("status") != "diagnostic_complete_candidate_gates_failed"
    ):
        raise ValueError("PW-0105 report authority mismatch")
    ledger = report.get("metal_ledger", {})
    experts = ledger.get("expert_tomography")
    layers = ledger.get("layer_tomography")
    if not isinstance(experts, list) or len(experts) != 376:
        raise ValueError("PW-0105 must contain 376 expert records")
    if not isinstance(layers, list) or len(layers) != 47:
        raise ValueError("PW-0105 must contain 47 routed-layer records")
    projections = [projection for expert in experts for projection in expert["projections"]]
    if len(projections) != 1128:
        raise ValueError("PW-0105 must contain 1,128 projection records")
    if any(
        [projection["projection"] for projection in expert["projections"]]
        != ["gate", "up", "down"]
        or len(expert["source_shards"]) != 3
        for expert in experts
    ):
        raise ValueError("PW-0105 projection identity mismatch")
    if sorted(layer["layer"] for layer in layers) != list(range(1, 48)):
        raise ValueError("PW-0105 routed-layer identity mismatch")
    if any(layer["expert_count"] != 8 for layer in layers):
        raise ValueError("PW-0105 routed layer must contain eight experts")
    if any(projection["gpu_interval_ms"] is None for projection in projections):
        raise ValueError("Apple M1 GPU timestamps unexpectedly unavailable")

    projection_fields = [
        "source_buffer_install_ms",
        "small_buffer_install_ms",
        "command_encode_ms",
        "commit_call_ms",
        "synchronous_wait_ms",
        "gpu_interval_ms",
        "readback_ms",
        "explicit_release_ms",
        "wall_ms",
    ]
    expert_fields = [
        "tensor_lookup_validation_ms",
        "dynamic_input_ms",
        "gate_up_bf16_round_ms",
        "gate_up_sparse_repair_ms",
        "swiglu_ms",
        "dynamic_hidden_ms",
        "down_bf16_round_ms",
        "down_sparse_repair_ms",
        "weighted_scatter_ms",
        "matrix_transient_release_ms",
        "wall_ms",
    ]
    projection_distributions = {
        field: distribution([record[field] for record in projections])
        for field in projection_fields
    }
    expert_distributions = {
        field: distribution([record[field] for record in experts]) for field in expert_fields
    }

    projection_wall = projection_distributions["wall_ms"]["sum_ms"]
    projection_nonoverlap = sum(
        projection_distributions[field]["sum_ms"]
        for field in projection_fields
        if field not in ("wall_ms", "gpu_interval_ms")
    )
    projection_wrapper = projection_wall - projection_nonoverlap
    if projection_wrapper < -1.0:
        raise ValueError("projection non-overlapping timing ledger exceeds wall")

    expert_named = sum(
        expert_distributions[field]["sum_ms"]
        for field in expert_fields
        if field != "wall_ms"
    ) + projection_wall
    expert_wall = expert_distributions["wall_ms"]["sum_ms"]
    expert_wrapper = expert_wall - expert_named
    if expert_wrapper < -1.0:
        raise ValueError("expert non-overlapping timing ledger exceeds wall")

    routed_wall = sum(layer["routed_mlp_wall_ms"] for layer in layers)
    layer_expert_sum = sum(layer["expert_wall_sum_ms"] for layer in layers)
    if abs(layer_expert_sum - expert_wall) > 1e-6:
        raise ValueError("layer/expert timing ledger mismatch")
    route_schedule = sum(layer["route_and_schedule_ms"] for layer in layers)
    layer_round = sum(layer["layer_final_bf16_round_ms"] for layer in layers)
    routed_wrapper = routed_wall - layer_expert_sum - route_schedule - layer_round
    if routed_wrapper < -1.0:
        raise ValueError("routed-layer timing ledger exceeds wall")

    incremental_traces = report["steps"][1]["layer_traces"]
    if len(incremental_traces) != 48:
        raise ValueError("incremental report must contain 48 layer traces")
    all_layer_wall = sum(layer["wall_ms"] for layer in incremental_traces)
    incremental_wall = report["incremental_wall_ms"]
    non_moe_layer_wall = all_layer_wall - routed_wall
    outside_layer_wall = incremental_wall - all_layer_wall
    if min(non_moe_layer_wall, outside_layer_wall) < 0:
        raise ValueError("token timing ledger exceeds incremental wall")

    page_acquisition_and_validation = expert_distributions["tensor_lookup_validation_ms"][
        "sum_ms"
    ]
    page_invalidation_and_release = expert_distributions["matrix_transient_release_ms"][
        "sum_ms"
    ]
    source_copy = projection_distributions["source_buffer_install_ms"]["sum_ms"]
    wait = projection_distributions["synchronous_wait_ms"]["sum_ms"]
    gpu = projection_distributions["gpu_interval_ms"]["sum_ms"]
    removable_named = page_acquisition_and_validation + page_invalidation_and_release + source_copy + wait
    transaction_target_fraction = removable_named / routed_wall

    expert_disk = sum(record["activity"]["disk_bytes_read"] for record in experts)
    expert_pageins = sum(record["activity"]["pageins"] for record in experts)
    source_bytes = sum(
        projection["source_weight_bytes"] + projection["source_scale_bytes"]
        for projection in projections
    )
    if source_bytes != ledger["installed_source_bytes"]:
        raise ValueError("projection source-byte ledger mismatch")

    safety = report["safety_snapshots"]
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
        "evidence_class": "pw0105_weight_install_tomography_analysis",
        "raw_report_sha256": RAW_SHA256,
        "commit": report["commit"],
        "analysis_commit": analysis_commit,
        "analysis_dirty": analysis_dirty,
        "incremental_wall_ms": incremental_wall,
        "pw0100_incremental_wall_ms": PW0100_INCREMENTAL_WALL_MS,
        "instrumented_wall_change_fraction": incremental_wall / PW0100_INCREMENTAL_WALL_MS - 1,
        "experts": len(experts),
        "projections": len(projections),
        "routed_layers": len(layers),
        "projection_distributions": projection_distributions,
        "expert_distributions": expert_distributions,
        "projection_wrapper_ms": projection_wrapper,
        "expert_wrapper_ms": expert_wrapper,
        "routed_wrapper_ms": routed_wrapper,
        "token_partition_ms": {
            "routed_moe": routed_wall,
            "non_moe_inside_layers": non_moe_layer_wall,
            "outside_layers": outside_layer_wall,
        },
        "routed_primary_categories_ms": {
            "tensor_lookup_validation_page_acquisition": page_acquisition_and_validation,
            "checkpoint_page_invalidation_release": page_invalidation_and_release,
            "source_buffer_copy": source_copy,
            "synchronous_wait_including_gpu": wait,
            "gpu_interval_subset_of_wait": gpu,
        },
        "transaction_target_fraction_of_routed_wall": transaction_target_fraction,
        "gpu_interval_fraction_of_routed_wall": gpu / routed_wall,
        "expert_activity_disk_bytes_read": expert_disk,
        "expert_activity_pageins": expert_pageins,
        "installed_source_bytes": source_bytes,
        "complete_process_disk_bytes_read": report["ledger"]["actual_process_disk_bytes_read"],
        "safety": safety_summary,
        "decision": "promote_cold_layer_transaction_artifact_no_copy_async_branch",
        "limitations": "one rejected L3 incremental text token after a 27-token prefill; no endpoint TPS or correctness promotion",
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
