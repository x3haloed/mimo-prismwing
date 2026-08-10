#!/usr/bin/env python3
"""Authenticate and adjudicate PW-0157's tied-top-k route-coverage walks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

try:
    from tools.analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
RUNTIME_COMMIT = "f677893a12fc5631ddcdecf8fc407b7d1178c3f5"
TOPK_FIXTURE_SHA256 = "5c232ccc5823aeff1e91e0e674c78d635ed8ef4ffbaa4fef2135d76d58bc2243"
CORPUS_FIXTURE_SHA256 = "3b5bc4e8f41fed2a13867bc96ea8236d1630bf994eee5608a8366f1f846a79d5"
VERIFICATION_SHA256 = "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
PREFIXES = (512, 1024, 2048, 4096, 8000)
FROZEN_PREFIX_HASHES = {
    512: "32fa8954e875e6c8c53b5092827820940f51225d2bf24322caf5b782295004b9",
    1024: "f5e0e67a06ffd3867ecec84c38311ef9f0f409a63d7b12ebe26d5fa1fca61004",
    2048: "ce0e07bdcf0d5f2aacc366224ab402b7666df249e1e738eb3fda0f32ed9dba2a",
    4096: "658d2635e8aee4e97ce5a10d7eb1ac347b722f251b663c163814707d3d3f77cc",
}
GLOBAL_LAYERS = (0, 5, 11, 17, 23, 29, 35, 41, 47)
SOURCE_EXPERT_BYTES = 25_171_968
FREE_HBM_SLOTS = 660
MAXIMUM_STREAMABLE_RECORDS = 8_342
FIRST_DECISIVE_RECORD_COUNT = 9_003
GIB = 1024**3
MIB = 1024**2


def compact_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_topk_fixture(fixture: dict) -> None:
    if (
        fixture.get("schema_version") != 1
        or fixture.get("semantic") != "pinned_pytorch_cpu_unsorted_topk_tied_rows"
        or fixture.get("torch_version") != "2.13.0"
        or fixture.get("torch_commit") != "cf30153c4c131c8164ee7798e5022d810682e2cb"
        or fixture.get("topk_impl_sha256")
        != "1ff24ba878ccb3816511ba34609d7247225342c6aa61740b51917c8ca79407ab"
        or fixture.get("standard_library") != "libc++"
        or fixture.get("cpu_capability") != "DEFAULT"
        or fixture.get("width") != 256
        or fixture.get("top_k") != 8
    ):
        raise ValueError("pinned top-k fixture identity mismatch")
    cases = fixture.get("cases")
    expected_names = (
        "boundary_pair",
        "multiway_boundary",
        "all_equal",
        "repeated_plateaus",
        "signed_zero",
    )
    if not isinstance(cases, list) or tuple(row.get("name") for row in cases) != expected_names:
        raise ValueError("pinned top-k fixture case inventory mismatch")
    for row in cases:
        if (
            len(row.get("corrected_f32_u32", ())) != 256
            or len(row.get("selected_experts", ())) != 8
            or len(set(row["selected_experts"])) != 8
            or any(not 0 <= index < 256 for index in row["selected_experts"])
        ):
            raise ValueError("pinned top-k fixture case shape mismatch")


def validate_safety(snapshots: object) -> dict:
    if not isinstance(snapshots, list) or len(snapshots) < 4:
        raise ValueError("missing Gate-8 snapshots")
    phases = {row.get("phase") for row in snapshots}
    if not {"process_start", "checkpoint_released", "final_service_health"} <= phases:
        raise ValueError("missing Gate-8 release or service-health phase")
    baseline_services = {
        name for name, pids in snapshots[0].get("protected_service_pids", {}).items() if pids
    }
    for row in snapshots:
        current_services = {
            name for name, pids in row.get("protected_service_pids", {}).items() if pids
        }
        if (
            row.get("system_memory_free_percent", -1) < 20
            or row.get("process_physical_footprint_bytes", 1 << 63) > 8 * GIB
            or row.get("process_peak_resident_bytes", 1 << 63) > 8 * GIB
            or row.get("swap_growth_bytes", 1 << 63) > 512 * MIB
            or row.get("new_throttled_pages", 1) != 0
            or not baseline_services <= current_services
        ):
            raise ValueError("Gate-8 safety violation")
    release = next(row for row in snapshots if row["phase"] == "checkpoint_released")
    if release["process_physical_footprint_bytes"] > 4 * GIB:
        raise ValueError("Gate-8 post-release footprint violation")
    return {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(
            row["system_memory_free_percent"] for row in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            row["process_physical_footprint_bytes"] for row in snapshots
        ),
        "maximum_process_peak_resident_bytes": max(
            row["process_peak_resident_bytes"] for row in snapshots
        ),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in snapshots),
        "maximum_new_throttled_pages": max(row["new_throttled_pages"] for row in snapshots),
        "post_release_physical_footprint_bytes": release["process_physical_footprint_bytes"],
        "protected_services": sorted(baseline_services),
    }


def validate_route_rows(manifest: dict, positions: int) -> set[tuple[int, int]]:
    traces = manifest.get("layer_traces")
    if not isinstance(traces, list) or len(traces) != 48:
        raise ValueError("route trace layer inventory mismatch")
    observed: set[tuple[int, int]] = set()
    for layer, trace in enumerate(traces):
        expected_attention = "full" if layer in GLOBAL_LAYERS else "sliding_window_128"
        if (
            trace.get("layer") != layer
            or trace.get("attention") != expected_attention
            or trace.get("cache_length") != positions
            or not isinstance(trace.get("wall_ms"), (int, float))
            or not math.isfinite(trace["wall_ms"])
            or trace["wall_ms"] < 0.0
        ):
            raise ValueError("route trace layer identity mismatch")
        selected = trace.get("selected_experts_by_position")
        weights = trace.get("route_weights_by_position")
        if layer == 0:
            if selected != [] or weights != []:
                raise ValueError("dense layer zero unexpectedly has expert routes")
            continue
        if not isinstance(selected, list) or not isinstance(weights, list):
            raise ValueError("missing routed rows")
        if len(selected) != positions or len(weights) != positions:
            raise ValueError("routed row count mismatch")
        for experts, route_weights in zip(selected, weights, strict=True):
            if (
                len(experts) != 8
                or len(set(experts)) != 8
                or any(not isinstance(expert, int) or not 0 <= expert < 256 for expert in experts)
                or len(route_weights) != 8
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0
                    for value in route_weights
                )
                or abs(sum(route_weights) - 1.0) > 2.0e-6
            ):
                raise ValueError("invalid routed expert or weight row")
            observed.update((layer, expert) for expert in experts)
    return observed


def compare_route_prefix(
    previous: list[dict], current: list[dict], previous_positions: int
) -> dict:
    if len(previous) != len(current):
        raise ValueError("route-prefix layer inventory mismatch")
    routed_rows = 0
    exact_selected_order_rows = 0
    exact_selected_set_rows = 0
    exact_weight_rows = 0
    exact_expert_weight_mapping_rows = 0
    common_selected_experts = 0
    maximum_common_expert_weight_absolute_difference = 0.0
    first_selected_order_divergence = None
    first_weight_divergence = None
    for earlier, later in zip(previous, current, strict=True):
        if earlier["layer"] != later["layer"]:
            raise ValueError("route-prefix layer identity mismatch")
        layer = earlier["layer"]
        earlier_selected = earlier["selected_experts_by_position"]
        later_selected = later["selected_experts_by_position"][:previous_positions]
        earlier_weights = earlier["route_weights_by_position"]
        later_weights = later["route_weights_by_position"][:previous_positions]
        if layer == 0:
            continue
        if not (
            len(earlier_selected)
            == len(later_selected)
            == len(earlier_weights)
            == len(later_weights)
            == previous_positions
        ):
            raise ValueError("route-prefix row count mismatch")
        for position, (selected_a, selected_b, weights_a, weights_b) in enumerate(
            zip(earlier_selected, later_selected, earlier_weights, later_weights, strict=True)
        ):
            routed_rows += 1
            if selected_a == selected_b:
                exact_selected_order_rows += 1
            elif first_selected_order_divergence is None:
                first_selected_order_divergence = {"layer": layer, "position": position}
            if set(selected_a) == set(selected_b):
                exact_selected_set_rows += 1
            common_selected_experts += len(set(selected_a) & set(selected_b))
            if weights_a == weights_b:
                exact_weight_rows += 1
            elif first_weight_divergence is None:
                first_weight_divergence = {"layer": layer, "position": position}
            mapping_a = dict(zip(selected_a, weights_a, strict=True))
            mapping_b = dict(zip(selected_b, weights_b, strict=True))
            if mapping_a == mapping_b:
                exact_expert_weight_mapping_rows += 1
            for expert in mapping_a.keys() & mapping_b.keys():
                maximum_common_expert_weight_absolute_difference = max(
                    maximum_common_expert_weight_absolute_difference,
                    abs(mapping_a[expert] - mapping_b[expert]),
                )
    return {
        "compared_prefix_positions": previous_positions,
        "compared_routed_rows": routed_rows,
        "exact_selected_order_rows": exact_selected_order_rows,
        "exact_selected_order_fraction": exact_selected_order_rows / routed_rows,
        "exact_selected_set_rows": exact_selected_set_rows,
        "exact_selected_set_fraction": exact_selected_set_rows / routed_rows,
        "selected_expert_overlap_fraction": common_selected_experts / (routed_rows * 8),
        "exact_weight_rows": exact_weight_rows,
        "exact_weight_row_fraction": exact_weight_rows / routed_rows,
        "exact_expert_weight_mapping_rows": exact_expert_weight_mapping_rows,
        "exact_expert_weight_mapping_fraction": exact_expert_weight_mapping_rows / routed_rows,
        "maximum_common_expert_weight_absolute_difference": (
            maximum_common_expert_weight_absolute_difference
        ),
        "first_selected_order_divergence": first_selected_order_divergence,
        "first_weight_divergence": first_weight_divergence,
        "interpretation": "diagnostic_batch_shape_sensitivity_not_cross_run_identity_gate",
    }


def validate_manifest(manifest: dict, positions: int, corpus: dict) -> tuple[dict, list[dict]]:
    expected_input_sha256 = compact_sha256(corpus["expected_prompt_token_ids"][:positions])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("semantic") != "mimo_target_faithful_prefill_route_coverage_rust_trace"
        or manifest.get("revision") != REVISION
        or manifest.get("commit") != RUNTIME_COMMIT
        or manifest.get("fixture_sha256") != CORPUS_FIXTURE_SHA256
        or manifest.get("checkpoint_verification_sha256") != VERIFICATION_SHA256
        or manifest.get("corpus_positions") != 8000
        or manifest.get("traced_prefix_positions") != positions
        or manifest.get("input_token_ids_sha256") != expected_input_sha256
        or manifest.get("numerics")
        != "dynamic_fp8_e4m3fn_per_token_group_128_bf16_boundaries"
        or manifest.get("batch_size") != 1
        or manifest.get("concurrency") != 1
        or manifest.get("accepted_tokens") != 0
        or manifest.get("performance_claim") is not None
    ):
        raise ValueError(f"prefix-{positions} manifest identity mismatch")
    observed = validate_route_rows(manifest, positions)
    coverage = manifest.get("coverage", {})
    distinct = len(observed)
    streamed = max(0, distinct - FREE_HBM_SLOTS)
    expected_coverage = {
        "distinct_layer_expert_records": distinct,
        "source_expert_bytes_per_record": SOURCE_EXPERT_BYTES,
        "distinct_source_expert_bytes": distinct * SOURCE_EXPERT_BYTES,
        "granted_free_hbm_expert_slots": FREE_HBM_SLOTS,
        "minimum_streamed_records_after_offline_residency": streamed,
        "minimum_streamed_source_expert_bytes": streamed * SOURCE_EXPERT_BYTES,
        "granted_storage_lanes": 4,
        "granted_bytes_per_second_per_lane": 3_500_000_000,
        "ttft_limit_seconds": 15,
        "maximum_streamable_complete_records": MAXIMUM_STREAMABLE_RECORDS,
        "first_decisive_distinct_record_count": FIRST_DECISIVE_RECORD_COUNT,
        "exceeds_optimistic_15_second_storage_bound": distinct >= FIRST_DECISIVE_RECORD_COUNT,
    }
    if coverage != expected_coverage:
        raise ValueError(f"prefix-{positions} coverage ledger mismatch")
    ledger = manifest.get("ledger", {})
    if (
        ledger.get("routed_expert_executions") != distinct
        or not isinstance(ledger.get("pytorch_topk_boundary_tie_rows"), int)
        or ledger["pytorch_topk_boundary_tie_rows"] <= 0
        or ledger.get("peak_resident_bytes", 1 << 63) > 8 * GIB
        or not isinstance(manifest.get("complete_wall_ms"), (int, float))
        or not math.isfinite(manifest["complete_wall_ms"])
        or manifest["complete_wall_ms"] <= 0.0
    ):
        raise ValueError(f"prefix-{positions} execution ledger mismatch")
    safety = validate_safety(manifest.get("safety_snapshots"))
    summary = {
        "positions": positions,
        "input_token_ids_sha256": expected_input_sha256,
        "layer_routes_sha256": manifest.get("layer_routes_sha256"),
        "distinct_layer_expert_records": distinct,
        "minimum_streamed_records_after_offline_residency": streamed,
        "exceeds_optimistic_15_second_storage_bound": distinct >= FIRST_DECISIVE_RECORD_COUNT,
        "pytorch_topk_boundary_tie_rows": ledger["pytorch_topk_boundary_tie_rows"],
        "actual_process_disk_bytes_read": ledger.get("actual_process_disk_bytes_read"),
        "complete_wall_ms": manifest["complete_wall_ms"],
        "safety": safety,
    }
    return summary, manifest["layer_traces"]


def run(
    topk_fixture_path: Path,
    corpus_path: Path,
    verification_path: Path,
    manifest_paths: list[Path],
    output: Path,
    commit: str,
) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    if len(manifest_paths) != len(PREFIXES):
        raise ValueError("PW-0157 requires exactly five prefix manifests")
    authenticate_implementation_commit(commit)
    if sha256_file(topk_fixture_path) != TOPK_FIXTURE_SHA256:
        raise ValueError("top-k fixture SHA-256 mismatch")
    if sha256_file(corpus_path) != CORPUS_FIXTURE_SHA256:
        raise ValueError("corpus fixture SHA-256 mismatch")
    if sha256_file(verification_path) != VERIFICATION_SHA256:
        raise ValueError("checkpoint verification SHA-256 mismatch")
    topk_fixture = json.loads(topk_fixture_path.read_text())
    validate_topk_fixture(topk_fixture)
    corpus = json.loads(corpus_path.read_text())
    if (
        corpus.get("schema_version") != 5
        or corpus.get("semantic") != "mimo_v2_5_target_faithful_8k_prefill_route_coverage"
        or corpus.get("revision") != REVISION
        or corpus.get("route_trace_positions") != 8000
        or len(corpus.get("expected_prompt_token_ids", ())) != 8000
    ):
        raise ValueError("corpus authority mismatch")
    source_hashes = {
        "topk_fixture": TOPK_FIXTURE_SHA256,
        "corpus_fixture": CORPUS_FIXTURE_SHA256,
        "checkpoint_verification": VERIFICATION_SHA256,
    }
    summaries = []
    previous_routes = None
    previous_positions = None
    previous_distinct = -1
    previous_ties = -1
    for positions, path in zip(PREFIXES, manifest_paths, strict=True):
        actual_hash = sha256_file(path)
        if positions in FROZEN_PREFIX_HASHES and actual_hash != FROZEN_PREFIX_HASHES[positions]:
            raise ValueError(f"frozen prefix-{positions} manifest SHA-256 mismatch")
        manifest = json.loads(path.read_text())
        summary, routes = validate_manifest(manifest, positions, corpus)
        if previous_routes is not None:
            summary["comparison_with_previous_prefix"] = compare_route_prefix(
                previous_routes, routes, previous_positions
            )
        if (
            summary["distinct_layer_expert_records"] < previous_distinct
            or summary["pytorch_topk_boundary_tie_rows"] < previous_ties
        ):
            raise ValueError("prefix coverage or tie incidence regressed")
        source_hashes[f"prefix_{positions}"] = actual_hash
        summaries.append(summary)
        previous_routes = routes
        previous_positions = positions
        previous_distinct = summary["distinct_layer_expert_records"]
        previous_ties = summary["pytorch_topk_boundary_tie_rows"]
        del manifest
        gc.collect()
    final = summaries[-1]
    exceeds = final["exceeds_optimistic_15_second_storage_bound"]
    report = {
        "schema_version": 1,
        "evidence_class": "pw0157_pinned_pytorch_topk_route_coverage",
        "revision": REVISION,
        "analyzer_commit": commit,
        "runtime_commit": RUNTIME_COMMIT,
        "source_hashes": source_hashes,
        "prefixes": summaries,
        "final_coverage": {
            "positions": 8000,
            "distinct_layer_expert_records": final["distinct_layer_expert_records"],
            "first_decisive_distinct_record_count": FIRST_DECISIVE_RECORD_COUNT,
            "headroom_records_below_rejection": max(
                0, FIRST_DECISIVE_RECORD_COUNT - final["distinct_layer_expert_records"]
            ),
            "exceeds_optimistic_15_second_storage_bound": exceeds,
        },
        "decision": (
            "reject_four_lane_8k_storage_even_under_impossible_offline_residency"
            if exceeds
            else "retain_four_lane_8k_storage_capacity_only;_complete_two_p100_system_remains_rejected_by_pw0158"
        ),
        "tie_authority": "passed_exact_pinned_pytorch_2_13_0_libcxx_bridge",
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "exact source route coverage and optimistic storage capacity only; not measured storage, "
            "CUDA, full-capability hardware, prefill latency, accepted decode, or endpoint TPS"
        ),
    }
    atomic_write_new(output, canonical_json(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("topk_fixture", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("verification", type=Path)
    for positions in PREFIXES:
        parser.add_argument(f"prefix_{positions}", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [getattr(args, f"prefix_{positions}") for positions in PREFIXES]
    report = run(
        args.topk_fixture,
        args.corpus,
        args.verification,
        paths,
        args.output,
        args.commit,
    )
    print(canonical_json(report), end="")


if __name__ == "__main__":
    main()
