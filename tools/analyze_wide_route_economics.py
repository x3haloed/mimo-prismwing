#!/usr/bin/env python3
"""Analyze PW-0112 wide target-route union and exact expert-cache bounds."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Hashable, Sequence

try:
    from tools.analyze_real_route_cache import belady_hits, lru_hits
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_real_route_cache import belady_hits, lru_hits
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TRACE_COMMIT = "64be9a2db15b2cd2479cf146a44050bcff98f959"
TRACE_MANIFEST_SHA256 = "PENDING"
FIXTURE_SHA256 = "8f3da9f077df42c25490e71bf0a472194a365cb3dfc2a17d5047fe9c2186e1e5"
CHECKPOINT_VERIFICATION_SHA256 = (
    "9ddc8a99755f04ae2ea3c2484f6dd022d3f3a681b5a72c915ee4de833dbb0d03"
)
EXPERT_BYTES = 25_171_968
ROUTED_LAYERS = 47
TOP_K = 8
PROMPT_POSITIONS = 87
HOSTED_SUFFIX_POSITIONS = 192
SUFFIX_POSITIONS = 137
CACHE_POSITIONS = 128
TOTAL_POSITIONS = PROMPT_POSITIONS + SUFFIX_POSITIONS
ACQUISITION_FLOOR_SECONDS = 2.727590151
WIDTHS = (8, 16, 32, 64, 94, 128, 137)
CAPACITIES_GIB = (2, 3, 4)
PROTECTED_SERVICES = ("ChatGPT", "WindowServer", "nxnode", "syncthing")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Sequence[float | int]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p90": None,
                "p99": None, "maximum": None}
    ordered = sorted(values)

    def nearest_rank(fraction: float) -> float | int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": nearest_rank(0.90),
        "p99": nearest_rank(0.99),
        "maximum": ordered[-1],
    }


def normalized_window_union(
    routed_rows: Sequence[Sequence[Sequence[int]]], start: int, width: int
) -> tuple[float, list[int]]:
    if width <= 0 or start < 0 or start + width > len(routed_rows[0]):
        raise ValueError("invalid route-union window")
    counts = [
        len({expert for row in layer_rows[start : start + width] for expert in row})
        for layer_rows in routed_rows
    ]
    return statistics.fmean(count / TOP_K for count in counts), counts


def route_windows(routed_rows: Sequence[Sequence[Sequence[int]]]) -> dict:
    result = {}
    for width in WIDTHS:
        windows = []
        for start in range(SUFFIX_POSITIONS - width + 1):
            union, layer_counts = normalized_window_union(routed_rows, start, width)
            accepted_per_union = width / union
            windows.append(
                {
                    "suffix_start": start,
                    "q": width,
                    "mean_U": union,
                    "A_over_U_at_impossible_perfect_acceptance": accepted_per_union,
                    "optimistic_accepted_tps": accepted_per_union
                    / ACQUISITION_FLOOR_SECONDS,
                    "layer_unique_experts": layer_counts,
                }
            )
        result[str(width)] = {
            "q": width,
            "window_count": len(windows),
            "mean_U": distribution([window["mean_U"] for window in windows]),
            "A_over_U_at_impossible_perfect_acceptance": distribution(
                [window["A_over_U_at_impossible_perfect_acceptance"] for window in windows]
            ),
            "optimistic_accepted_tps": distribution(
                [window["optimistic_accepted_tps"] for window in windows]
            ),
            "windows": windows,
        }
    return result


def causal_accesses(
    routed_rows: Sequence[Sequence[Sequence[int]]], start: int, positions: int
) -> list[tuple[int, int]]:
    accesses = []
    for position in range(start, start + positions):
        for layer_offset, layer_rows in enumerate(routed_rows):
            accesses.extend((layer_offset + 1, expert) for expert in layer_rows[position])
    return accesses


def policy_metrics(accesses: list[Hashable], capacity: int, hits: int) -> dict:
    misses = len(accesses) - hits
    equivalent_tokens = len(accesses) / (ROUTED_LAYERS * TOP_K)
    return {
        "hits": hits,
        "misses": misses,
        "hit_ratio": hits / len(accesses),
        "avoided_logical_bytes": hits * EXPERT_BYTES,
        "avoided_logical_bytes_per_token": hits * EXPERT_BYTES / equivalent_tokens,
        "logical_miss_bytes": misses * EXPERT_BYTES,
        "logical_miss_bytes_per_token": misses * EXPERT_BYTES / equivalent_tokens,
    }


def cache_curves(accesses: list[tuple[int, int]]) -> list[dict]:
    curves = []
    for gib in CAPACITIES_GIB:
        capacity = gib * 1024**3 // EXPERT_BYTES
        lru = policy_metrics(accesses, capacity, lru_hits(accesses, capacity))
        belady = policy_metrics(accesses, capacity, belady_hits(accesses, capacity))
        if belady["hits"] < lru["hits"]:
            raise ValueError("Belady failed to upper-bound LRU")
        curves.append(
            {
                "capacity_gib": gib,
                "capacity_experts": capacity,
                "lru": lru,
                "belady": belady,
            }
        )
    return curves


def calibrated_frequency_cache(
    calibration: list[tuple[int, int]], holdout: list[tuple[int, int]], capacity: int
) -> dict:
    frequencies = Counter(calibration)
    residents = {
        key
        for key, _count in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )[:capacity]
    }
    hits = sum(item in residents for item in holdout)
    return {
        "capacity_experts": capacity,
        "calibration_positions": 32,
        "holdout_positions": 96,
        "resident_experts": len(residents),
        **policy_metrics(holdout, capacity, hits),
    }


def reuse_metrics(accesses: list[tuple[int, int]]) -> dict:
    previous: dict[tuple[int, int], int] = {}
    gaps = []
    by_layer: dict[int, list[int]] = {layer: [] for layer in range(1, 48)}
    for index, item in enumerate(accesses):
        if item in previous:
            gap = index - previous[item] - 1
            gaps.append(gap)
            by_layer[item[0]].append(gap)
        previous[item] = index
    return {
        "intervening_accesses": distribution(gaps),
        "next_use_intervening_accesses": distribution(gaps),
        "by_layer": {
            str(layer): {
                "reuse_intervening_accesses": distribution(layer_gaps),
                "next_use_intervening_accesses": distribution(layer_gaps),
            }
            for layer, layer_gaps in by_layer.items()
        },
    }


def expert_frequency(accesses: list[tuple[int, int]]) -> dict:
    frequencies = Counter(accesses)
    ordered = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    by_layer = {}
    for layer in range(1, 48):
        layer_counts = [count for (item_layer, _expert), count in ordered if item_layer == layer]
        by_layer[str(layer)] = {
            "distinct_experts": len(layer_counts),
            "access_frequency": distribution(layer_counts),
        }
    return {
        "equal_physical_bytes_per_expert": EXPERT_BYTES,
        "access_frequency": distribution(list(frequencies.values())),
        "top_32_layer_experts": [
            {"layer": key[0], "expert": key[1], "accesses": count,
             "selected_bytes": count * EXPERT_BYTES}
            for key, count in ordered[:32]
        ],
        "by_layer": by_layer,
    }


def route_set_persistence(routed_rows: Sequence[Sequence[Sequence[int]]]) -> dict:
    intersections = []
    jaccards = []
    identical = 0
    comparisons = 0
    by_layer = {}
    for layer, rows in enumerate(routed_rows, start=1):
        layer_intersections = []
        layer_identical = 0
        for position in range(1, SUFFIX_POSITIONS):
            left = set(rows[position - 1])
            right = set(rows[position])
            intersection = len(left & right)
            layer_intersections.append(intersection)
            intersections.append(intersection)
            jaccards.append(intersection / len(left | right))
            layer_identical += left == right
            identical += left == right
            comparisons += 1
        by_layer[str(layer)] = {
            "intersection": distribution(layer_intersections),
            "identical_count": layer_identical,
            "identical_ratio": layer_identical / (SUFFIX_POSITIONS - 1),
        }
    return {
        "comparisons": comparisons,
        "intersection": distribution(intersections),
        "jaccard": distribution(jaccards),
        "identical_count": identical,
        "identical_ratio": identical / comparisons,
        "by_layer": by_layer,
    }


def validate_safety(snapshots: object) -> dict:
    if not isinstance(snapshots, list):
        raise ValueError("missing Gate 8 snapshots")
    expected_phases = ["process_start", "checkpoint_open"] + [
        f"layer_{layer}_complete" for layer in range(48)
    ] + ["route_evidence_serialized", "checkpoint_released", "final_service_health"]
    phases = [snapshot.get("phase") for snapshot in snapshots]
    if phases != expected_phases:
        raise ValueError("Gate 8 phase ledger mismatch")
    baseline = snapshots[0].get("protected_service_pids")
    if not isinstance(baseline, dict):
        raise ValueError("missing protected-service baseline")
    for snapshot in snapshots:
        services = snapshot.get("protected_service_pids")
        if (
            snapshot.get("system_memory_free_percent", -1) < 20
            or snapshot.get("swap_growth_bytes", 2**63) > 512 * 1024**2
            or snapshot.get("new_throttled_pages") != 0
            or snapshot.get("process_physical_footprint_bytes", 2**63) > 8 * 1024**3
            or snapshot.get("process_peak_resident_bytes", 2**63) > 8 * 1024**3
            or not isinstance(services, dict)
            or any(not services.get(name) for name in PROTECTED_SERVICES)
        ):
            raise ValueError(f"Gate 8 failed at {snapshot.get('phase')}")
    if snapshots[-1]["process_physical_footprint_bytes"] > 4 * 1024**3:
        raise ValueError("released footprint exceeds Gate 8 limit")
    return {
        "snapshot_count": len(snapshots),
        "minimum_system_memory_free_percent": min(
            snapshot["system_memory_free_percent"] for snapshot in snapshots
        ),
        "maximum_swap_growth_bytes": max(
            snapshot["swap_growth_bytes"] for snapshot in snapshots
        ),
        "maximum_new_throttled_pages": max(
            snapshot["new_throttled_pages"] for snapshot in snapshots
        ),
        "maximum_process_physical_footprint_bytes": max(
            snapshot["process_physical_footprint_bytes"] for snapshot in snapshots
        ),
        "maximum_process_peak_resident_bytes": max(
            snapshot["process_peak_resident_bytes"] for snapshot in snapshots
        ),
        "final_process_physical_footprint_bytes": snapshots[-1][
            "process_physical_footprint_bytes"
        ],
        "protected_services_stable": True,
    }


def load_routes(manifest: dict) -> list[list[list[int]]]:
    traces = manifest.get("layer_traces")
    if not isinstance(traces, list) or len(traces) != 48:
        raise ValueError("route layer count mismatch")
    if traces[0].get("selected_experts_by_position") != []:
        raise ValueError("dense layer unexpectedly reports routed experts")
    routed_rows = []
    for layer in range(1, 48):
        trace = traces[layer]
        rows = trace.get("selected_experts_by_position")
        weights = trace.get("route_weights_by_position")
        if trace.get("layer") != layer or not isinstance(rows, list) or len(rows) != TOTAL_POSITIONS:
            raise ValueError(f"route row authority mismatch at layer {layer}")
        if not isinstance(weights, list) or len(weights) != TOTAL_POSITIONS:
            raise ValueError(f"route-weight authority mismatch at layer {layer}")
        for experts, route_weights in zip(rows, weights):
            if (
                not isinstance(experts, list)
                or len(experts) != TOP_K
                or len(set(experts)) != TOP_K
                or any(not isinstance(expert, int) or expert < 0 or expert >= 256 for expert in experts)
                or not isinstance(route_weights, list)
                or len(route_weights) != TOP_K
                or any(not isinstance(weight, (float, int)) or not math.isfinite(weight)
                       for weight in route_weights)
            ):
                raise ValueError(f"route identity mismatch at layer {layer}")
        routed_rows.append(rows[PROMPT_POSITIONS:])
    return routed_rows


def analyze(manifest_path: Path) -> dict:
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != TRACE_MANIFEST_SHA256:
        raise ValueError("PW-0112 trace manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("semantic") != "mimo_teacher_forced_route_only_rust_trace"
        or manifest.get("revision") != REVISION
        or manifest.get("commit") != TRACE_COMMIT
        or manifest.get("fixture_sha256") != FIXTURE_SHA256
        or manifest.get("checkpoint_verification_sha256") != CHECKPOINT_VERIFICATION_SHA256
        or manifest.get("prompt_positions") != PROMPT_POSITIONS
        or manifest.get("hosted_suffix_positions") != HOSTED_SUFFIX_POSITIONS
        or manifest.get("teacher_forced_positions") != SUFFIX_POSITIONS
        or manifest.get("total_positions") != TOTAL_POSITIONS
        or manifest.get("accepted_tokens") != 0
        or manifest.get("performance_claim") is not None
    ):
        raise ValueError("PW-0112 trace authority mismatch")
    safety = validate_safety(manifest.get("safety_snapshots"))
    routed_rows = load_routes(manifest)
    windows = route_windows(routed_rows)
    full_accesses = causal_accesses(routed_rows, 0, CACHE_POSITIONS)
    calibration = causal_accesses(routed_rows, 0, 32)
    holdout = causal_accesses(routed_rows, 32, 96)
    expected_accesses = CACHE_POSITIONS * ROUTED_LAYERS * TOP_K
    if len(full_accesses) != expected_accesses:
        raise ValueError("causal access ledger mismatch")
    curves = cache_curves(full_accesses)
    calibration_curves = cache_curves(calibration)
    holdout_curves = cache_curves(holdout)
    frequency_holdout = []
    for gib in CAPACITIES_GIB:
        capacity = gib * 1024**3 // EXPERT_BYTES
        frequency_holdout.append(
            {"capacity_gib": gib, **calibrated_frequency_cache(calibration, holdout, capacity)}
        )
    four_gib_belady = curves[-1]["belady"]["hit_ratio"]
    best_q94 = windows["94"]["A_over_U_at_impossible_perfect_acceptance"]["maximum"]
    best_q137 = windows["137"]["A_over_U_at_impossible_perfect_acceptance"]["maximum"]
    wide_34_killed = best_q94 < 93.556
    wide_50_killed = best_q137 < 136.380
    cache_killed = four_gib_belady < 0.30
    analyzer_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
        ).stdout.strip()
    )
    return {
        "schema_version": 1,
        "evidence_class": "pw0112_wide_teacher_forced_route_economics",
        "revision": REVISION,
        "trace_manifest_sha256": manifest_sha256,
        "trace_commit": TRACE_COMMIT,
        "analyzer_commit": analyzer_commit,
        "git_dirty": dirty,
        "fixture_sha256": FIXTURE_SHA256,
        "checkpoint_verification_sha256": CHECKPOINT_VERIFICATION_SHA256,
        "prompt_positions": PROMPT_POSITIONS,
        "teacher_forced_positions": SUFFIX_POSITIONS,
        "cache_positions": CACHE_POSITIONS,
        "routed_layers": ROUTED_LAYERS,
        "top_k": TOP_K,
        "expert_bytes": EXPERT_BYTES,
        "pw0110_acquisition_floor_seconds": ACQUISITION_FLOOR_SECONDS,
        "sliding_widths": windows,
        "cache": {
            "order": "continuation_token_major_then_layer_1_through_47_then_native_top8",
            "accesses": len(full_accesses),
            "distinct_layer_experts": len(set(full_accesses)),
            "logical_selected_bytes": len(full_accesses) * EXPERT_BYTES,
            "access_list_sha256": hashlib.sha256(
                canonical_json([[layer, expert] for layer, expert in full_accesses])
            ).hexdigest(),
            "first_128_position_curves": curves,
            "calibration_first_32_cold_curves": calibration_curves,
            "holdout_following_96_cold_curves": holdout_curves,
            "calibration_frequency_cache_on_holdout": frequency_holdout,
            "reuse_distance": reuse_metrics(full_accesses),
            "expert_frequency": expert_frequency(full_accesses),
            "route_set_persistence": route_set_persistence(routed_rows),
        },
        "gates": {
            "best_q94_A_over_U": best_q94,
            "required_q94_A_over_U_for_34_3": 93.556,
            "source_fp8_wide_speculation_killed_for_34_3_on_trace": wide_34_killed,
            "best_q137_A_over_U": best_q137,
            "required_q137_A_over_U_for_50": 136.380,
            "source_fp8_wide_speculation_killed_for_50_on_trace": wide_50_killed,
            "four_gib_belady_hit_ratio": four_gib_belady,
            "four_gib_cache_kill_threshold": 0.30,
            "two_to_four_gib_cache_killed_on_trace": cache_killed,
        },
        "safety": safety,
        "decision": (
            "kill_source_fp8_wide_speculation_and_2_to_4_gib_exact_cache_on_this_trace"
            if wide_34_killed and wide_50_killed and cache_killed
            else "retain_only_mechanisms_whose_frozen_gate_passed"
        ),
        "limitations": (
            "single frozen 192-token text continuation; impossible-perfect speculative "
            "acceptance; logical equal-size source-FP8 expert bytes; Belady is noncausal; "
            "no modality trace, physical cache timing, proposer cost, or endpoint TPS claim"
        ),
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.manifest)
        atomic_write_new(arguments.output, canonical_json(result))
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "decision": result["decision"],
                    "gates": result["gates"],
                }
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
