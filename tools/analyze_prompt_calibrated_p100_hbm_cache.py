#!/usr/bin/env python3
"""Run PW-0154's authenticated prompt-calibrated P100 HBM-cache bound."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import time

try:
    from tools.analyze_owned_epyc_companion_envelope import (
        authenticate_implementation_commit,
    )
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
CONFIG_SHA256 = "292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587"
TRACE_SHA256 = "584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e"
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
CENSUS_SHA256 = "8ac4a179c7b0a06baee05e380dc76acd0a1a64cff4d3e2abe9572ce59afb5c52"
TRACE_COMMIT = "9647b740f8f19c075ec752ed044795fa20c1102a"
TRACE_SEMANTIC = "mimo_teacher_forced_route_only_rust_trace"
CENSUS_EVIDENCE_CLASS = "pinned_remote_headers_not_local_payload_verification"
PROMPT_POSITIONS = 87
SUFFIX_POSITIONS = 137
TOTAL_POSITIONS = PROMPT_POSITIONS + SUFFIX_POSITIONS
ROUTED_LAYERS = 47
TOP_K = 8
EXPERT_BYTES = 25_171_968
EXPECTED_Q137_BYTES = 22_730_287_104
Q137_COMPUTE_SECONDS = 0.20994483892464225
TWO_P100_HBM_DECIMAL_BYTES = 32_000_000_000
THREE_ARENA_BYTES = 2_340_993_024


def kv_capacity_bytes(config: dict, full_positions: int = 8_000) -> dict:
    if full_positions <= 0:
        raise ValueError("full positions must be positive")
    pattern = config.get("hybrid_layer_pattern")
    if not isinstance(pattern, list) or len(pattern) != 48 or set(pattern) != {0, 1}:
        raise ValueError("hybrid attention pattern mismatch")
    full_layers = pattern.count(0)
    swa_layers = pattern.count(1)
    full_per_position = (
        config.get("num_key_value_heads")
        * (config.get("head_dim") + config.get("v_head_dim"))
        * 2
    )
    swa_per_position = (
        config.get("swa_num_key_value_heads")
        * (config.get("swa_head_dim") + config.get("swa_v_head_dim"))
        * 2
    )
    sliding_positions = min(full_positions, config.get("sliding_window_size"))
    full_bytes = full_layers * full_positions * full_per_position
    swa_bytes = swa_layers * sliding_positions * swa_per_position
    return {
        "dtype": "BF16",
        "bytes_per_value": 2,
        "full_attention_layers": full_layers,
        "sliding_window_layers": swa_layers,
        "full_attention_positions": full_positions,
        "sliding_window_positions": sliding_positions,
        "full_attention_bytes": full_bytes,
        "sliding_window_bytes": swa_bytes,
        "total_bytes": full_bytes + swa_bytes,
    }


def expert_cache_capacity(non_routed_bytes: int, kv_bytes: int) -> dict:
    if non_routed_bytes <= 0 or kv_bytes <= 0:
        raise ValueError("cache capacity inputs must be positive")
    reserved = non_routed_bytes + THREE_ARENA_BYTES + kv_bytes
    available = TWO_P100_HBM_DECIMAL_BYTES - reserved
    if available < EXPERT_BYTES:
        raise ValueError("aggregate HBM has no complete expert slot")
    slots = available // EXPERT_BYTES
    return {
        "aggregate_hbm_decimal_bytes": TWO_P100_HBM_DECIMAL_BYTES,
        "all_non_routed_source_tensor_bytes": non_routed_bytes,
        "three_arena_bytes": THREE_ARENA_BYTES,
        "bf16_8k_kv_bytes": kv_bytes,
        "reserved_before_cache_bytes": reserved,
        "available_for_complete_experts_bytes": available,
        "complete_expert_slots": slots,
        "expert_cache_bytes": slots * EXPERT_BYTES,
        "unallocated_tail_bytes": available - slots * EXPERT_BYTES,
    }


def _causal_accesses(
    routes: list[list[list[int]]], start: int, positions: int
) -> list[tuple[int, int]]:
    return [
        (layer, expert)
        for layer, rows in enumerate(routes, start=1)
        for row in rows[start : start + positions]
        for expert in row
    ]


def prompt_frequency_cache(
    routes: list[list[list[int]]], capacity: int
) -> dict:
    if len(routes) != ROUTED_LAYERS or capacity <= 0:
        raise ValueError("cache analysis input mismatch")
    prompt = _causal_accesses(routes, 0, PROMPT_POSITIONS)
    suffix = _causal_accesses(routes, PROMPT_POSITIONS, SUFFIX_POSITIONS)
    frequencies = Counter(prompt)
    residents = {
        key
        for key, _count in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )[:capacity]
    }
    suffix_union = set(suffix)
    union_hits = suffix_union & residents
    union_misses = suffix_union - residents
    access_hits = sum(key in residents for key in suffix)
    by_card = {}
    for name, first_layer, last_layer in (
        ("card_0_layers_1_24", 1, 24),
        ("card_1_layers_25_47", 25, 47),
    ):
        count = sum(first_layer <= layer <= last_layer for layer, _expert in residents)
        by_card[name] = {
            "resident_experts": count,
            "resident_expert_bytes": count * EXPERT_BYTES,
        }
    return {
        "policy": "static_prompt_frequency_with_layer_expert_tie_break",
        "prompt_positions": PROMPT_POSITIONS,
        "suffix_positions": SUFFIX_POSITIONS,
        "capacity_experts": capacity,
        "prompt_distinct_experts": len(frequencies),
        "resident_experts": len(residents),
        "resident_expert_bytes": len(residents) * EXPERT_BYTES,
        "suffix_accesses": len(suffix),
        "suffix_access_hits": access_hits,
        "suffix_access_hit_ratio": access_hits / len(suffix),
        "suffix_union_experts": len(suffix_union),
        "suffix_union_bytes": len(suffix_union) * EXPERT_BYTES,
        "suffix_union_hits": len(union_hits),
        "suffix_union_misses": len(union_misses),
        "suffix_union_avoided_ratio": len(union_hits) / len(suffix_union),
        "residual_union_bytes": len(union_misses) * EXPERT_BYTES,
        "contiguous_layer_sharding_diagnostic": by_card,
    }


def storage_scenarios(residual_bytes: int) -> list[dict]:
    if residual_bytes <= 0:
        raise ValueError("residual bytes must be positive")
    scenarios = []
    for per_lane in (2.5e9, 3.5e9):
        for lanes in range(1, 5):
            acquisition = residual_bytes / (per_lane * lanes)
            serial = acquisition + Q137_COMPUTE_SECONDS
            targets = {}
            for target in (34.3, 50.0):
                required_a = math.ceil(target * serial)
                targets[str(target)] = {
                    "minimum_integer_A": required_a,
                    "fraction_of_q137": required_a / SUFFIX_POSITIONS,
                    "possible_with_A_at_most_q": required_a <= SUFFIX_POSITIONS,
                }
            scenarios.append(
                {
                    "lanes": lanes,
                    "granted_nameplate_bytes_per_second_per_lane": per_lane,
                    "granted_aggregate_bytes_per_second": per_lane * lanes,
                    "residual_expert_acquisition_seconds": acquisition,
                    "direct_fp32_block_compute_seconds": Q137_COMPUTE_SECONDS,
                    "serial_residual_expert_plus_matrix_floor_seconds": serial,
                    "impossible_perfect_acceptance_tps": SUFFIX_POSITIONS / serial,
                    "targets": targets,
                }
            )
    return scenarios


def _load_and_validate_routes(manifest: dict) -> list[list[list[int]]]:
    traces = manifest.get("layer_traces")
    if not isinstance(traces, list) or len(traces) != 48:
        raise ValueError("trace layer count mismatch")
    routes = []
    for layer in range(1, 48):
        trace = traces[layer]
        rows = trace.get("selected_experts_by_position")
        weights = trace.get("route_weights_by_position")
        if (
            trace.get("layer") != layer
            or not isinstance(rows, list)
            or len(rows) != TOTAL_POSITIONS
            or not isinstance(weights, list)
            or len(weights) != TOTAL_POSITIONS
        ):
            raise ValueError(f"route authority mismatch at layer {layer}")
        for experts, route_weights in zip(rows, weights):
            if (
                not isinstance(experts, list)
                or len(experts) != TOP_K
                or len(set(experts)) != TOP_K
                or any(not isinstance(expert, int) or not 0 <= expert < 256 for expert in experts)
                or not isinstance(route_weights, list)
                or len(route_weights) != TOP_K
                or any(not isinstance(weight, (int, float)) or not math.isfinite(weight) for weight in route_weights)
            ):
                raise ValueError(f"route row mismatch at layer {layer}")
        routes.append(rows)
    return routes


def _authenticate_sources(
    config_path: Path,
    trace_path: Path,
    pw0151_path: Path,
    census_path: Path,
) -> tuple[dict, list[list[list[int]]], dict, dict]:
    expected = {
        config_path: CONFIG_SHA256,
        trace_path: TRACE_SHA256,
        pw0151_path: PW0151_SHA256,
        census_path: CENSUS_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"PW-0154 source hash mismatch: {path.name}")
    config = json.loads(config_path.read_text())
    trace = json.loads(trace_path.read_text())
    pw0151 = json.loads(pw0151_path.read_text())
    census = json.loads(census_path.read_text())
    if (
        trace.get("semantic") != TRACE_SEMANTIC
        or trace.get("revision") != REVISION
        or trace.get("commit") != TRACE_COMMIT
        or trace.get("prompt_positions") != PROMPT_POSITIONS
        or trace.get("teacher_forced_positions") != SUFFIX_POSITIONS
        or trace.get("total_positions") != TOTAL_POSITIONS
        or trace.get("accepted_tokens") != 0
        or trace.get("performance_claim") is not None
    ):
        raise ValueError("PW-0112 trace authority mismatch")
    if (
        config.get("model_type") != "mimo_v2"
        or config.get("num_hidden_layers") != 48
        or config.get("n_routed_experts") != 256
        or config.get("num_experts_per_tok") != TOP_K
        or config.get("dtype") != "bfloat16"
    ):
        raise ValueError("pinned config authority mismatch")
    q137 = pw0151.get("route_windows", {}).get("137", [])
    if (
        pw0151.get("evidence_class") != "pw0151_owned_epyc_companion_envelope"
        or len(q137) != 1
        or q137[0].get("selected_source_expert_bytes") != EXPECTED_Q137_BYTES
        or pw0151.get("named_surviving_envelope", {}).get(
            "q137_direct_fp32_block_compute_seconds"
        )
        != Q137_COMPUTE_SECONDS
    ):
        raise ValueError("PW-0151 authority mismatch")
    if (
        census.get("evidence_class") != CENSUS_EVIDENCE_CLASS
        or census.get("revision") != REVISION
        or census.get("categories", {}).get("routed_experts", {}).get("data_bytes")
        != 302_869_118_976
    ):
        raise ValueError("checkpoint census authority mismatch")
    return config, _load_and_validate_routes(trace), pw0151, census


def run(
    config_path: Path,
    trace_path: Path,
    pw0151_path: Path,
    census_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    config, routes, pw0151, census = _authenticate_sources(
        config_path, trace_path, pw0151_path, census_path
    )
    safety.checkpoint("source_evidence_authenticated")

    kv = kv_capacity_bytes(config)
    non_routed_bytes = (
        census["tensor_data_bytes"]
        - census["categories"]["routed_experts"]["data_bytes"]
    )
    capacity = expert_cache_capacity(non_routed_bytes, kv["total_bytes"])
    cache = prompt_frequency_cache(routes, capacity["complete_expert_slots"])
    if capacity["complete_expert_slots"] != 661:
        raise ValueError("HBM expert capacity changed")
    if cache["suffix_union_misses"] != 424:
        raise ValueError("causal suffix miss union changed")
    scenarios = storage_scenarios(cache["residual_union_bytes"])
    safety.checkpoint("capacity_cache_and_storage_analysis_complete")
    safety.release_checkpoint(
        "source_reports_released",
        ["config", "PW-0112 route trace", "PW-0151 report", "checkpoint census"],
    )
    safety.checkpoint("final_service_health")

    one_lane_fast = next(
        row
        for row in scenarios
        if row["lanes"] == 1
        and row["granted_nameplate_bytes_per_second_per_lane"] == 3.5e9
    )
    four_lane_fast = next(
        row
        for row in scenarios
        if row["lanes"] == 4
        and row["granted_nameplate_bytes_per_second_per_lane"] == 3.5e9
    )
    report = {
        "schema_version": 1,
        "evidence_class": "pw0154_prompt_calibrated_p100_hbm_cache_bound",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "config_sha256": CONFIG_SHA256,
            "pw0112_route_trace_sha256": TRACE_SHA256,
            "pw0151_analysis_sha256": PW0151_SHA256,
            "checkpoint_census_sha256": CENSUS_SHA256,
        },
        "bf16_8k_kv_ledger": kv,
        "aggregate_hbm_capacity_ledger": capacity,
        "causal_prompt_frequency_cache": cache,
        "storage_scenarios": scenarios,
        "structural_findings": {
            "one_3_5_GBps_lane_perfect_50_tps_possible": one_lane_fast["targets"]["50.0"]["possible_with_A_at_most_q"],
            "one_3_5_GBps_lane_perfect_tps": one_lane_fast["impossible_perfect_acceptance_tps"],
            "four_3_5_GBps_lane_required_A_34_3": four_lane_fast["targets"]["34.3"]["minimum_integer_A"],
            "four_3_5_GBps_lane_required_A_50": four_lane_fast["targets"]["50.0"]["minimum_integer_A"],
            "published_width_16_can_reach_four_lane_34_3_in_one_transaction": False,
            "published_width_16_can_reach_four_lane_50_in_one_transaction": False,
        },
        "decision": (
            "reject_one_lane_for_prismwing_50_even_at_perfect_acceptance;"
            "retain_exact_hbm_cache_with_two_to_four_storage_lanes_as_unproven_envelope"
        ),
        "limitations": [
            "aggregate HBM arithmetic does not prove per-card sharding or tensor-parallel communication",
            "cache hits and storage bandwidth are logical/nameplate values, not measured I/O",
            "cache installation, FP8 decode, host work, KV movement, dispatch, and overlap are omitted",
            "8K prefill storage and 1M-context KV capacity are not solved by this cache ledger",
            "no complete BOM, electrical, thermal, physical, CUDA, fidelity, or endpoint evidence exists",
        ],
        "A": 0,
        "accepted_tokens": 0,
        "performance_claim": "none; analytical exact-cache envelope only",
        "gates_passed": True,
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "safety": safety.evidence(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--pw0151", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.trace,
        arguments.pw0151,
        arguments.census,
        arguments.output,
        arguments.commit,
    )
    print(canonical_json(result).decode(), end="")


if __name__ == "__main__":
    main()
