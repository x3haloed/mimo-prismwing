#!/usr/bin/env python3
"""Authenticate and analyze PW-0170's single-A770 storage envelope."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

try:
    from tools.analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from tools.analyze_prompt_calibrated_p100_hbm_cache import (
        EXPERT_BYTES,
        THREE_ARENA_BYTES,
        _load_and_validate_routes,
        kv_capacity_bytes,
        prompt_frequency_cache,
    )
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_prompt_calibrated_p100_hbm_cache import (
        EXPERT_BYTES,
        THREE_ARENA_BYTES,
        _load_and_validate_routes,
        kv_capacity_bytes,
        prompt_frequency_cache,
    )
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
CONFIG_SHA256 = "292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587"
TRACE_SHA256 = "584d3a8b1b09b12d4f83908be1fa5471b9fd66373500cc56332213928cd0bc3e"
CENSUS_SHA256 = "8ac4a179c7b0a06baee05e380dc76acd0a1a64cff4d3e2abe9572ce59afb5c52"
PW0152_SHA256 = "68783813c30d08aabb6c23971d65b2579655314819ea8d6e1aef8b19328bc686"
PW0154_SHA256 = "1b57250d45f1b24e32f43e93a653fc3d00fa061e37cd0df1c6f0fdff551535f2"
PW0155_SHA256 = "226603fb2b44e1162a038f51bae47520238150f3b26e39e1cf33c7420b88b064"
PW0169_SHA256 = "127a898e54f51044bf68bf58f80d071e98b2e10130f2b008a6fe0d313d2d9db3"

A770_HBM_DECIMAL_BYTES = 16_000_000_000
A770_DERIVED_BF16_F32ACC_FLOPS = 131_000_000_000_000.0
EPYC_IMPOSSIBLE_FP32_FLOPS = 742_400_000_000.0
MANDATORY_OPERATIONS_PER_TOKEN = 29_641_146_368
Q = 137
ACTIVE_CARD_PLUS_SHIPPING_USD = 311.71
HARDWARE_CAP_USD = 500.0


def a770_capacity(non_routed_bytes: int, kv_bytes: int) -> dict:
    if non_routed_bytes <= 0 or kv_bytes <= 0:
        raise ValueError("capacity inputs must be positive")
    reserved = non_routed_bytes + THREE_ARENA_BYTES + kv_bytes
    available = A770_HBM_DECIMAL_BYTES - reserved
    if available < 0:
        raise ValueError("A770 HBM cannot hold mandatory resident state")
    slots = available // EXPERT_BYTES
    return {
        "a770_hbm_decimal_bytes": A770_HBM_DECIMAL_BYTES,
        "all_non_routed_source_tensor_bytes": non_routed_bytes,
        "three_arena_bytes": THREE_ARENA_BYTES,
        "bf16_8k_kv_bytes": kv_bytes,
        "reserved_before_cache_bytes": reserved,
        "available_for_complete_experts_bytes": available,
        "complete_expert_slots": slots,
        "expert_cache_bytes": slots * EXPERT_BYTES,
        "unallocated_tail_bytes": available - slots * EXPERT_BYTES,
    }


def storage_scenarios(residual_bytes: int) -> list[dict]:
    if residual_bytes <= 0:
        raise ValueError("residual bytes must be positive")
    mandatory_operations = MANDATORY_OPERATIONS_PER_TOKEN * Q
    granted_compute = A770_DERIVED_BF16_F32ACC_FLOPS + EPYC_IMPOSSIBLE_FP32_FLOPS
    compute_seconds = mandatory_operations / granted_compute
    scenarios = []
    for bytes_per_second_per_lane in (2.5e9, 3.5e9):
        for lanes in range(1, 5):
            storage_seconds = residual_bytes / (bytes_per_second_per_lane * lanes)
            serial_seconds = storage_seconds + compute_seconds
            targets = {}
            for target in (34.3, 50.0):
                required_a = math.ceil(target * serial_seconds)
                targets[str(target)] = {
                    "minimum_integer_A": required_a,
                    "fraction_of_q": required_a / Q,
                    "possible_with_A_at_most_q": required_a <= Q,
                }
            scenarios.append(
                {
                    "lanes": lanes,
                    "granted_nameplate_bytes_per_second_per_lane": bytes_per_second_per_lane,
                    "granted_aggregate_bytes_per_second": lanes * bytes_per_second_per_lane,
                    "residual_expert_acquisition_seconds": storage_seconds,
                    "granted_a770_bf16_f32acc_flops": A770_DERIVED_BF16_F32ACC_FLOPS,
                    "granted_epyc_fp32_flops": EPYC_IMPOSSIBLE_FP32_FLOPS,
                    "mandatory_q137_compute_seconds": compute_seconds,
                    "serial_storage_plus_compute_seconds": serial_seconds,
                    "impossible_perfect_acceptance_tps": Q / serial_seconds,
                    "targets": targets,
                }
            )
    return scenarios


def conditional_match_probability(expected_a: int) -> float:
    if not 0 < expected_a <= Q:
        raise ValueError("expected acceptance must fit q")
    low, high = 0.0, 1.0
    for _ in range(100):
        probability = (low + high) / 2.0
        expectation = sum(probability**index for index in range(Q))
        if expectation < expected_a:
            low = probability
        else:
            high = probability
    return (low + high) / 2.0


def authenticate(paths: dict[str, Path]) -> tuple[dict, list[list[list[int]]], dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "trace": TRACE_SHA256,
        "census": CENSUS_SHA256,
        "pw0152": PW0152_SHA256,
        "pw0154": PW0154_SHA256,
        "pw0155": PW0155_SHA256,
        "pw0169": PW0169_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0170 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    config = json.loads(paths["config"].read_text())
    trace = json.loads(paths["trace"].read_text())
    census = json.loads(paths["census"].read_text())
    pw0152 = json.loads(paths["pw0152"].read_text())
    pw0154 = json.loads(paths["pw0154"].read_text())
    pw0155 = json.loads(paths["pw0155"].read_text())
    pw0169 = json.loads(paths["pw0169"].read_text())
    if (
        "USD $500 total" not in target
        or "50 accepted TPS" not in target
        or config.get("model_type") != "mimo_v2"
        or census.get("revision") != REVISION
        or trace.get("revision") != REVISION
        or pw0152.get("evidence_class") != "pw0152_wide_proposer_acceptance_prerequisite"
        or pw0154.get("evidence_class") != "pw0154_prompt_calibrated_p100_hbm_cache_bound"
        or pw0155.get("evidence_class") != "pw0155_owned_epyc_installable_bom_prerequisite"
        or pw0169.get("evidence_class") != "pw0169_active_a770_limited_edition_bom_preflight"
        or pw0169.get("exact_board", {}).get("product") != "Intel Arc A770 Limited Edition 16GB"
        or pw0169.get("cost", {}).get("observed_item_plus_shipping_usd")
        != ACTIVE_CARD_PLUS_SHIPPING_USD
        or pw0169.get("performance_continuation", {}).get(
            "minimum_device_bf16_f32acc_tflops_at_zero_other_cost"
        )
        != 118.23859445778204
        or pw0155.get("topology", {}).get("x16_slots") != [2, 4, 6]
        or "x4x4x4x4" not in pw0155.get("topology", {}).get("x16_bifurcation_options", [])
        or pw0155.get("topology", {}).get("logical_lane_topology_supported") is not True
    ):
        raise ValueError("PW-0170 source semantic mismatch")
    return config, _load_and_validate_routes(trace), census, pw0169


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    config, routes, census, pw0169 = authenticate(paths)
    safety.checkpoint("all_capacity_route_hardware_sources_authenticated")
    kv = kv_capacity_bytes(config)
    non_routed_bytes = census["tensor_data_bytes"] - census["categories"]["routed_experts"]["data_bytes"]
    capacity = a770_capacity(non_routed_bytes, kv["total_bytes"])
    cache = prompt_frequency_cache(routes, capacity["complete_expert_slots"])
    scenarios = storage_scenarios(cache["residual_union_bytes"])
    if capacity["complete_expert_slots"] != 25 or cache["suffix_union_misses"] != 878:
        raise ValueError("PW-0170 frozen A770 capacity result changed")
    fast_four = next(
        row for row in scenarios
        if row["lanes"] == 4
        and row["granted_nameplate_bytes_per_second_per_lane"] == 3.5e9
    )
    conservative_four = next(
        row for row in scenarios
        if row["lanes"] == 4
        and row["granted_nameplate_bytes_per_second_per_lane"] == 2.5e9
    )
    safety.checkpoint("a770_capacity_cache_and_storage_envelope_computed")
    safety.release_checkpoint(
        "source_payloads_released",
        ["target", "config", "route trace", "checkpoint census", "prior reports"],
    )
    safety.checkpoint("final_service_health")
    remaining_pre_tax = HARDWARE_CAP_USD - ACTIVE_CARD_PLUS_SHIPPING_USD
    report = {
        "schema_version": 1,
        "evidence_class": "pw0170_single_a770_storage_envelope",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "bf16_8k_kv_ledger": kv,
        "a770_hbm_capacity_ledger": capacity,
        "causal_prompt_frequency_cache": cache,
        "storage_scenarios": scenarios,
        "strongest_nameplate_survivor": {
            "lanes": 4,
            "bytes_per_second_per_lane": 3.5e9,
            "minimum_A_34_3": fast_four["targets"]["34.3"]["minimum_integer_A"],
            "minimum_A_50": fast_four["targets"]["50.0"]["minimum_integer_A"],
            "diagnostic_independent_match_probability_for_A_34_3": conditional_match_probability(
                fast_four["targets"]["34.3"]["minimum_integer_A"]
            ),
            "diagnostic_independent_match_probability_for_A_50": conditional_match_probability(
                fast_four["targets"]["50.0"]["minimum_integer_A"]
            ),
            "width_8_or_16_can_reach_required_A_in_one_transaction": False,
        },
        "conservative_four_lane_sensitivity": {
            "minimum_A_34_3": conservative_four["targets"]["34.3"]["minimum_integer_A"],
            "minimum_A_50": conservative_four["targets"]["50.0"]["minimum_integer_A"],
        },
        "project_ledger": {
            "hardware_cap_usd": HARDWARE_CAP_USD,
            "active_card_plus_observed_shipping_usd": ACTIVE_CARD_PLUS_SHIPPING_USD,
            "maximum_remaining_before_tax_storage_carrier_cables_and_cooling_usd": remaining_pre_tax,
            "four_independent_storage_lanes_owned": False,
            "complete_delivered_bom_proven": False,
            "purchase_authorized": False,
        },
        "decision": (
            "reject_single_a770_hbm_cache_as_primary_mechanism;"
            "retain_only_four_lane_q137_nameplate_envelope_pending_installed_compute_storage_proposer_and_complete_bom"
        ),
        "limitations": (
            "analytical nameplate upper bound; HBM is treated as fungible and all common tensors remain resident; "
            "no measured A770 BF16, PCIe, NVMe, ReBAR-off, oneAPI, proposer, power, endpoint, modality, fidelity, or TPS result"
        ),
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "endpoint_tps": None,
        "performance_claim": None,
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "safety": safety.evidence(),
    }
    atomic_write_new(output, canonical_json(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("target", "config", "trace", "census", "pw0152", "pw0154", "pw0155", "pw0169"):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ("target", "config", "trace", "census", "pw0152", "pw0154", "pw0155", "pw0169")
    report = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(report).decode(), end="")


if __name__ == "__main__":
    main()
