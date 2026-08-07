#!/usr/bin/env python3
"""Run PW-0128's authenticated legacy-accelerator physical ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

try:
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
PW0112_SHA256 = "e93d930549ee9fe761d7fc98bf59642088b3eb9f41c712968f8df26d5b2c8b98"
PW0127_SHA256 = "6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140"
EXPECTED_MACS = {
    "attention_projections": 4_482_662_400,
    "dense_layer0_mlp": 201_326_592,
    "routers": 49_283_072,
    "selected_experts": 9_462_349_824,
    "lm_head": 624_951_296,
}
CPU_PEAK_FLOPS = 1.152e12
PCIE3_X16_BYTES_PER_SECOND = 15.754e9
PREFILL_POSITIONS = 8_000
TTFT_LIMIT_SECONDS = 15.0


def configuration_ceiling(
    name: str,
    gpu_count: int,
    gpu_peak_flops_each: float,
    mandatory_macs_per_position: int,
) -> dict:
    if not name or gpu_count <= 0 or gpu_peak_flops_each <= 0 or mandatory_macs_per_position <= 0:
        raise ValueError("accelerator configuration inputs must be positive")
    operations_per_position = 2 * mandatory_macs_per_position
    gpu_peak = gpu_count * gpu_peak_flops_each
    combined_peak = CPU_PEAK_FLOPS + gpu_peak
    prefill_floor = PREFILL_POSITIONS * operations_per_position / combined_peak
    return {
        "name": name,
        "gpu_count": gpu_count,
        "gpu_peak_flops_each": gpu_peak_flops_each,
        "gpu_peak_flops_total": gpu_peak,
        "granted_cpu_peak_flops": CPU_PEAK_FLOPS,
        "impossible_combined_peak_flops": combined_peak,
        "mandatory_operations_per_position": operations_per_position,
        "ordinary_decode_compute_tps_ceiling": combined_peak / operations_per_position,
        "required_fraction_of_peak_at_34_3_tps": operations_per_position * 34.3 / combined_peak,
        "required_fraction_of_peak_at_50_tps": operations_per_position * 50.0 / combined_peak,
        "prefill_positions": PREFILL_POSITIONS,
        "mandatory_matrix_prefill_floor_seconds": prefill_floor,
        "ttft_limit_seconds": TTFT_LIMIT_SECONDS,
        "passes_impossible_prefill_floor": prefill_floor <= TTFT_LIMIT_SECONDS,
    }


def route_window_ceiling(
    q: int,
    suffix_start: int,
    layer_unique_experts: list[int],
    expert_bytes: int,
    accelerator_count: int,
) -> dict:
    if (
        q <= 0
        or suffix_start < 0
        or len(layer_unique_experts) != 47
        or min(layer_unique_experts) <= 0
        or expert_bytes <= 0
        or accelerator_count <= 0
    ):
        raise ValueError("route-window ceiling inputs are invalid")
    total_unique = sum(layer_unique_experts)
    maximum_layer_unique = max(layer_unique_experts)
    transfer_bytes = total_unique * expert_bytes
    aggregate_pcie = accelerator_count * PCIE3_X16_BYTES_PER_SECOND
    transfer_seconds = transfer_bytes / aggregate_pcie
    layer_payload = maximum_layer_unique * expert_bytes
    return {
        "q": q,
        "suffix_start": suffix_start,
        "accelerator_count": accelerator_count,
        "total_layer_expert_records": total_unique,
        "maximum_layer_expert_records": maximum_layer_unique,
        "source_expert_transfer_bytes": transfer_bytes,
        "granted_aggregate_pcie_bytes_per_second": aggregate_pcie,
        "impossible_expert_only_transfer_seconds": transfer_seconds,
        "impossible_perfect_acceptance_tps": q / transfer_seconds,
        "maximum_single_layer_expert_bytes": layer_payload,
        "arena_residency_bytes": {
            str(arenas): arenas * layer_payload for arenas in (1, 2, 3)
        },
        "three_arenas_fit_24_decimal_gb": 3 * layer_payload <= 24_000_000_000,
    }


def _authenticate_sources(route_path: Path, arithmetic_path: Path) -> tuple[dict, dict]:
    if sha256_file(route_path) != PW0112_SHA256:
        raise ValueError("PW-0112 analysis hash mismatch")
    if sha256_file(arithmetic_path) != PW0127_SHA256:
        raise ValueError("PW-0127 arithmetic report hash mismatch")
    route = json.loads(route_path.read_text())
    arithmetic = json.loads(arithmetic_path.read_text())
    if (
        route.get("schema_version") != 1
        or route.get("evidence_class") != "pw0112_wide_teacher_forced_route_economics"
        or route.get("revision") != REVISION
        or route.get("routed_layers") != 47
        or route.get("top_k") != 8
        or route.get("expert_bytes") != 25_171_968
        or route.get("performance_claim") is not None
        or arithmetic.get("schema_version") != 1
        or arithmetic.get("evidence_class") != "pw0127_r720_cpu_arithmetic_ceiling"
        or arithmetic.get("revision") != REVISION
        or arithmetic.get("mandatory_macs_by_category") != EXPECTED_MACS
        or arithmetic.get("accepted_tokens") != 0
        or arithmetic.get("A") != 0
        or arithmetic.get("performance_claim") is not None
        or arithmetic.get("decision") != "reject_cpu_only_dual_e5_2680v2_for_prismwing_50"
    ):
        raise ValueError("PW-0128 source authority mismatch")
    return route, arithmetic


def run(
    route_path: Path,
    arithmetic_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    route, arithmetic = _authenticate_sources(route_path, arithmetic_path)
    safety.checkpoint("source_manifests_authenticated")

    mandatory_macs = sum(EXPECTED_MACS.values())
    configurations = [
        configuration_ceiling("one_tesla_m40_24gb", 1, 7e12, mandatory_macs),
        configuration_ceiling("two_tesla_m40_24gb", 2, 7e12, mandatory_macs),
        configuration_ceiling("one_tesla_p40_24gb", 1, 12e12, mandatory_macs),
    ]
    route_ceilings: dict[str, dict[str, list[dict]]] = {}
    for accelerator_count in (1, 2):
        by_width: dict[str, list[dict]] = {}
        for q in (94, 137):
            source_width = route["sliding_widths"][str(q)]
            windows = []
            for source_window in source_width["windows"]:
                windows.append(
                    route_window_ceiling(
                        q,
                        int(source_window["suffix_start"]),
                        [int(value) for value in source_window["layer_unique_experts"]],
                        int(route["expert_bytes"]),
                        accelerator_count,
                    )
                )
            if len(windows) != int(source_width["window_count"]):
                raise ValueError(f"q={q} route window count mismatch")
            by_width[str(q)] = windows
        route_ceilings[str(accelerator_count)] = by_width
        safety.checkpoint(f"route_ceilings_{accelerator_count}_accelerators")

    if any(row["passes_impossible_prefill_floor"] for row in configurations):
        raise ValueError("PW-0128 expected legacy direct-FP32 prefill rejection absent")
    if not all(
        row["three_arenas_fit_24_decimal_gb"]
        for counts in route_ceilings.values()
        for windows in counts.values()
        for row in windows
    ):
        raise ValueError("PW-0128 unexpected layer-arena capacity failure")

    safety.release_checkpoint(
        "source_reports_released",
        ["PW-0112 route analysis", "PW-0127 arithmetic report"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0128_legacy_24gib_accelerator_ceiling",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "pw0112_analysis_sha256": PW0112_SHA256,
            "pw0127_arithmetic_sha256": PW0127_SHA256,
        },
        "mandatory_macs_by_category": EXPECTED_MACS,
        "mandatory_macs_per_position": mandatory_macs,
        "configurations": configurations,
        "route_ceilings": route_ceilings,
        "market_and_platform": {
            "sold_r720_price_usd": 303.75,
            "observed_m40_card_only_price_usd": 150.0,
            "one_m40_subtotal_usd": 453.75,
            "two_m40_subtotal_usd": 603.75,
            "complete_bom_proven": False,
            "missing_bom_items": [
                "GPU enablement kit and low-profile CPU heatsinks",
                "redundant 1100-W power supplies if absent",
                "boot storage",
                "networking and M1 adapter if absent",
                "shipping and tax",
                "supported cooling environment",
            ],
            "m40_compute_capability": "5.2",
            "last_supported_cuda_toolkit_family": "12.x",
        },
        "gates_passed": False,
        "decision": "reject_named_legacy_direct_fp32_configs_on_8k_ttft",
        "safety_snapshots": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "omitted_work": [
            "attention scores and value aggregation",
            "KV reads and writes",
            "RMSNorm, RoPE, softmax, and nonlinearities",
            "FP8 decode, scales, and dynamic quantization",
            "host DRAM and NUMA contention",
            "PCIe protocol loss and dense/static transfers",
            "routing, drafting, correction, rollback, networking, and sampling",
        ],
        "limitations": (
            "analytical direct-FP32 legacy-accelerator ceiling; not an active complete "
            "BOM, owned machine, CUDA runtime, endpoint result, accepted-token timing, "
            "or rejection of exact codecs and named modified low-bit modes"
        ),
        "platform": platform.platform(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-analysis", required=True, type=Path)
    parser.add_argument("--arithmetic-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.route_analysis,
            arguments.arithmetic_report,
            arguments.output,
            arguments.commit,
        )
        print(json.dumps({"output": str(arguments.output), "decision": result["decision"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
