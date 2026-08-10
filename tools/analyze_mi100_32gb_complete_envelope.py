#!/usr/bin/env python3
"""Run PW-0163's authenticated MI100 complete-system envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

try:
    from tools.analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from tools.analyze_pw0116_corpus import sha256_file
    from tools.host_safety import HostSafetyMonitor
    from tools.openrouter_reference import atomic_write_new, canonical_json
except ModuleNotFoundError:
    from analyze_owned_epyc_companion_envelope import authenticate_implementation_commit
    from analyze_pw0116_corpus import sha256_file
    from host_safety import HostSafetyMonitor
    from openrouter_reference import atomic_write_new, canonical_json


REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
CONFIG_SHA256 = "292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587"
PW0127_SHA256 = "6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140"
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
PW0158_SHA256 = "3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315"
PW0161_SHA256 = "fc438d593d8ac99be3cc426496feb830256ffc48c75d58fc8bb9d6b09a2c6c8f"
AMD_PRODUCT_SHA256 = "9d0b74dc18ac8afcced3a9efbca17f77f6fd4b148c6823c11a5f479d8f9cbcc6"
AMD_BROCHURE_SHA256 = "ad383b0c0d2bcb8c719ddcf09ed5a4d7a0afeb901c3b51bb2490fa3e65e6dc2e"
ROCM_REQUIREMENTS_SHA256 = "dead3ad053cde897c83aa33d58f096b1a9b25878abbe82dbcd5206c3f86d3772"
MARKET_SHA256 = "bfdc3bcd99685518f810f4d7f5caaa7d6563511cc1f62dfed3f317f2d0bd9022"

POSITIONS = 1_000_000
TTFT_SECONDS = 1_800.0
MANDATORY_TOTAL_FLOPS = 214_165_790_024_007_680
MANDATORY_MACS_PER_TOKEN = 14_820_573_184
MANDATORY_ATTENTION_FLOPS = 184_524_643_656_007_680
EPYC_IMPOSSIBLE_PEAK_FLOPS = 742_400_000_000
HBM_BYTES = 32_000_000_000
EXACT_BF16_1M_KV_BYTES = 23_065_559_040
THREE_ARENA_BYTES = 2_340_993_024
NON_ROUTED_SOURCE_BYTES = 12_814_555_472
EXPERT_BYTES = 25_171_968
EPYC_TDP_WATTS = 170
GPU_BOARD_WATTS = 300
PSU_12V_WATTS = 732

MODES = (
    ("direct_fp32_control", 23_100_000_000_000, "L0/L1 numerical control"),
    ("fp32_matrix_favorable_ceiling", 46_100_000_000_000, "favorable matrix diagnostic"),
    ("bf16_matrix_source_oriented_ceiling", 92_300_000_000_000, "source-oriented upper bound"),
    ("l3_fp16_matrix_ceiling", 184_600_000_000_000, "L3 accumulation diagnostic"),
)


def arithmetic_ledger() -> list[dict]:
    rows = []
    for mode, gpu_rate, exactness in MODES:
        combined = gpu_rate + EPYC_IMPOSSIBLE_PEAK_FLOPS
        floor = MANDATORY_TOTAL_FLOPS / combined
        rows.append(
            {
                "mode": mode,
                "exactness": exactness,
                "gpu_nameplate_flops_per_second": gpu_rate,
                "granted_concurrent_epyc_flops_per_second": EPYC_IMPOSSIBLE_PEAK_FLOPS,
                "impossible_combined_flops_per_second": combined,
                "mandatory_matrix_plus_attention_flops": MANDATORY_TOTAL_FLOPS,
                "floor_seconds": floor,
                "remaining_1m_ttft_seconds": TTFT_SECONDS - floor,
                "passes_1m_ttft_arithmetic_gate": floor <= TTFT_SECONDS,
            }
        )
    return rows


def capacity_ledger() -> dict:
    full = EXACT_BF16_1M_KV_BYTES + THREE_ARENA_BYTES + NON_ROUTED_SOURCE_BYTES
    optimistic_available = HBM_BYTES - EXACT_BF16_1M_KV_BYTES - THREE_ARENA_BYTES
    slots, tail = divmod(optimistic_available, EXPERT_BYTES)
    return {
        "hbm_decimal_bytes": HBM_BYTES,
        "exact_bf16_1m_kv_bytes": EXACT_BF16_1M_KV_BYTES,
        "three_maximum_layer_arenas_bytes": THREE_ARENA_BYTES,
        "non_routed_source_tensor_bytes": NON_ROUTED_SOURCE_BYTES,
        "full_source_resident_control_bytes": full,
        "full_source_resident_control_over_hbm_bytes": max(0, full - HBM_BYTES),
        "full_source_resident_control_fits": full <= HBM_BYTES,
        "optimistic_non_routed_tensors_streamed_for_free": True,
        "optimistic_available_for_complete_experts_bytes": optimistic_available,
        "optimistic_complete_expert_slots": slots,
        "optimistic_expert_cache_bytes": slots * EXPERT_BYTES,
        "optimistic_tail_bytes": tail,
    }


def power_ledger() -> dict:
    subtotal = GPU_BOARD_WATTS + EPYC_TDP_WATTS
    return {
        "gpu_board_tdp_watts": GPU_BOARD_WATTS,
        "epyc_tdp_watts": EPYC_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": subtotal,
        "psu_combined_12v_watts": PSU_12V_WATTS,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": PSU_12V_WATTS - subtotal,
        "formal_project_wall_cap_watts": 1_000,
        "passively_cooled": True,
        "full_height_dual_slot_length_inches": 10.5,
        "single_card_nameplate_is_installation_proof": False,
        "auxiliary_cable_requirement_authenticated": False,
        "forced_airflow_required_before_load": True,
    }


def cost_ledger(market: dict) -> dict:
    if (
        market.get("evidence_class")
        != "dated_semantic_listing_transcription_not_purchase_authority"
        or market.get("listing_id") != "285796378466"
        or market.get("active_at_capture") is not True
        or market.get("condition") != "used"
        or market.get("observed_price_usd") != 999.0
        or market.get("observed_shipping_usd") != 0.0
        or market.get("tax_known") is not False
    ):
        raise ValueError("MI100 market authority mismatch")
    subtotal = market["observed_price_usd"] + market["observed_shipping_usd"]
    return {
        "active_used_32gb_listing": True,
        "card_subtotal_before_tax_usd": subtotal,
        "card_alone_over_complete_cap_usd": subtotal - 500.0,
        "card_alone_within_complete_cap_before_tax": subtotal <= 500.0,
        "tax_known": False,
        "cable_cooling_storage_os_cost_included": False,
        "captured_procurement_branch_survives": False,
        "listing": market,
    }


def _authenticate(paths: dict[str, Path]) -> tuple[dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "pw0127": PW0127_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0158": PW0158_SHA256,
        "pw0161": PW0161_SHA256,
        "amd_product": AMD_PRODUCT_SHA256,
        "amd_brochure": AMD_BROCHURE_SHA256,
        "rocm_requirements": ROCM_REQUIREMENTS_SHA256,
        "market": MARKET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0163 source hash mismatch: {name}")
    if paths["amd_brochure"].read_bytes()[:5] != b"%PDF-":
        raise ValueError("AMD MI100 brochure is not a PDF")
    target = paths["target"].read_text(errors="strict")
    if "1M-token smoke case" not in target or "USD $500 total" not in target:
        raise ValueError("TARGET authority mismatch")
    config = json.loads(paths["config"].read_text())
    pw0127 = json.loads(paths["pw0127"].read_text())
    pw0151 = json.loads(paths["pw0151"].read_text())
    pw0158 = json.loads(paths["pw0158"].read_text())
    pw0161 = json.loads(paths["pw0161"].read_text())
    market = json.loads(paths["market"].read_text())
    if (
        config.get("max_position_embeddings", 0) < POSITIONS
        or pw0127.get("revision") != REVISION
        or pw0127.get("arithmetic_ceiling", {}).get("mandatory_macs_per_token")
        != MANDATORY_MACS_PER_TOKEN
    ):
        raise ValueError("model arithmetic authority mismatch")
    if (
        pw0151.get("evidence_class") != "pw0151_owned_epyc_companion_envelope"
        or pw0151.get("cpu_only", {}).get("impossible_peak_fp32")
        != EPYC_IMPOSSIBLE_PEAK_FLOPS
        or pw0151.get("owned_hardware", {}).get("psu", {}).get("combined_12v_watts")
        != PSU_12V_WATTS
    ):
        raise ValueError("owned host authority mismatch")
    if (
        pw0158.get("evidence_class") != "pw0158_million_context_two_p100_attention_ceiling"
        or pw0158.get("attention_work_ledger", {}).get("mandatory_attention_flops")
        != MANDATORY_ATTENTION_FLOPS
    ):
        raise ValueError("attention authority mismatch")
    if (
        pw0161.get("evidence_class") != "pw0161_volta_32gb_complete_system_envelope"
        or pw0161.get("positions") != POSITIONS
        or pw0161.get("arithmetic", [{}])[0].get("modes", [{}, {}])[1].get(
            "mandatory_matrix_plus_attention_flops"
        )
        != MANDATORY_TOTAL_FLOPS
    ):
        raise ValueError("complete arithmetic authority mismatch")
    product = paths["amd_product"].read_text(errors="strict")
    for value in (
        "184.6 TFLOPs",
        "92.3 TFLOPs",
        "300W Peak",
        "32 GB",
        "PCIe® 3.0 x16",
        "Passive",
    ):
        if value not in product:
            raise ValueError(f"official MI100 specification missing: {value}")
    rocm = paths["rocm_requirements"].read_text(errors="strict")
    required_support = (
        "AMD Instinct MI100 GPU only supports Ubuntu 24.04.3, Ubuntu 22.04.5, "
        "RHEL 10.0, RHEL 9.6, RHEL 9.4, RHEL 8.10, and SLES 15 SP7."
    )
    if required_support not in rocm or "Debian 13 is supported only" not in rocm:
        raise ValueError("ROCm MI100 operating-system authority mismatch")
    return market, expected


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    market, source_hashes = _authenticate(paths)
    safety.checkpoint("all_source_evidence_authenticated")
    arithmetic = arithmetic_ledger()
    capacity = capacity_ledger()
    power = power_ledger()
    cost = cost_ledger(market)
    safety.checkpoint("complete_envelope_computed")
    bf16 = next(row for row in arithmetic if row["mode"] == "bf16_matrix_source_oriented_ceiling")
    fp16 = next(row for row in arithmetic if row["mode"] == "l3_fp16_matrix_ceiling")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0163_mi100_32gb_complete_system_envelope",
        "revision": REVISION,
        "commit": commit,
        "positions": POSITIONS,
        "source_hashes": source_hashes,
        "arithmetic": arithmetic,
        "capacity": capacity,
        "power": power,
        "cost": cost,
        "software_support": {
            "gpu": "AMD Instinct MI100 gfx908",
            "captured_rocm_requirements": "7.1.0",
            "owned_host_current_os": "Debian 13",
            "current_owned_host_os_supported_for_mi100": False,
            "supported_os_installation_is_deployment_prerequisite_not_permanent_rejection": True,
        },
        "decision": {
            "source_oriented_bf16": "reject_ordinary_dense_1m_arithmetic_at_impossible_peak",
            "l3_fp16": "retain_price_triggered_numerical_hypothesis_only",
            "captured_procurement": "reject_card_alone_over_complete_cap",
            "bf16_shortfall_seconds": -bf16["remaining_1m_ttft_seconds"],
            "fp16_idealized_remaining_seconds": fp16["remaining_1m_ttft_seconds"],
            "purchase_authorized": False,
            "runtime_implementation_authorized": False,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "authenticated analytical nameplate, software-support, and moving-market envelope; "
            "not installed hardware, measured HIP, source-BF16 attainment, FP16 fidelity, "
            "delivered BOM, endpoint, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official AMD specifications", "ROCm requirements", "market transcription", "prior manifests"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "target",
        "config",
        "pw0127",
        "pw0151",
        "pw0158",
        "pw0161",
        "amd_product",
        "amd_brochure",
        "rocm_requirements",
        "market",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        "target",
        "config",
        "pw0127",
        "pw0151",
        "pw0158",
        "pw0161",
        "amd_product",
        "amd_brochure",
        "rocm_requirements",
        "market",
    )
    manifest = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
