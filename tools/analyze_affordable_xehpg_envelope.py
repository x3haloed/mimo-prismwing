#!/usr/bin/env python3
"""Run PW-0167's authenticated affordable-Xe-HPG arithmetic envelope."""

from __future__ import annotations

import argparse
import html
import json
import platform
from pathlib import Path
import re
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
PW0155_SHA256 = "226603fb2b44e1162a038f51bae47520238150f3b26e39e1cf33c7420b88b064"
PW0158_SHA256 = "3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315"
PRODUCT_SHA256 = "b4691de4514c938e8c0d386a6d1fa6583b96479b4c11ad4aed2726ac1527eccd"
ARCHITECTURE_SHA256 = "0bc5fddeb681428ce63a8972b6b5eb53a002ea4e7eb6541fab52898f62771d0b"
QUICK_START_SHA256 = "c7402269d97f457527b7de660dc87adbd62e1284ce0afc57a97fc246f0fb9133"
ONEAPI_SHA256 = "58c984149a1e39359c0211826da1a288a610bc97e0b2646668b988d79cb8cec2"
REBAR_FAQ_SHA256 = "4441674b68b105dcd82df6bfb938e7781af6e84aa45e0ac566b3ff7ba9b36794"
HOST_INVENTORY_SHA256 = "b8b84a557eabea9c1781186357cf0f2f4fbf75ca7c8a74656beff26fac15978b"
MARKET_SHA256 = "e1774ad682ed47ee897831df47719164850dc0ba6c8136e9c3821cd708ee4385"

POSITIONS = 1_000_000
TTFT_SECONDS = 1_800.0
MANDATORY_MACS_PER_TOKEN = 14_820_573_184
MANDATORY_MATRIX_FLOPS = MANDATORY_MACS_PER_TOKEN * 2 * POSITIONS
MANDATORY_ATTENTION_FLOPS = 184_524_643_656_007_680
MANDATORY_TOTAL_FLOPS = MANDATORY_MATRIX_FLOPS + MANDATORY_ATTENTION_FLOPS
EPYC_IMPOSSIBLE_PEAK_FLOPS = 742_400_000_000

INT8_XMX_TOPS = 262_000_000_000_000
INT8_OPS_PER_XE_CORE_CYCLE = 4_096
BF16_OPS_PER_XE_CORE_CYCLE = 2_048
VRAM_BYTES = 16_000_000_000
EXACT_BF16_1M_KV_BYTES = 23_065_559_040
THREE_ARENA_BYTES = 2_340_993_024
NON_ROUTED_SOURCE_BYTES = 12_814_555_472
GPU_BOARD_WATTS = 225
EPYC_TDP_WATTS = 170
PSU_12V_WATTS = 732


def normalized_html(path: Path) -> str:
    raw = html.unescape(path.read_text(errors="strict"))
    return " ".join(re.sub(r"<[^>]+>", " ", raw).split())


def derived_bf16_peak() -> int:
    return INT8_XMX_TOPS * BF16_OPS_PER_XE_CORE_CYCLE // INT8_OPS_PER_XE_CORE_CYCLE


def arithmetic_ledger() -> dict:
    bf16_peak = derived_bf16_peak()
    combined = bf16_peak + EPYC_IMPOSSIBLE_PEAK_FLOPS
    floor = MANDATORY_TOTAL_FLOPS / combined
    return {
        "official_a770_peak_int8_xmx_operations_per_second": INT8_XMX_TOPS,
        "official_xehpg_int8_operations_per_xe_core_cycle": INT8_OPS_PER_XE_CORE_CYCLE,
        "official_xehpg_bf16_operations_per_xe_core_cycle": BF16_OPS_PER_XE_CORE_CYCLE,
        "derived_bf16_f32acc_operations_per_second": bf16_peak,
        "granted_concurrent_epyc_flops_per_second": EPYC_IMPOSSIBLE_PEAK_FLOPS,
        "impossible_combined_flops_per_second": combined,
        "mandatory_matrix_flops": MANDATORY_MATRIX_FLOPS,
        "mandatory_attention_flops": MANDATORY_ATTENTION_FLOPS,
        "mandatory_matrix_plus_attention_flops": MANDATORY_TOTAL_FLOPS,
        "floor_seconds": floor,
        "remaining_1m_ttft_seconds": TTFT_SECONDS - floor,
        "passes_1m_ttft_arithmetic_gate": floor <= TTFT_SECONDS,
        "derivation": (
            "Intel specifies 4096 INT8 versus 2048 FP16/BF16 XMX operations per "
            "Xe-HPG core-cycle, so the source-oriented BF16 ceiling is half of "
            "A770's official dense INT8 XMX TOPS"
        ),
    }


def capacity_ledger() -> dict:
    full = EXACT_BF16_1M_KV_BYTES + THREE_ARENA_BYTES + NON_ROUTED_SOURCE_BYTES
    return {
        "vram_decimal_bytes": VRAM_BYTES,
        "exact_bf16_1m_kv_bytes": EXACT_BF16_1M_KV_BYTES,
        "kv_alone_over_vram_bytes": EXACT_BF16_1M_KV_BYTES - VRAM_BYTES,
        "kv_alone_fits": EXACT_BF16_1M_KV_BYTES <= VRAM_BYTES,
        "three_maximum_layer_arenas_bytes": THREE_ARENA_BYTES,
        "non_routed_source_tensor_bytes": NON_ROUTED_SOURCE_BYTES,
        "full_source_resident_control_bytes": full,
        "full_source_resident_control_over_vram_bytes": full - VRAM_BYTES,
        "full_source_resident_control_fits": full <= VRAM_BYTES,
        "layer_major_or_host_storage_streaming_required": True,
    }


def power_ledger() -> dict:
    subtotal = GPU_BOARD_WATTS + EPYC_TDP_WATTS
    return {
        "gpu_total_board_power_watts": GPU_BOARD_WATTS,
        "epyc_tdp_watts": EPYC_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": subtotal,
        "psu_combined_12v_watts": PSU_12V_WATTS,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": PSU_12V_WATTS - subtotal,
        "formal_project_wall_cap_watts": 1_000,
        "single_card_nameplate_is_installation_proof": False,
    }


def _authenticate(paths: dict[str, Path]) -> dict[str, str]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "pw0127": PW0127_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0155": PW0155_SHA256,
        "pw0158": PW0158_SHA256,
        "product": PRODUCT_SHA256,
        "architecture": ARCHITECTURE_SHA256,
        "quick_start": QUICK_START_SHA256,
        "oneapi": ONEAPI_SHA256,
        "rebar_faq": REBAR_FAQ_SHA256,
        "host_inventory": HOST_INVENTORY_SHA256,
        "market": MARKET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0167 source hash mismatch: {name}")

    target = paths["target"].read_text(errors="strict")
    config = json.loads(paths["config"].read_text())
    pw0127 = json.loads(paths["pw0127"].read_text())
    pw0151 = json.loads(paths["pw0151"].read_text())
    pw0155 = json.loads(paths["pw0155"].read_text())
    pw0158 = json.loads(paths["pw0158"].read_text())
    if "1M-token smoke case" not in target or "USD $500 total" not in target:
        raise ValueError("TARGET authority mismatch")
    if (
        config.get("max_position_embeddings", 0) < POSITIONS
        or pw0127.get("arithmetic_ceiling", {}).get("mandatory_macs_per_token")
        != MANDATORY_MACS_PER_TOKEN
        or pw0151.get("cpu_only", {}).get("impossible_peak_fp32")
        != EPYC_IMPOSSIBLE_PEAK_FLOPS
        or pw0151.get("owned_hardware", {}).get("psu", {}).get("combined_12v_watts")
        != PSU_12V_WATTS
        or pw0155.get("topology", {}).get("logical_lane_topology_supported") is not True
        or pw0158.get("attention_work_ledger", {}).get("mandatory_attention_flops")
        != MANDATORY_ATTENTION_FLOPS
    ):
        raise ValueError("model, host, or arithmetic authority mismatch")

    product = normalized_html(paths["product"])
    for value in (
        "Intel® Arc™ A770 Graphics",
        "Intel® X e Matrix Extensions (Intel® XMX) Engines 512",
        "GPU Peak TOPS (Int8)",
        "262",
        "225 W",
        "16 GB",
        "560 GB/s",
        "Up to PCI Express 4.0 x16",
        "oneAPI Support",
    ):
        if value not in product:
            raise ValueError(f"official A770 product fact missing: {value}")
    architecture = normalized_html(paths["architecture"])
    for value in (
        "X e -HPG (Arc A770)",
        "4096 int8 and 2048 FP16/BF16 operations/cycle",
        "It powers the Intel ® Arc GPUs",
    ):
        if value not in architecture:
            raise ValueError(f"official Xe-HPG architecture fact missing: {value}")
    quick_start = normalized_html(paths["quick_start"])
    for value in (
        "Resizable BAR or Smart Access Memory must be enabled for optimal performance",
        "there may be performance or stability issues",
    ):
        if value not in quick_start:
            raise ValueError(f"official Arc platform prerequisite missing: {value}")
    oneapi = normalized_html(paths["oneapi"])
    for value in ("All other client GPU platforms", "Ubuntu LTS 24.04", "Ubuntu LTS 26.04"):
        if value not in oneapi:
            raise ValueError(f"official oneAPI client-GPU requirement missing: {value}")
    rebar = normalized_html(paths["rebar_faq"])
    for value in (
        "Could H11SSL-i support Resizable BAR support?",
        "it is not supported on H11 motherboards",
        "AMD EPYC 7001/7002 series",
    ):
        if value not in rebar:
            raise ValueError(f"official H11SSL ReBAR fact missing: {value}")
    inventory = paths["host_inventory"].read_text(errors="strict")
    if "6.12.100+deb13-amd64" not in inventory or "AMD EPYC 7351P" not in inventory:
        raise ValueError("owned host inventory mismatch")
    market = json.loads(paths["market"].read_text())
    rows = market.get("observations", [])
    if (
        market.get("evidence_class")
        != "dated_semantic_transcription_after_direct_ebay_fetch_403"
        or [(row.get("item_id"), row.get("sold_price_usd"), row.get("availability")) for row in rows]
        != [("358221920938", 245.0, "sold"), ("137194058039", 215.5, "sold")]
    ):
        raise ValueError("dated sold-market authority mismatch")
    return expected


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    source_hashes = _authenticate(paths)
    safety.checkpoint("all_source_evidence_authenticated")
    arithmetic = arithmetic_ledger()
    capacity = capacity_ledger()
    power = power_ledger()
    safety.checkpoint("complete_envelope_computed")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0167_affordable_xehpg_complete_system_envelope",
        "revision": REVISION,
        "commit": commit,
        "positions": POSITIONS,
        "source_hashes": source_hashes,
        "arithmetic": arithmetic,
        "capacity": capacity,
        "power": power,
        "platform_prerequisites": {
            "owned_h11ssl_native_resizable_bar_supported": False,
            "intel_requires_resizable_bar_for_optimal_arc_performance": True,
            "owned_debian_13_listed_for_oneapi_client_gpu": False,
            "officially_listed_client_gpu_linux": ["Ubuntu 24.04", "Ubuntu 26.04"],
            "installed_a770_performance_or_stability_proven": False,
        },
        "cost": {
            "dated_sold_card_prices_usd": [245.0, 215.5],
            "active_delivered_card_available": False,
            "complete_delivered_bom_proven": False,
            "purchase_authorized": False,
        },
        "decision": {
            "source_oriented_bf16": "retain_arithmetic_survivor_only",
            "arithmetic_headroom_seconds": arithmetic["remaining_1m_ttft_seconds"],
            "next_gate": "active_complete_bom_then_owned_host_rebar_off_on_oneapi_component_microbenchmark",
            "purchase_authorized": False,
            "runtime_implementation_authorized": False,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "authenticated analytical arithmetic, capacity, platform, power, and sold-market "
            "envelope; not an active BOM, installed Arc device, measured oneAPI/ReBAR result, "
            "complete runtime, endpoint, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official Intel and Supermicro pages", "owned-host inventory", "prior manifests"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    names = (
        "target", "config", "pw0127", "pw0151", "pw0155", "pw0158", "product",
        "architecture", "quick_start", "oneapi", "rebar_faq", "host_inventory", "market",
    )
    for name in names:
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        "target", "config", "pw0127", "pw0151", "pw0155", "pw0158", "product",
        "architecture", "quick_start", "oneapi", "rebar_faq", "host_inventory", "market",
    )
    manifest = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
