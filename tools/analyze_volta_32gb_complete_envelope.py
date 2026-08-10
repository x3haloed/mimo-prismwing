#!/usr/bin/env python3
"""Run PW-0161's authenticated 32-GB Volta system envelope."""

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
PW0155_SHA256 = "226603fb2b44e1162a038f51bae47520238150f3b26e39e1cf33c7420b88b064"
PW0158_SHA256 = "3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315"
PW0159_SHA256 = "945079702501f990e2cdd40a326b09fad0f2bb71b3f9615c8114c0bbd71590c2"
NVIDIA_V100_HTML_SHA256 = "39557823ad6871fbfe5afd7d572d5192c754e27feef90c8cc092e562f59b4f4d"
V100_PRODUCT_BRIEF_SHA256 = "7e2a80764520d744ae146ec276655a6359ecd2bcd83feaba802cb29efcedadee"
MARKET_SHA256 = "c7f378c65bd2c24633ccce238f0dcaffc1731de8f670a34721d0fb50cc3c010c"

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
GPU_BOARD_WATTS = 250
PSU_12V_WATTS = 732

CARD_SPECS = (
    {
        "id": "v100_pcie_32gb",
        "official_fp32_flops_per_second": 14_000_000_000_000,
        "advertised_deep_learning_flops_per_second": 112_000_000_000_000,
        "hbm_bandwidth_bytes_per_second": 900_000_000_000,
    },
    {
        "id": "v100s_pcie_32gb",
        "official_fp32_flops_per_second": 16_400_000_000_000,
        "advertised_deep_learning_flops_per_second": 130_000_000_000_000,
        "hbm_bandwidth_bytes_per_second": 1_134_000_000_000,
    },
)


def arithmetic_scenarios() -> list[dict]:
    rows = []
    for spec in CARD_SPECS:
        modes = (
            ("direct_fp32_control", spec["official_fp32_flops_per_second"]),
            ("l3_unvalidated_fp16_tensor_ceiling", spec["advertised_deep_learning_flops_per_second"]),
        )
        rendered_modes = []
        for mode, gpu_rate in modes:
            combined = gpu_rate + EPYC_IMPOSSIBLE_PEAK_FLOPS
            floor = MANDATORY_TOTAL_FLOPS / combined
            rendered_modes.append(
                {
                    "mode": mode,
                    "gpu_nameplate_flops_per_second": gpu_rate,
                    "granted_concurrent_epyc_flops_per_second": EPYC_IMPOSSIBLE_PEAK_FLOPS,
                    "impossible_combined_flops_per_second": combined,
                    "mandatory_matrix_plus_attention_flops": MANDATORY_TOTAL_FLOPS,
                    "floor_seconds": floor,
                    "remaining_1m_ttft_seconds": TTFT_SECONDS - floor,
                    "passes_1m_ttft_arithmetic_gate": floor <= TTFT_SECONDS,
                }
            )
        rows.append(
            {
                **spec,
                "modes": rendered_modes,
                "ordinary_dense_candidate_survives_favorable_arithmetic": rendered_modes[1][
                    "passes_1m_ttft_arithmetic_gate"
                ],
            }
        )
    return rows


def capacity_ledger() -> dict:
    exact_control_total = EXACT_BF16_1M_KV_BYTES + THREE_ARENA_BYTES + NON_ROUTED_SOURCE_BYTES
    optimistic_available = HBM_BYTES - EXACT_BF16_1M_KV_BYTES - THREE_ARENA_BYTES
    slots, tail = divmod(optimistic_available, EXPERT_BYTES)
    return {
        "hbm_decimal_bytes": HBM_BYTES,
        "exact_bf16_1m_kv_bytes": EXACT_BF16_1M_KV_BYTES,
        "three_maximum_layer_arenas_bytes": THREE_ARENA_BYTES,
        "non_routed_source_tensor_bytes": NON_ROUTED_SOURCE_BYTES,
        "full_source_resident_control_bytes": exact_control_total,
        "full_source_resident_control_over_hbm_bytes": max(0, exact_control_total - HBM_BYTES),
        "full_source_resident_control_fits": exact_control_total <= HBM_BYTES,
        "optimistic_non_routed_tensors_streamed_for_free": True,
        "optimistic_available_for_complete_experts_bytes": optimistic_available,
        "optimistic_complete_expert_slots": slots,
        "optimistic_expert_cache_bytes": slots * EXPERT_BYTES,
        "optimistic_tail_bytes": tail,
    }


def cost_ledger(market: dict) -> list[dict]:
    if market.get("evidence_class") != "dated_semantic_listing_transcription_not_purchase_authority":
        raise ValueError("market evidence class mismatch")
    listings = market.get("listings")
    if not isinstance(listings, list) or len(listings) != 2:
        raise ValueError("expected exactly two market listings")
    by_id = {row.get("id"): row for row in listings}
    if set(by_id) != {spec["id"] for spec in CARD_SPECS}:
        raise ValueError("market candidate identity mismatch")
    rows = []
    for spec in CARD_SPECS:
        listing = by_id[spec["id"]]
        if (
            listing.get("active") is not True
            or listing.get("condition") != "used"
            or listing.get("memory_gb") != 32
            or listing.get("seller_positive_percent", 0) <= 0
            or listing.get("tax_known") is not False
        ):
            raise ValueError(f"unqualified market listing: {spec['id']}")
        before_tax = listing["observed_price_usd"] + listing["observed_shipping_usd"]
        rows.append(
            {
                "id": spec["id"],
                "active_used_32gb_listing": True,
                "card_subtotal_before_tax_usd": before_tax,
                "card_alone_over_complete_cap_usd": max(0.0, before_tax - 500.0),
                "card_alone_within_complete_cap_before_tax": before_tax <= 500.0,
                "tax_known": False,
                "cable_cooling_storage_cost_included": False,
                "captured_procurement_branch_survives": False,
                "listing": listing,
            }
        )
    return rows


def power_ledger() -> dict:
    subtotal = GPU_BOARD_WATTS + EPYC_TDP_WATTS
    return {
        "gpu_board_watts": GPU_BOARD_WATTS,
        "epyc_tdp_watts": EPYC_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": subtotal,
        "psu_combined_12v_watts": PSU_12V_WATTS,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": PSU_12V_WATTS - subtotal,
        "formal_project_wall_cap_watts": 1_000,
        "single_card_nameplate_is_installation_proof": False,
        "required_auxiliary_connector": "one CPU 8-pin or NVIDIA 030-0571-000 dongle",
        "passive_card_requires_forced_airflow": True,
    }


def _authenticate(paths: dict[str, Path]) -> tuple[dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "pw0127": PW0127_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0155": PW0155_SHA256,
        "pw0158": PW0158_SHA256,
        "pw0159": PW0159_SHA256,
        "nvidia_html": NVIDIA_V100_HTML_SHA256,
        "product_brief": V100_PRODUCT_BRIEF_SHA256,
        "market": MARKET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0161 source hash mismatch: {name}")
    if paths["product_brief"].read_bytes()[:5] != b"%PDF-":
        raise ValueError("V100 product brief is not a PDF")
    target = paths["target"].read_text(errors="strict")
    if any(value not in target for value in ("USD $500 total", "begin generation within 30 minutes")):
        raise ValueError("TARGET authority mismatch")
    config = json.loads(paths["config"].read_text())
    pw0127 = json.loads(paths["pw0127"].read_text())
    pw0151 = json.loads(paths["pw0151"].read_text())
    pw0155 = json.loads(paths["pw0155"].read_text())
    pw0158 = json.loads(paths["pw0158"].read_text())
    pw0159 = json.loads(paths["pw0159"].read_text())
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
        or pw0151.get("cpu_only", {}).get("impossible_peak_fp32") != EPYC_IMPOSSIBLE_PEAK_FLOPS
    ):
        raise ValueError("owned EPYC ceiling authority mismatch")
    if (
        pw0155.get("evidence_class") != "pw0155_owned_epyc_installable_bom_prerequisite"
        or pw0155.get("power", {}).get("psu_combined_12v_watts") != PSU_12V_WATTS
        or pw0155.get("power", {}).get("epyc_7351p_tdp_watts") != EPYC_TDP_WATTS
    ):
        raise ValueError("owned host authority mismatch")
    if (
        pw0158.get("evidence_class") != "pw0158_million_context_two_p100_attention_ceiling"
        or pw0158.get("attention_work_ledger", {}).get("mandatory_attention_flops")
        != MANDATORY_ATTENTION_FLOPS
    ):
        raise ValueError("attention authority mismatch")
    fp16 = pw0159.get("one_million_arithmetic_ceilings", {}).get("l3_fp16_fp16_accumulate", {})
    if (
        pw0159.get("evidence_class") != "pw0159_ampere_12gb_complete_system_envelope"
        or fp16.get("mandatory_matrix_plus_attention_flops") != MANDATORY_TOTAL_FLOPS
    ):
        raise ValueError("complete arithmetic authority mismatch")
    html = paths["nvidia_html"].read_text(errors="strict")
    for value in (
        "V100 for PCIe",
        "V100S for PCIe",
        '<span class="value-spec">112</span> teraFLOPS',
        '<span class="value-spec">130</span> teraFLOPS',
        '<span class="value-spec">32</span> GB HBM2',
        '<span class="value-spec">1134</span> GB/s',
    ):
        if value not in html:
            raise ValueError(f"official V100 specification missing: {value}")
    return market, expected


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    market, source_hashes = _authenticate(paths)
    safety.checkpoint("all_source_evidence_authenticated")
    arithmetic = arithmetic_scenarios()
    capacity = capacity_ledger()
    costs = cost_ledger(market)
    power = power_ledger()
    safety.checkpoint("complete_envelopes_computed")
    standard = next(row for row in arithmetic if row["id"] == "v100_pcie_32gb")
    v100s = next(row for row in arithmetic if row["id"] == "v100s_pcie_32gb")
    standard_tensor = standard["modes"][1]
    v100s_tensor = v100s["modes"][1]
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0161_volta_32gb_complete_system_envelope",
        "revision": REVISION,
        "commit": commit,
        "positions": POSITIONS,
        "source_hashes": source_hashes,
        "arithmetic": arithmetic,
        "capacity": capacity,
        "costs": costs,
        "power": power,
        "decision": {
            "v100_pcie_32gb": "reject_ordinary_dense_1m_arithmetic_even_at_l3_tensor_peak",
            "v100s_pcie_32gb": "reject_captured_procurement_and_retain_price_triggered_l3_only",
            "standard_v100_tensor_shortfall_seconds": -standard_tensor["remaining_1m_ttft_seconds"],
            "v100s_tensor_idealized_remaining_seconds": v100s_tensor["remaining_1m_ttft_seconds"],
            "purchase_authorized": False,
            "runtime_implementation_authorized": False,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "authenticated analytical nameplate and moving-market envelope; not installed hardware, "
            "measured CUDA, sustained storage, exact FP16 fidelity, delivered BOM, endpoint, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official specification text", "market transcription", "prior manifests"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("pw0127", type=Path)
    parser.add_argument("pw0151", type=Path)
    parser.add_argument("pw0155", type=Path)
    parser.add_argument("pw0158", type=Path)
    parser.add_argument("pw0159", type=Path)
    parser.add_argument("nvidia_html", type=Path)
    parser.add_argument("product_brief", type=Path)
    parser.add_argument("market", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ("target", "config", "pw0127", "pw0151", "pw0155", "pw0158", "pw0159", "nvidia_html", "product_brief", "market")
    manifest = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
