#!/usr/bin/env python3
"""Run PW-0164's authenticated affordable-Blackwell envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
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
PRODUCT_SHA256 = "238d00c79c20939e5208e1a6507a6949e00e21ab3c3c3cc79d25b97eb0af20fd"
ARCHITECTURE_SHA256 = "906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa"
LAUNCH_SHA256 = "76ca4fce0315435079d72f3725174b704b9b8990b3be7d89591471333a418394"
MARKET_SHA256 = "98400749a4ca60351ff71b0450bf545ee542691051e783426c2c183219774cf6"

POSITIONS = 1_000_000
TTFT_SECONDS = 1_800.0
MANDATORY_TOTAL_FLOPS = 214_165_790_024_007_680
MANDATORY_MACS_PER_TOKEN = 14_820_573_184
MANDATORY_ATTENTION_FLOPS = 184_524_643_656_007_680
EPYC_IMPOSSIBLE_PEAK_FLOPS = 742_400_000_000

CUDA_CORES = 4_608
CUDA_CORES_PER_SM = 128
TENSOR_CORES_PER_SM = 4
BOOST_MHZ = 2_570
REFERENCE_5070_SMS = 48
REFERENCE_5070_BOOST_MHZ = 2_512
REFERENCE_5070_BF16_FP32ACC_FLOPS = 61_700_000_000_000
REFERENCE_5070_FP16_FP16ACC_FLOPS = 123_500_000_000_000
AI_TOPS_NOT_DENSE_BF16 = 759_000_000_000_000

VRAM_BYTES = 16_000_000_000
EXACT_BF16_1M_KV_BYTES = 23_065_559_040
THREE_ARENA_BYTES = 2_340_993_024
NON_ROUTED_SOURCE_BYTES = 12_814_555_472
GPU_BOARD_WATTS = 180
EPYC_TDP_WATTS = 170
PSU_12V_WATTS = 732


def derived_rates() -> dict[str, int]:
    sms = CUDA_CORES // CUDA_CORES_PER_SM
    scale = (sms / REFERENCE_5070_SMS) * (BOOST_MHZ / REFERENCE_5070_BOOST_MHZ)
    return {
        "direct_fp32": CUDA_CORES * 2 * BOOST_MHZ * 1_000_000,
        "bf16_tensor_fp32_accumulate": round(REFERENCE_5070_BF16_FP32ACC_FLOPS * scale),
        "fp16_tensor_fp16_accumulate": round(REFERENCE_5070_FP16_FP16ACC_FLOPS * scale),
    }


def arithmetic_ledger() -> list[dict]:
    rates = derived_rates()
    definitions = (
        ("direct_fp32_control", rates["direct_fp32"], "L0/L1 direct arithmetic control"),
        (
            "bf16_tensor_fp32_accumulate_source_oriented_ceiling",
            rates["bf16_tensor_fp32_accumulate"],
            "source-oriented dense Tensor ceiling derived from official RTX 5070 same-generation table",
        ),
        (
            "l3_fp16_tensor_fp16_accumulate_ceiling",
            rates["fp16_tensor_fp16_accumulate"],
            "favorable L3 dense Tensor accumulation diagnostic",
        ),
    )
    rows = []
    for mode, gpu_rate, exactness in definitions:
        combined = gpu_rate + EPYC_IMPOSSIBLE_PEAK_FLOPS
        floor = MANDATORY_TOTAL_FLOPS / combined
        rows.append(
            {
                "mode": mode,
                "exactness": exactness,
                "gpu_derived_dense_flops_per_second": gpu_rate,
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
        "host_or_storage_streaming_required_even_if_all_experts_are_streamed": True,
    }


def power_ledger() -> dict:
    subtotal = GPU_BOARD_WATTS + EPYC_TDP_WATTS
    return {
        "gpu_total_graphics_power_watts": GPU_BOARD_WATTS,
        "epyc_tdp_watts": EPYC_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": subtotal,
        "psu_combined_12v_watts": PSU_12V_WATTS,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": PSU_12V_WATTS - subtotal,
        "formal_project_wall_cap_watts": 1_000,
        "official_required_system_power_watts": 600,
        "official_supplementary_power": "one PCIe 8-pin or 300-W-or-greater PCIe Gen 5 cable",
        "single_card_nameplate_is_installation_proof": False,
    }


def cost_ledger(market: dict) -> dict:
    if (
        market.get("evidence_class")
        != "dated_semantic_listing_transcription_not_purchase_authority"
        or market.get("product")
        != "GIGABYTE WindForce GeForce RTX 5060 Ti GV-N506TWF2-16GD"
        or market.get("observed_price_usd") != 479.99
        or market.get("availability") != "out_of_stock"
        or market.get("purchase_authority") is not False
    ):
        raise ValueError("RTX 5060 Ti market authority mismatch")
    return {
        "official_launch_msrp_16gb_usd": 429.0,
        "captured_market_price_usd": market["observed_price_usd"],
        "captured_market_availability": market["availability"],
        "card_only_arithmetic_margin_below_cap_usd": 500.0 - market["observed_price_usd"],
        "shipping_known": False,
        "tax_known": False,
        "complete_delivered_bom_within_cap": False,
        "purchase_authorized": False,
        "listing": market,
    }


def pdf_text(path: Path) -> str:
    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _authenticate(paths: dict[str, Path]) -> tuple[dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "pw0127": PW0127_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0158": PW0158_SHA256,
        "pw0161": PW0161_SHA256,
        "product": PRODUCT_SHA256,
        "architecture": ARCHITECTURE_SHA256,
        "launch": LAUNCH_SHA256,
        "market": MARKET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0164 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    config = json.loads(paths["config"].read_text())
    pw0127 = json.loads(paths["pw0127"].read_text())
    pw0151 = json.loads(paths["pw0151"].read_text())
    pw0158 = json.loads(paths["pw0158"].read_text())
    pw0161 = json.loads(paths["pw0161"].read_text())
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
        or pw0158.get("attention_work_ledger", {}).get("mandatory_attention_flops")
        != MANDATORY_ATTENTION_FLOPS
        or pw0161.get("arithmetic", [{}])[0].get("modes", [{}, {}])[1].get(
            "mandatory_matrix_plus_attention_flops"
        )
        != MANDATORY_TOTAL_FLOPS
    ):
        raise ValueError("model, host, or arithmetic authority mismatch")
    product = paths["product"].read_text(errors="strict")
    for value in ("4608", "759 AI TOPS", "16 GB / 8 GB GDDR7", "2.57", ">180</td>"):
        if value not in product:
            raise ValueError(f"official RTX 5060 Ti specification missing: {value}")
    architecture = pdf_text(paths["architecture"])
    for value in (
        "APPENDIX C: Blackwell GB205 GPU",
        "Peak BF16 Tensor TFLOPS",
        "61.7/123.5",
        "Peak FP16 Tensor TFLOPS",
        "123.5/246.9",
    ):
        if value not in architecture:
            raise ValueError(f"official Blackwell throughput authority missing: {value}")
    launch = pdf_text(paths["launch"])
    if "$429 and $379" not in launch or "16GB or 8GB" not in launch:
        raise ValueError("official RTX 5060 Ti launch-price authority mismatch")
    market = json.loads(paths["market"].read_text())
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
    source = next(row for row in arithmetic if row["mode"].startswith("bf16_tensor"))
    l3 = next(row for row in arithmetic if row["mode"].startswith("l3_fp16"))
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0164_affordable_blackwell_complete_system_envelope",
        "revision": REVISION,
        "commit": commit,
        "positions": POSITIONS,
        "source_hashes": source_hashes,
        "rate_derivation": {
            "candidate_cuda_cores": CUDA_CORES,
            "cuda_cores_per_sm": CUDA_CORES_PER_SM,
            "candidate_sms": CUDA_CORES // CUDA_CORES_PER_SM,
            "tensor_cores_per_sm": TENSOR_CORES_PER_SM,
            "candidate_boost_mhz": BOOST_MHZ,
            "reference_rtx5070_sms": REFERENCE_5070_SMS,
            "reference_rtx5070_boost_mhz": REFERENCE_5070_BOOST_MHZ,
            "reference_rtx5070_dense_bf16_fp32acc_flops": REFERENCE_5070_BF16_FP32ACC_FLOPS,
            "reference_rtx5070_dense_fp16_fp16acc_flops": REFERENCE_5070_FP16_FP16ACC_FLOPS,
            "advertised_ai_tops_not_used_as_dense_bf16": AI_TOPS_NOT_DENSE_BF16,
            "sparsity_not_granted": True,
        },
        "arithmetic": arithmetic,
        "capacity": capacity,
        "power": power,
        "cost": cost,
        "decision": {
            "source_oriented_bf16": "reject_ordinary_dense_1m_arithmetic_at_impossible_peak",
            "l3_fp16": "reject_ordinary_dense_1m_arithmetic_at_impossible_peak",
            "card_class": "reject_rtx5060ti_for_ordinary_dense_1m_regardless_of_price",
            "source_bf16_shortfall_seconds": -source["remaining_1m_ttft_seconds"],
            "l3_fp16_shortfall_seconds": -l3["remaining_1m_ttft_seconds"],
            "purchase_authorized": False,
            "runtime_implementation_authorized": False,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "authenticated analytical same-generation rate derivation and moving-market envelope; "
            "not installed hardware, measured CUDA, delivered BOM, endpoint, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official NVIDIA specifications", "market transcription", "prior manifests"],
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
        "product",
        "architecture",
        "launch",
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
        "product",
        "architecture",
        "launch",
        "market",
    )
    manifest = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
