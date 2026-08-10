#!/usr/bin/env python3
"""Run PW-0165's authenticated affordable-RDNA4 envelope."""

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
PRODUCT_SHA256 = "9013b9e7dfd1e4ecc805e2756df2838b60f3ee0d69a44d25ff8059671194f4ba"
ISA_SHA256 = "96dc97df3468a4e63a13095e2540ba13aaa75cf4635a29516b59760695e25e0c"
LAUNCH_SHA256 = "03df4b873908c7e15ef80644888bfa4f1a49999628eda9c4260e34c6c2cdb977"
MARKET_SHA256 = "79065195a1e523514aa377a91dad8f514db72a504e533b595303699d3148f718"

POSITIONS = 1_000_000
TTFT_SECONDS = 1_800.0
MANDATORY_TOTAL_FLOPS = 214_165_790_024_007_680
MANDATORY_MACS_PER_TOKEN = 14_820_573_184
MANDATORY_ATTENTION_FLOPS = 184_524_643_656_007_680
EPYC_IMPOSSIBLE_PEAK_FLOPS = 742_400_000_000
DIRECT_FP32_FLOPS = 25_600_000_000_000
DENSE_HALF_MATRIX_FLOPS = 103_000_000_000_000
SPARSE_HALF_MATRIX_FLOPS = 205_000_000_000_000

VRAM_BYTES = 16_000_000_000
EXACT_BF16_1M_KV_BYTES = 23_065_559_040
THREE_ARENA_BYTES = 2_340_993_024
NON_ROUTED_SOURCE_BYTES = 12_814_555_472
GPU_BOARD_WATTS = 160
EPYC_TDP_WATTS = 170
PSU_12V_WATTS = 732


def arithmetic_ledger() -> list[dict]:
    definitions = (
        ("direct_fp32_control", DIRECT_FP32_FLOPS, True, "L0/L1 direct arithmetic control"),
        (
            "dense_bf16_f32acc_source_oriented_ceiling",
            DENSE_HALF_MATRIX_FLOPS,
            True,
            "source-oriented favorable grant using the full official dense half-matrix rate",
        ),
        (
            "dense_fp16_l3_ceiling",
            DENSE_HALF_MATRIX_FLOPS,
            True,
            "L3 dense half-matrix diagnostic",
        ),
        (
            "structured_sparse_half_diagnostic",
            SPARSE_HALF_MATRIX_FLOPS,
            False,
            "inadmissible 2:4 structured-sparsity diagnostic",
        ),
    )
    rows = []
    for mode, gpu_rate, admissible, exactness in definitions:
        combined = gpu_rate + EPYC_IMPOSSIBLE_PEAK_FLOPS
        floor = MANDATORY_TOTAL_FLOPS / combined
        rows.append(
            {
                "mode": mode,
                "exactness": exactness,
                "unchanged_source_weights_admissible": admissible,
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
        "gpu_typical_board_power_watts": GPU_BOARD_WATTS,
        "epyc_tdp_watts": EPYC_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": subtotal,
        "psu_combined_12v_watts": PSU_12V_WATTS,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": PSU_12V_WATTS - subtotal,
        "formal_project_wall_cap_watts": 1_000,
        "official_minimum_psu_watts": 450,
        "official_supplementary_power": "one PCIe 8-pin",
        "single_card_nameplate_is_installation_proof": False,
    }


def cost_ledger(market: dict) -> dict:
    if (
        market.get("evidence_class")
        != "dated_semantic_listing_transcription_not_purchase_authority"
        or market.get("item_number") != "N82E16814150910"
        or market.get("condition") != "new"
        or market.get("observed_price_usd") != 449.99
        or market.get("observed_shipping_usd") != 0.0
        or market.get("availability") != "in_stock"
        or market.get("purchase_authority") is not False
    ):
        raise ValueError("RX 9060 XT market authority mismatch")
    subtotal = market["observed_price_usd"] + market["observed_shipping_usd"]
    return {
        "official_launch_sep_16gb_usd": 349.0,
        "captured_new_card_subtotal_before_tax_usd": subtotal,
        "card_only_margin_below_complete_cap_before_tax_usd": 500.0 - subtotal,
        "captured_market_availability": market["availability"],
        "tax_known": False,
        "cable_or_other_installation_cost_included": False,
        "complete_delivered_bom_within_cap": False,
        "purchase_authorized": False,
        "listing": market,
    }


def pdf_text(path: Path) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _authenticate(paths: dict[str, Path]) -> tuple[dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "pw0127": PW0127_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0158": PW0158_SHA256,
        "pw0161": PW0161_SHA256,
        "product": PRODUCT_SHA256,
        "isa": ISA_SHA256,
        "launch": LAUNCH_SHA256,
        "market": MARKET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0165 source hash mismatch: {name}")
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
    for value in ("103 TFLOPs", "205 TFLOPs", "160W", "450W", "16 GB", "1x8-Pin"):
        if value not in product:
            raise ValueError(f"official RX 9060 XT specification missing: {value}")
    isa = pdf_text(paths["isa"])
    for value in (
        "V_WMMA_F32_16X16X16_BF16",
        "V_WMMA_F16_16X16X16_F16",
        "V_SWMMAC_F32_16X16X32_BF16",
        "which 2 elements out of every 4 are zero",
    ):
        if value not in isa:
            raise ValueError(f"official RDNA4 ISA authority missing: {value}")
    launch = paths["launch"].read_text(errors="strict")
    if "RX 9060 XT 16GB" not in launch or "$349" not in launch or "Starting at 160W" not in launch:
        raise ValueError("official RX 9060 XT launch authority mismatch")
    return json.loads(paths["market"].read_text()), expected


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
    dense = next(row for row in arithmetic if row["mode"].startswith("dense_bf16"))
    sparse = next(row for row in arithmetic if row["mode"].startswith("structured_sparse"))
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0165_affordable_rdna4_complete_system_envelope",
        "revision": REVISION,
        "commit": commit,
        "positions": POSITIONS,
        "source_hashes": source_hashes,
        "arithmetic": arithmetic,
        "capacity": capacity,
        "power": power,
        "cost": cost,
        "decision": {
            "dense_source_oriented_bf16": "reject_ordinary_dense_1m_arithmetic_at_impossible_peak",
            "dense_l3_fp16": "reject_ordinary_dense_1m_arithmetic_at_impossible_peak",
            "structured_sparse_half": "inadmissible_for_unchanged_dense_source_weights",
            "dense_shortfall_seconds": -dense["remaining_1m_ttft_seconds"],
            "sparse_idealized_remaining_seconds": sparse["remaining_1m_ttft_seconds"],
            "purchase_authorized": False,
            "runtime_implementation_authorized": False,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "authenticated analytical nameplate, ISA, and moving-market envelope; not installed "
            "hardware, measured HIP, delivered BOM, endpoint, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official AMD specifications", "RDNA4 ISA", "market transcription", "prior manifests"],
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
        "isa",
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
        "isa",
        "launch",
        "market",
    )
    manifest = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
