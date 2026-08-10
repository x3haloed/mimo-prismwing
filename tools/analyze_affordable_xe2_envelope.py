#!/usr/bin/env python3
"""Run PW-0166's authenticated affordable-Xe2 arithmetic envelope."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import platform
import re
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
IGC_COMMIT = "2eefea9414f2064b2250045305b28a2f73d4f644"
TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
CONFIG_SHA256 = "292a60e74ae9a6d53422b31b21468ce2111c0ab3f7f7a4f4e9c7cd5133b96587"
PW0127_SHA256 = "6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140"
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
PW0158_SHA256 = "3b5b94cae112bee558ec46566ec09652c58bd434c3f47bebd3e0bc7c533fd315"
PW0161_SHA256 = "fc438d593d8ac99be3cc426496feb830256ffc48c75d58fc8bb9d6b09a2c6c8f"
PRODUCT_SHA256 = "f823f01910776e04f4ac5b3bb151b960cb857c96630ca9cbead15a86986679c8"
ARCHITECTURE_SHA256 = "ae4b7eaa179b7eabb5383b951f7b6bd8ae27058f727c724a534acc835899881f"
DATATYPES_SHA256 = "79c9b9a32ccb7d1869777d85384cd06ddc4b2238218eead74cd03c978a40f3d1"
QRG_SHA256 = "6957f49863018e0226b126f5500a97304ff7cab2a9fe61e75019cc7db51b1d4e"
LAUNCH_SHA256 = "597c943c6a4a7ab6d929a4f47c6731fe45427d1b5715bd7369765f8b3437e934"
DPAS_SHA256 = "79ba16ab6716e9099aaaf88875d7213c1a2581601aae8fd7e20fcd70d7737170"
LATENCY_SHA256 = "17502f5b5050ec5538ae3424d09d07a6aea5d32f92b01d71b221bb58f60800c6"

POSITIONS = 1_000_000
TTFT_SECONDS = 1_800.0
MANDATORY_TOTAL_FLOPS = 214_165_790_024_007_680
MANDATORY_MACS_PER_TOKEN = 14_820_573_184
MANDATORY_ATTENTION_FLOPS = 184_524_643_656_007_680
EPYC_IMPOSSIBLE_PEAK_FLOPS = 742_400_000_000

INT8_XMX_TOPS = 233_000_000_000_000
INT8_OPS_PER_CHANNEL = 4
BF16_OPS_PER_CHANNEL = 2

VRAM_BYTES = 12_000_000_000
EXACT_BF16_1M_KV_BYTES = 23_065_559_040
THREE_ARENA_BYTES = 2_340_993_024
NON_ROUTED_SOURCE_BYTES = 12_814_555_472
GPU_BOARD_WATTS = 190
EPYC_TDP_WATTS = 170
PSU_12V_WATTS = 732


def derived_bf16_peak() -> int:
    return INT8_XMX_TOPS * BF16_OPS_PER_CHANNEL // INT8_OPS_PER_CHANNEL


def arithmetic_ledger() -> dict:
    bf16_peak = derived_bf16_peak()
    combined = bf16_peak + EPYC_IMPOSSIBLE_PEAK_FLOPS
    floor = MANDATORY_TOTAL_FLOPS / combined
    return {
        "official_b580_peak_int8_xmx_operations_per_second": INT8_XMX_TOPS,
        "intel_dpas_int8_operations_per_channel": INT8_OPS_PER_CHANNEL,
        "intel_dpas_bf16_operations_per_channel": BF16_OPS_PER_CHANNEL,
        "xe2_dpas_same_precision_independent_latency_and_occupancy": True,
        "derived_bf16_f32acc_operations_per_second": bf16_peak,
        "granted_concurrent_epyc_flops_per_second": EPYC_IMPOSSIBLE_PEAK_FLOPS,
        "impossible_combined_flops_per_second": combined,
        "mandatory_matrix_plus_attention_flops": MANDATORY_TOTAL_FLOPS,
        "floor_seconds": floor,
        "remaining_1m_ttft_seconds": TTFT_SECONDS - floor,
        "passes_1m_ttft_arithmetic_gate": floor <= TTFT_SECONDS,
        "derivation": (
            "Intel's pinned DPAS semantics execute two BF16 versus four INT8 operations per "
            "channel; its Xe2 scheduler uses the same DPAS latency and occupancy independent "
            "of precision, so BF16 throughput is one half of the official INT8 XMX peak"
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
        "host_or_storage_streaming_required_even_if_all_experts_are_streamed": True,
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


def pdf_text(path: Path) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def normalized_html(path: Path) -> str:
    raw = html.unescape(path.read_text(errors="strict"))
    return " ".join(re.sub(r"<[^>]+>", " ", raw).split())


def _authenticate(paths: dict[str, Path]) -> dict[str, str]:
    expected = {
        "target": TARGET_SHA256,
        "config": CONFIG_SHA256,
        "pw0127": PW0127_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0158": PW0158_SHA256,
        "pw0161": PW0161_SHA256,
        "product": PRODUCT_SHA256,
        "architecture": ARCHITECTURE_SHA256,
        "datatypes": DATATYPES_SHA256,
        "qrg": QRG_SHA256,
        "launch": LAUNCH_SHA256,
        "dpas": DPAS_SHA256,
        "latency": LATENCY_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0166 source hash mismatch: {name}")

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

    product = normalized_html(paths["product"])
    for value in ("Intel® Arc™ B580 Graphics", "GPU Peak TOPS (Int8)", "233", "456 GB/s"):
        if value not in product:
            raise ValueError(f"official B580 product fact missing: {value}")
    architecture = normalized_html(paths["architecture"])
    for value in (
        "Architecture X e 2-HPG",
        "GPU Model Intel ® Arc TM B580 Graphics",
        "BattleMage",
        "Matrix Engine (XMX Support or DPAS)",
    ):
        if value not in architecture:
            raise ValueError(f"official Xe2 architecture fact missing: {value}")
    datatypes = normalized_html(paths["datatypes"])
    for value in ("XMX", "FP16", "BF16", "INT8", "Intel® Arc™ B580 Graphics"):
        if value not in datatypes:
            raise ValueError(f"official datatype fact missing: {value}")
    qrg = pdf_text(paths["qrg"])
    for value in ("Intel® Arc B580", "160", "12 GB", "190W", "233", "2670 MHz"):
        if value not in qrg:
            raise ValueError(f"official B580 QRG fact missing: {value}")
    launch = pdf_text(paths["launch"])
    if "Intel Arc B580" not in launch or "from $249" not in launch:
        raise ValueError("official B580 launch-price authority mismatch")

    dpas = paths["dpas"].read_text(errors="strict")
    for value in (
        "if (Src1PrecisionInBits == 16)",
        "OPS_PER_CHAN = 2;",
        "Src1Precision == 8 || Src2Precision == 8",
        "OPS_PER_CHAN = 4;",
        "Exec_size = isPrePVC ? 8 : 16;",
        "DPAS.bf.bf.8.8",
    ):
        if value not in dpas:
            raise ValueError(f"pinned Intel DPAS semantic missing: {value}")
    latency = paths["latency"].read_text(errors="strict")
    for value in (
        "return getDPASLatency(dpas->getRepeatCount());",
        "case Xe2:",
        "case 8:\n      return 33;",
        "LatencyTableXe<PlatformGen::XE>::getOccupancy(const G4_INST *Inst) const",
        "return value_of(LI::OC_OTHERS) * Scale;",
    ):
        if value not in latency:
            raise ValueError(f"pinned Intel Xe2 scheduling fact missing: {value}")
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
        "evidence_class": "pw0166_affordable_xe2_complete_system_envelope",
        "revision": REVISION,
        "commit": commit,
        "igc_commit": IGC_COMMIT,
        "positions": POSITIONS,
        "source_hashes": source_hashes,
        "arithmetic": arithmetic,
        "capacity": capacity,
        "power": power,
        "cost": {
            "official_launch_price_usd": 249,
            "complete_delivered_bom_proven": False,
            "purchase_authorized": False,
        },
        "decision": {
            "source_oriented_bf16": "reject_ordinary_dense_1m_arithmetic_at_derived_xe2_peak",
            "shortfall_seconds": -arithmetic["remaining_1m_ttft_seconds"],
            "purchase_authorized": False,
            "runtime_implementation_authorized": False,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "authenticated analytical product, ISA/compiler, power, and launch-price envelope; "
            "not installed hardware, measured oneAPI, delivered BOM, endpoint, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official Intel specifications", "pinned Intel IGC sources", "prior manifests"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    names = (
        "target", "config", "pw0127", "pw0151", "pw0158", "pw0161",
        "product", "architecture", "datatypes", "qrg", "launch", "dpas", "latency",
    )
    for name in names:
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        "target", "config", "pw0127", "pw0151", "pw0158", "pw0161",
        "product", "architecture", "datatypes", "qrg", "launch", "dpas", "latency",
    )
    manifest = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
