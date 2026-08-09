#!/usr/bin/env python3
"""Run PW-0151's authenticated owned-companion pre-purchase envelope."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
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
PRIMARY_CENSUS_SHA256 = "b8b84a557eabea9c1781186357cf0f2f4fbf75ca7c8a74656beff26fac15978b"
DETAIL_CENSUS_SHA256 = "35121e5cd0d6f3ad9715489eaf3b37f4d4e93f55a20770a31573ba4ccec227ea"
PSU_PHOTO_SHA256 = "3c398ea5c2a12b71908c5b9adcf16d58fc6e26e867cd7c38c550f42bea367b42"
PW0112_SHA256 = "e93d930549ee9fe761d7fc98bf59642088b3eb9f41c712968f8df26d5b2c8b98"
PW0127_SHA256 = "6b81023921824906fea94e2bd5756e9a8ac2ab3f98411e1bfe62fe26d125e140"
MANDATORY_MACS = 14_820_573_184
MANDATORY_OPERATIONS = 2 * MANDATORY_MACS
EPYC_IMPOSSIBLE_FP32 = 16 * 2.9e9 * 16
PREFILL_POSITIONS = 8_000
TTFT_LIMIT_SECONDS = 15.0
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def authenticate_implementation_commit(commit: str) -> None:
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ValueError("implementation commit must be lowercase 40-hex")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if commit != actual:
        raise ValueError("implementation commit does not match repository HEAD")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if dirty:
        raise ValueError("repository has tracked changes; implementation identity is not clean")


def accelerator_ceiling(name: str, count: int, fp32_each: float) -> dict:
    if not name or count <= 0 or fp32_each <= 0:
        raise ValueError("accelerator ceiling inputs must be positive")
    peak = EPYC_IMPOSSIBLE_FP32 + count * fp32_each
    floor = PREFILL_POSITIONS * MANDATORY_OPERATIONS / peak
    return {
        "name": name,
        "accelerator_count": count,
        "advertised_fp32_each": fp32_each,
        "granted_epyc_fp32": EPYC_IMPOSSIBLE_FP32,
        "impossible_combined_fp32": peak,
        "ordinary_decode_compute_tps_ceiling": peak / MANDATORY_OPERATIONS,
        "mandatory_8k_prefill_floor_seconds": floor,
        "passes_impossible_15_second_prefill_floor": floor <= TTFT_LIMIT_SECONDS,
    }


def route_storage_rows(
    q: int,
    layer_unique_experts: list[int],
    expert_bytes: int,
    block_compute_seconds: float,
) -> dict:
    if q <= 0 or len(layer_unique_experts) != 47 or min(layer_unique_experts) < 8:
        raise ValueError("route window identity mismatch")
    records = sum(layer_unique_experts)
    selected_bytes = records * expert_bytes
    scenarios = []
    for per_lane in (2.5e9, 3.5e9):
        for lanes in range(1, 5):
            bandwidth = lanes * per_lane
            storage_seconds = selected_bytes / bandwidth
            compute_seconds = block_compute_seconds
            serial_floor = storage_seconds + compute_seconds
            targets = {}
            for target in (34.3, 50.0):
                required_a = math.ceil(target * serial_floor)
                targets[str(target)] = {
                    "minimum_integer_A": required_a,
                    "fraction_of_q": required_a / q,
                    "possible_with_A_at_most_q": required_a <= q,
                }
            scenarios.append(
                {
                    "lanes": lanes,
                    "granted_nameplate_bytes_per_second_per_lane": per_lane,
                    "granted_aggregate_bytes_per_second": bandwidth,
                    "expert_storage_seconds": storage_seconds,
                    "direct_fp32_block_compute_seconds": compute_seconds,
                    "serial_expert_plus_matrix_floor_seconds": serial_floor,
                    "impossible_perfect_acceptance_tps": q / serial_floor,
                    "targets": targets,
                }
            )
    return {
        "q": q,
        "layer_expert_records": records,
        "selected_source_expert_bytes": selected_bytes,
        "mean_normalized_union_u": records / (47 * 8),
        "scenarios": scenarios,
    }


def _authenticate(
    primary_path: Path,
    detail_path: Path,
    photo_path: Path,
    route_path: Path,
    arithmetic_path: Path,
) -> tuple[dict, dict]:
    expected = {
        primary_path: PRIMARY_CENSUS_SHA256,
        detail_path: DETAIL_CENSUS_SHA256,
        photo_path: PSU_PHOTO_SHA256,
        route_path: PW0112_SHA256,
        arithmetic_path: PW0127_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"PW-0151 source hash mismatch: {path.name}")
    primary = primary_path.read_text(errors="strict")
    detail = detail_path.read_text(errors="strict")
    primary_fragments = (
        "AMD EPYC 7351P 16-Core Processor",
        "CPU(s):                      32",
        "Mem:            15Gi",
    )
    detail_fragments = (
        "Product Name: H11SSL-i",
        "Version: 1.01",
        "Locator: DIMMB1",
        "Locator: DIMMD1",
        "Locator: DIMMF1",
        "Locator: DIMMH1",
        "CPU SLOT2 PCI-E 3.0 X16",
        "CPU SLOT4 PCI-E 3.0 X16",
        "CPU SLOT6 PCI-E 3.0 X16",
        "M.2 PCIE X4",
        "RTL8192EE PCIe Wireless Network Adapter",
        "ST2000DM001-1CH164",
        "SSD 60GB",
    )
    if any(value not in primary for value in primary_fragments):
        raise ValueError("PW-0151 primary census semantic mismatch")
    if any(value not in detail for value in detail_fragments):
        raise ValueError("PW-0151 detailed census semantic mismatch")
    route = json.loads(route_path.read_text())
    arithmetic = json.loads(arithmetic_path.read_text())
    if (
        route.get("evidence_class") != "pw0112_wide_teacher_forced_route_economics"
        or route.get("revision") != REVISION
        or route.get("routed_layers") != 47
        or route.get("top_k") != 8
        or route.get("expert_bytes") != 25_171_968
        or arithmetic.get("evidence_class") != "pw0127_r720_cpu_arithmetic_ceiling"
        or arithmetic.get("revision") != REVISION
        or arithmetic.get("arithmetic_ceiling", {}).get("mandatory_macs_per_token")
        != MANDATORY_MACS
    ):
        raise ValueError("PW-0151 frozen report authority mismatch")
    return route, arithmetic


def run(
    primary_path: Path,
    detail_path: Path,
    photo_path: Path,
    route_path: Path,
    arithmetic_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    route, _arithmetic = _authenticate(
        primary_path, detail_path, photo_path, route_path, arithmetic_path
    )
    safety.checkpoint("all_source_evidence_authenticated")

    cpu_tps = EPYC_IMPOSSIBLE_FP32 / MANDATORY_OPERATIONS
    cpu_only = {
        "impossible_peak_fp32": EPYC_IMPOSSIBLE_FP32,
        "mandatory_operations_per_token": MANDATORY_OPERATIONS,
        "impossible_maximum_tps": cpu_tps,
        "required_fraction_at_34_3_tps": 34.3 / cpu_tps,
        "required_fraction_at_50_tps": 50.0 / cpu_tps,
        "decision": "reject_owned_epyc_cpu_only_for_34_3_and_50_tps",
    }
    accelerators = [
        accelerator_ceiling("one_tesla_p40_direct_fp32", 1, 12e12),
        accelerator_ceiling("one_tesla_p100_direct_fp32", 1, 9.3e12),
        accelerator_ceiling("two_tesla_p100_direct_fp32", 2, 9.3e12),
        accelerator_ceiling("one_tesla_v100_direct_fp32", 1, 14e12),
    ]
    diagnostics_l3 = [
        {
            **accelerator_ceiling("one_tesla_p100_fp16_l3", 1, 18.7e12),
            "exactness": "L3_unvalidated_fp16_target_arithmetic",
        },
        {
            **accelerator_ceiling("one_tesla_v100_tensor_l3", 1, 112e12),
            "exactness": "L3_unvalidated_fp16_tensor_target_arithmetic",
        },
    ]
    surviving_fp32 = [
        row["name"] for row in accelerators if row["passes_impossible_15_second_prefill_floor"]
    ]
    if surviving_fp32 != ["two_tesla_p100_direct_fp32"]:
        raise ValueError("PW-0151 direct-FP32 survivor set changed")
    two_p100_peak = next(
        row["impossible_combined_fp32"]
        for row in accelerators
        if row["name"] == "two_tesla_p100_direct_fp32"
    )
    block_compute_seconds = 137 * MANDATORY_OPERATIONS / two_p100_peak
    route_windows = {}
    for q in (94, 137):
        windows = []
        for window in route["sliding_widths"][str(q)]["windows"]:
            compute_seconds = q * MANDATORY_OPERATIONS / two_p100_peak
            windows.append(
                {
                    "suffix_start": int(window["suffix_start"]),
                    **route_storage_rows(
                        q,
                        [int(value) for value in window["layer_unique_experts"]],
                        int(route["expert_bytes"]),
                        compute_seconds,
                    ),
                }
            )
        route_windows[str(q)] = windows
    safety.checkpoint("compute_and_storage_envelopes_complete")

    q137 = route_windows["137"][0]
    named_survivor = next(
        row
        for row in q137["scenarios"]
        if row["lanes"] == 4
        and row["granted_nameplate_bytes_per_second_per_lane"] == 2.5e9
    )
    if not named_survivor["targets"]["50.0"]["possible_with_A_at_most_q"]:
        raise ValueError("PW-0151 expected four-lane q137 envelope absent")

    safety.release_checkpoint(
        "source_reports_released",
        ["hardware census strings", "PW-0112 route report", "PW-0127 arithmetic report"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0151_owned_epyc_companion_envelope",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "primary_census_sha256": PRIMARY_CENSUS_SHA256,
            "detail_census_sha256": DETAIL_CENSUS_SHA256,
            "psu_photo_sha256": PSU_PHOTO_SHA256,
            "pw0112_analysis_sha256": PW0112_SHA256,
            "pw0127_arithmetic_sha256": PW0127_SHA256,
        },
        "source_authorities": {
            "amd_epyc_7351p": "https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7001-series/amd-epyc-7351p.html",
            "supermicro_h11ssl_i": "https://www.supermicro.com/en/products/motherboard/H11SSL-i",
            "evga_nex750b_manual": "https://www.evga.com/support/manuals/files/120-PB-0750.pdf",
            "nvidia_tesla_p40": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf",
            "nvidia_tesla_p100_pcie": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf",
            "nvidia_tesla_v100": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/v100-application-performance-guide.pdf",
        },
        "owned_hardware": {
            "board": "Supermicro H11SSL-i rev 1.01",
            "cpu": "AMD EPYC 7351P 16C/32T",
            "memory": "four 4-GB DDR4-2133 modules; four channels populated",
            "storage": "2-TB rotating SATA plus 60-GB SATA SSD; no NVMe",
            "slots": "three available PCIe 3.0 x16, two available x8, one x8 Wi-Fi, one M.2 x4",
            "network": "dual 1-GbE; one active",
            "psu": {
                "model": "EVGA SuperNOVA NEX750B",
                "continuous_watts_at_50c": 750,
                "combined_12v_amps": 61,
                "combined_12v_watts": 732,
                "rails": {"count": 4, "amps_each": 20},
                "vga_rail_map": {"VGA1": "+12V2", "VGA2": "+12V2", "VGA3": "+12V4", "VGA4": "+12V4"},
            },
        },
        "cpu_only": cpu_only,
        "direct_fp32_accelerators": accelerators,
        "l3_diagnostics": diagnostics_l3,
        "direct_fp32_prefill_survivors": surviving_fp32,
        "route_windows": route_windows,
        "named_surviving_envelope": {
            "configuration": "two_P100_direct_FP32_plus_four_independent_NVMe_lanes_plus_q137",
            "q137_selected_source_expert_bytes": q137["selected_source_expert_bytes"],
            "q137_direct_fp32_block_compute_seconds": block_compute_seconds,
            "four_lane_2_5_GBps_each": named_survivor,
            "requires_base_aligned_proposer": True,
            "supplied_dflash8_eligible": False,
            "complete_bom_proven": False,
            "electrical_thermal_install_proven": False,
            "measured_stage_proven": False,
            "purchase_authorized": False,
        },
        "market_observations_usd_2026_08_09": {
            "listed_used_p100_card_only_each_at_quantity_two": 74.37,
            "two_p100_card_only_subtotal": 148.74,
            "p100_listing": "https://www.ebay.com/itm/188207963486",
            "m2_to_pcie_adapter_each": 12.80,
            "m2_to_pcie_adapter_listing": "https://www.ebay.com/itm/336284264679",
            "not_included": [
                "NVMe drives with verified health and sustained-read behavior",
                "P100-specific original-compatible power cables or adapters",
                "ducted active cooling for passive accelerators",
                "tax and shipping",
                "power meter and any required network adapter",
            ],
        },
        "physical_stop_conditions": [
            "do not mix modular PSU cables from another model",
            "do not install until cable pinout and GPU connector are verified",
            "do not run passive GPUs without measured forced airflow",
            "do not promote until chassis clearance and slot spacing are photographed",
            "stop on PSU OCP, thermal throttling, ECC faults, or wall power above 1000 W",
        ],
        "decision": "reject_cpu_single_gpu_direct_fp32_and_retain_two_p100_four_lane_q137_as_unproven_envelope",
        "gates_passed": False,
        "safety": safety.evidence(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "accepted_tokens": 0,
        "A": 0,
        "performance_claim": None,
        "limitations": (
            "nameplate analytical envelope; not a BOM, purchase, installed device, storage benchmark, "
            "CUDA kernel, proposer, endpoint result, accepted-token timing, power measurement, or fidelity result"
        ),
        "platform": platform.platform(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-census", required=True, type=Path)
    parser.add_argument("--detail-census", required=True, type=Path)
    parser.add_argument("--psu-photo", required=True, type=Path)
    parser.add_argument("--route-analysis", required=True, type=Path)
    parser.add_argument("--arithmetic-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.primary_census,
            arguments.detail_census,
            arguments.psu_photo,
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
