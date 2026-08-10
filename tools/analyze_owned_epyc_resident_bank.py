#!/usr/bin/env python3
"""Run PW-0153's authenticated owned-EPYC resident-bank envelope."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import re
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
CENSUS_SHA256 = "8ac4a179c7b0a06baee05e380dc76acd0a1a64cff4d3e2abe9572ce59afb5c52"
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
MANUAL_SHA256 = "ec9a6b57cba938f74f555a731a0642df76ba83cdb350e51b855f6d0f9ad2dd1a"
MEMORY_LISTING_SHA256 = "2cabdefea523a9daeedb2547162f0b9e32e502db126a8210fc78f3a3b448ad4c"
EXPECTED_TENSOR_BYTES = 315_683_674_448
EXPECTED_Q137_EXPERT_BYTES = 22_730_287_104
EXPECTED_Q137_COMPUTE_SECONDS = 0.20994483892464225
DIMM_BYTES = 64 * 1024**3
DIMM_PRICE_USD = 247.19
TWO_P100_SUBTOTAL_USD = 148.74
PCIE3_X16_BYTES_PER_SECOND = 16e9 * 128 / 130


def capacity_row(module_count: int, tensor_bytes: int = EXPECTED_TENSOR_BYTES) -> dict:
    if module_count <= 0 or tensor_bytes <= 0:
        raise ValueError("capacity inputs must be positive")
    physical_bytes = module_count * DIMM_BYTES
    return {
        "module_count": module_count,
        "physical_bytes": physical_bytes,
        "physical_gib": physical_bytes / 1024**3,
        "tensor_bytes": tensor_bytes,
        "tensor_gib": tensor_bytes / 1024**3,
        "headroom_bytes": physical_bytes - tensor_bytes,
        "headroom_gib": (physical_bytes - tensor_bytes) / 1024**3,
        "fits_complete_tensor_payload": physical_bytes >= tensor_bytes,
        "population_status": (
            "byte_minimum_not_explicitly_enumerated_by_manual"
            if module_count == 5
            else "manual_enumerates_unbalanced_not_recommended"
            if module_count == 6
            else "manual_enumerates_balanced_recommended"
            if module_count == 8
            else "not_evaluated"
        ),
    }


def resident_q137_envelope(
    selected_bytes: int = EXPECTED_Q137_EXPERT_BYTES,
    compute_seconds: float = EXPECTED_Q137_COMPUTE_SECONDS,
) -> dict:
    if selected_bytes <= 0 or compute_seconds <= 0:
        raise ValueError("resident envelope inputs must be positive")
    dual_pcie = 2 * PCIE3_X16_BYTES_PER_SECOND
    five_channel_dram = 5 * 19.2e9
    eight_channel_dram = 8 * 19.2e9
    granted = min(dual_pcie, five_channel_dram)
    transfer_seconds = selected_bytes / granted
    serial_seconds = transfer_seconds + compute_seconds
    targets = {}
    for target in (34.3, 50.0):
        minimum_a = math.ceil(target * serial_seconds)
        targets[str(target)] = {
            "minimum_integer_A": minimum_a,
            "fraction_of_q137": minimum_a / 137,
            "possible_with_A_at_most_q": minimum_a <= 137,
        }
    return {
        "selected_source_expert_bytes": selected_bytes,
        "five_channel_ddr4_2400_nameplate_bytes_per_second": five_channel_dram,
        "eight_channel_ddr4_2400_nameplate_bytes_per_second": eight_channel_dram,
        "pcie3_x16_encoding_adjusted_bytes_per_second_each": PCIE3_X16_BYTES_PER_SECOND,
        "dual_pcie3_x16_nameplate_bytes_per_second": dual_pcie,
        "granted_bottleneck_bytes_per_second": granted,
        "expert_transfer_seconds": transfer_seconds,
        "direct_fp32_block_compute_seconds": compute_seconds,
        "serial_expert_plus_matrix_floor_seconds": serial_seconds,
        "impossible_perfect_acceptance_tps": 137 / serial_seconds,
        "targets": targets,
    }


def parse_memory_listing(html: str) -> dict:
    required = (
        "HMAA8GL7MMR4N-UH",
        "64GB DDR4-2400 ECC LRDIMM",
        "<strong>247</strong><sup>.19</sup>",
        "Sold  by",
        "A-Tech",
        "Add to cart",
    )
    if any(fragment not in html for fragment in required):
        raise ValueError("memory listing semantic mismatch")
    prices = re.findall(r'data-pp-amount="([0-9]+\.[0-9]{2})"', html)
    if not prices or float(prices[0]) != DIMM_PRICE_USD:
        raise ValueError("memory listing price mismatch")
    return {
        "part": "Hynix HMAA8GL7MMR4N-UH",
        "capacity_gib": 64,
        "type": "DDR4-2400 ECC LRDIMM",
        "seller": "A-Tech",
        "unit_price_usd": DIMM_PRICE_USD,
        "add_to_cart_present": True,
        "observation_date": "2026-08-09",
        "url": "https://www.newegg.com/hynix-64gb/p/0RN-000W-003B5",
    }


def _authenticate_sources(
    census_path: Path,
    pw0151_path: Path,
    manual_path: Path,
    listing_path: Path,
) -> tuple[dict, dict, dict]:
    expected = {
        census_path: CENSUS_SHA256,
        pw0151_path: PW0151_SHA256,
        manual_path: MANUAL_SHA256,
        listing_path: MEMORY_LISTING_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"PW-0153 source hash mismatch: {path.name}")
    if manual_path.read_bytes()[:5] != b"%PDF-":
        raise ValueError("Supermicro authority is not a PDF")

    census = json.loads(census_path.read_text())
    pw0151 = json.loads(pw0151_path.read_text())
    if (
        census.get("evidence_class") != "remote_header_census"
        or census.get("revision") != REVISION
        or census.get("tensor_data_bytes") != EXPECTED_TENSOR_BYTES
        or sum(row["data_bytes"] for row in census.get("categories", {}).values())
        != EXPECTED_TENSOR_BYTES
    ):
        raise ValueError("checkpoint census authority mismatch")
    q137 = pw0151.get("route_windows", {}).get("137", [])
    if (
        pw0151.get("evidence_class") != "pw0151_owned_epyc_companion_envelope"
        or pw0151.get("revision") != REVISION
        or len(q137) != 1
        or q137[0].get("selected_source_expert_bytes") != EXPECTED_Q137_EXPERT_BYTES
        or pw0151.get("named_surviving_envelope", {}).get(
            "q137_direct_fp32_block_compute_seconds"
        )
        != EXPECTED_Q137_COMPUTE_SECONDS
        or pw0151.get("direct_fp32_prefill_survivors")
        != ["two_tesla_p100_direct_fp32"]
    ):
        raise ValueError("PW-0151 authority mismatch")
    listing = parse_memory_listing(listing_path.read_text(errors="strict"))
    return census, pw0151, listing


def run(
    census_path: Path,
    pw0151_path: Path,
    manual_path: Path,
    listing_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    census, pw0151, listing = _authenticate_sources(
        census_path, pw0151_path, manual_path, listing_path
    )
    safety.checkpoint("source_evidence_authenticated")

    capacities = [capacity_row(count) for count in (5, 6, 8)]
    envelope = resident_q137_envelope()
    if envelope["targets"]["34.3"]["minimum_integer_A"] != 32:
        raise ValueError("34.3-TPS resident prerequisite changed")
    if envelope["targets"]["50.0"]["minimum_integer_A"] != 47:
        raise ValueError("50-TPS resident prerequisite changed")

    five_dimm_subtotal = 5 * listing["unit_price_usd"]
    minimum_named_subtotal = five_dimm_subtotal + TWO_P100_SUBTOTAL_USD
    safety.checkpoint("capacity_performance_and_cost_ledgers_complete")
    safety.release_checkpoint(
        "source_documents_released",
        ["checkpoint census", "PW-0151 report", "manual identity", "listing HTML"],
    )
    safety.checkpoint("final_service_health")

    report = {
        "schema_version": 1,
        "evidence_class": "pw0153_owned_epyc_resident_bank_envelope",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "remote_header_census_sha256": CENSUS_SHA256,
            "pw0151_analysis_sha256": PW0151_SHA256,
            "supermicro_manual_sha256": MANUAL_SHA256,
            "memory_listing_sha256": MEMORY_LISTING_SHA256,
        },
        "complete_source_tensor_payload": {
            "bytes": census["tensor_data_bytes"],
            "gib": census["tensor_data_bytes"] / 1024**3,
            "categories": census["categories"],
        },
        "official_population_authority": {
            "board_and_cpu_generation": "H11SSL rev 1.01; AMD EPYC 7001",
            "memory_channels_and_slots": 8,
            "supported_64_gib_capacity_gib": 512,
            "same_type_size_speed_required": True,
            "fewer_than_eight_channels_supported_but_not_recommended": True,
            "five_module_population_explicitly_enumerated": False,
            "six_module_population": "enumerated; unbalanced and not recommended",
            "eight_module_population": "enumerated balanced population",
            "existing_four_by_4_gib_bank_reused": False,
        },
        "capacity_ledger": capacities,
        "resident_q137_nameplate_envelope": envelope,
        "prefill_control": {
            "direct_fp32_survivor": pw0151["direct_fp32_prefill_survivors"],
            "two_p100_mandatory_8k_prefill_floor_seconds": next(
                row["mandatory_8k_prefill_floor_seconds"]
                for row in pw0151["direct_fp32_accelerators"]
                if row["name"] == "two_tesla_p100_direct_fp32"
            ),
            "passes_15_second_impossible_floor": True,
        },
        "physical_ledger": {
            "byte_minimum": "five identical 64-GiB modules provide 320 GiB",
            "minimum_directly_enumerated_population": (
                "six identical 64-GiB modules provide 384 GiB; unbalanced and not recommended"
            ),
            "preferred_population": "eight identical 64-GiB modules provide 512 GiB",
            "resident_architecture_physically_possible": True,
            "installation_validated": False,
        },
        "dated_project_ledger_usd": {
            "memory_listing": listing,
            "five_memory_modules_subtotal": five_dimm_subtotal,
            "two_used_p100_card_only_subtotal_from_pw0151": TWO_P100_SUBTOTAL_USD,
            "minimum_named_component_subtotal": minimum_named_subtotal,
            "incremental_hardware_cap": 500.0,
            "ram_alone_exceeds_cap_by": five_dimm_subtotal - 500.0,
            "minimum_named_subtotal_exceeds_cap_by": minimum_named_subtotal - 500.0,
            "omitted_unpriced_items": [
                "tax and any shipping",
                "original-compatible P100 power cables or adapters",
                "forced-air cooling for two passive P100s",
                "physical installation and measured wall-power validation",
            ],
            "complete_bom_within_cap": False,
            "purchase_authorized": False,
        },
        "decision": (
            "retain_source_resident_bank_as_physical_architecture;"
            "reject_dated_procurement_branch_under_500_usd;"
            "reopen_only_with_new_compatible_complete_bom"
        ),
        "limitations": [
            "DRAM and PCIe figures are nameplate ceilings, not measured bandwidth",
            "five modules are the byte minimum but are not explicitly enumerated in the manual's population table",
            "one listing rejects only the documented current procurement branch, not the entire market forever",
            "host work, topology, dispatch, transfer overlap, runtime memory, and endpoint costs are omitted",
            "no hardware was purchased or installed and no model endpoint ran",
        ],
        "A": 0,
        "accepted_tokens": 0,
        "performance_claim": "none; analytical resident-bank envelope only",
        "gates_passed": True,
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "safety": safety.evidence(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--pw0151", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--memory-listing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.census,
        arguments.pw0151,
        arguments.manual,
        arguments.memory_listing,
        arguments.output,
        arguments.commit,
    )
    print(canonical_json(result).decode(), end="")


if __name__ == "__main__":
    main()
