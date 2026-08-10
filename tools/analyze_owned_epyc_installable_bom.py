#!/usr/bin/env python3
"""Run PW-0155's authenticated two-P100 installable-BOM prerequisite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
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
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
PW0154_SHA256 = "1b57250d45f1b24e32f43e93a653fc3d00fa061e37cd0df1c6f0fdff551535f2"
H11SSL_MANUAL_SHA256 = "ec9a6b57cba938f74f555a731a0642df76ba83cdb350e51b855f6d0f9ad2dd1a"
P100_PRODUCT_BRIEF_SHA256 = "bda27f98b088ab9ff54e374048e18093374c510c781efcad1e9325b301df4662"
MARKET_OBSERVATIONS_SHA256 = "33cce2efccd896519d916add6bbe55bee18b6ad5ba8cfb6895c84519e9521492"
INCREMENTAL_HARDWARE_CAP_USD = 500.0
PSU_CONTINUOUS_WATTS = 750
PSU_COMBINED_12V_WATTS = 732
PSU_RAIL_AMPS = 20
P100_COUNT = 2
P100_BOARD_WATTS = 250
P100_AUX_MAX_WATTS = 240
P100_AUX_MAX_AMPS = 20
EPYC_7351P_TDP_WATTS = 170


def topology_ledger() -> dict:
    return {
        "x16_slots": [2, 4, 6],
        "x16_bifurcation_options": ["Auto", "x8x8", "x4x4x4x4"],
        "candidate_assignment": {
            "slot_2": "double_width_P100_0",
            "slot_4": "double_width_P100_1",
            "slot_6": "single_slot_quad_NVMe_carrier_x4x4x4x4",
        },
        "logical_lane_topology_supported": True,
        "physical_chassis_clearance_proven": False,
        "cooler_clearance_proven": False,
    }


def power_ledger() -> dict:
    gpu_plus_cpu = P100_COUNT * P100_BOARD_WATTS + EPYC_7351P_TDP_WATTS
    return {
        "psu_continuous_watts_at_50c": PSU_CONTINUOUS_WATTS,
        "psu_combined_12v_watts": PSU_COMBINED_12V_WATTS,
        "formal_project_wall_cap_watts": 1000,
        "installed_psu_is_tighter_than_project_cap": True,
        "two_p100_board_limit_watts": P100_COUNT * P100_BOARD_WATTS,
        "epyc_7351p_tdp_watts": EPYC_7351P_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": gpu_plus_cpu,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": (
            PSU_COMBINED_12V_WATTS - gpu_plus_cpu
        ),
        "p100_aux_input_maximum_each": {
            "watts": P100_AUX_MAX_WATTS,
            "amps": P100_AUX_MAX_AMPS,
        },
        "psu_rail_rating_amps_each": PSU_RAIL_AMPS,
        "auxiliary_input_can_consume_entire_labeled_rail": (
            P100_AUX_MAX_AMPS == PSU_RAIL_AMPS
        ),
        "full_power_electrical_install_proven": False,
        "reason": (
            "GPU plus CPU nameplates leave only 62 W of aggregate +12 V capacity for "
            "the board, memory, drives, fans, and transients; each P100 auxiliary "
            "input is specified up to the full 20-A label of one PSU rail"
        ),
    }


def parse_market_observations(payload: dict) -> tuple[list[dict], dict]:
    if (
        payload.get("evidence_class") != "dated_web_observation_transcription"
        or payload.get("observation_date") != "2026-08-09"
    ):
        raise ValueError("market observation identity mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 5:
        raise ValueError("market observation item count mismatch")
    by_id = {row.get("id"): row for row in items}
    expected = {
        "p100_cards": (2, 74.37),
        "nvme_drives": (4, 29.00),
        "quad_nvme_carrier": (1, 78.50),
        "p100_power_dongles": (2, 6.08),
        "p100_cooling_kits": (2, 23.99),
    }
    if set(by_id) != set(expected):
        raise ValueError("market observation component set mismatch")
    rows = []
    for item_id, (quantity, unit_price) in expected.items():
        row = by_id[item_id]
        if row.get("quantity") != quantity or row.get("unit_price_usd_at_quantity") != unit_price:
            raise ValueError(f"market observation price mismatch: {item_id}")
        rows.append({**row, "subtotal_usd": quantity * unit_price})
    named_subtotal = round(sum(row["subtotal_usd"] for row in rows), 2)
    storage = by_id["nvme_drives"]
    dongle = by_id["p100_power_dongles"]
    ledger = {
        "named_component_subtotal_usd": named_subtotal,
        "incremental_hardware_cap_usd": INCREMENTAL_HARDWARE_CAP_USD,
        "unallocated_before_tax_shipping_and_missing_parts_usd": (
            round(INCREMENTAL_HARDWARE_CAP_USD - named_subtotal, 2)
        ),
        "all_destination_shipping_known": all(row.get("shipping_usd") is not None for row in rows),
        "tax_known": False,
        "storage_identity_bound": storage.get("identity_issue") is None,
        "storage_sustained_read_proven": False,
        "dongle_pinout_and_construction_authenticated": dongle.get("identity_issue") is None,
        "original_psu_vga_cable_inventory_proven": False,
        "cooling_and_fit_proven": False,
        "arithmetic_subtotal_under_cap": named_subtotal <= INCREMENTAL_HARDWARE_CAP_USD,
        "complete_delivered_bom_under_cap": False,
        "purchase_authorized": False,
    }
    return rows, ledger


def _authenticate_sources(
    pw0151_path: Path,
    pw0154_path: Path,
    manual_path: Path,
    p100_brief_path: Path,
    market_path: Path,
) -> tuple[dict, dict, dict]:
    expected = {
        pw0151_path: PW0151_SHA256,
        pw0154_path: PW0154_SHA256,
        manual_path: H11SSL_MANUAL_SHA256,
        p100_brief_path: P100_PRODUCT_BRIEF_SHA256,
        market_path: MARKET_OBSERVATIONS_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"PW-0155 source hash mismatch: {path.name}")
    if manual_path.read_bytes()[:5] != b"%PDF-" or p100_brief_path.read_bytes()[:5] != b"%PDF-":
        raise ValueError("official hardware authority is not a PDF")
    pw0151 = json.loads(pw0151_path.read_text())
    pw0154 = json.loads(pw0154_path.read_text())
    market = json.loads(market_path.read_text())
    if (
        pw0151.get("evidence_class") != "pw0151_owned_epyc_companion_envelope"
        or pw0151.get("revision") != REVISION
        or pw0151.get("source_hashes", {}).get("psu_photo_sha256")
        != "3c398ea5c2a12b71908c5b9adcf16d58fc6e26e867cd7c38c550f42bea367b42"
        or pw0151.get("owned_hardware", {}).get("psu", {}).get("combined_12v_watts")
        != PSU_COMBINED_12V_WATTS
    ):
        raise ValueError("PW-0151 authority mismatch")
    if (
        pw0154.get("evidence_class") != "pw0154_prompt_calibrated_p100_hbm_cache_bound"
        or pw0154.get("revision") != REVISION
        or pw0154.get("aggregate_hbm_capacity_ledger", {}).get(
            "aggregate_hbm_decimal_bytes"
        )
        != 32_000_000_000
    ):
        raise ValueError("PW-0154 authority mismatch")
    return pw0151, pw0154, market


def run(
    pw0151_path: Path,
    pw0154_path: Path,
    manual_path: Path,
    p100_brief_path: Path,
    market_path: Path,
    output_path: Path,
    commit: str,
) -> dict:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    _pw0151, _pw0154, market = _authenticate_sources(
        pw0151_path, pw0154_path, manual_path, p100_brief_path, market_path
    )
    safety.checkpoint("all_source_evidence_authenticated")
    topology = topology_ledger()
    power = power_ledger()
    components, costs = parse_market_observations(market)
    if costs["named_component_subtotal_usd"] != 403.38:
        raise ValueError("PW-0155 named subtotal changed")
    if power["combined_12v_headroom_after_gpu_plus_cpu_watts"] != 62:
        raise ValueError("PW-0155 +12-V headroom changed")
    safety.checkpoint("topology_power_and_cost_ledgers_complete")
    safety.release_checkpoint(
        "source_documents_released",
        ["PW-0151", "PW-0154", "H11SSL manual", "P100 product brief", "market observations"],
    )
    safety.checkpoint("final_service_health")
    report = {
        "schema_version": 1,
        "evidence_class": "pw0155_owned_epyc_installable_bom_prerequisite",
        "revision": REVISION,
        "commit": commit,
        "source_hashes": {
            "pw0151_analysis_sha256": PW0151_SHA256,
            "pw0154_analysis_sha256": PW0154_SHA256,
            "h11ssl_manual_sha256": H11SSL_MANUAL_SHA256,
            "p100_product_brief_sha256": P100_PRODUCT_BRIEF_SHA256,
            "market_observations_sha256": MARKET_OBSERVATIONS_SHA256,
        },
        "topology": topology,
        "power": power,
        "dated_market_components": components,
        "cost_ledger": costs,
        "hard_stops_before_purchase": [
            "bind four delivered NVMe devices to one exact model with sufficient capacity and return rights",
            "measure or warrant sustained cold sequential read; listing interface claims are not bandwidth evidence",
            "authenticate both P100 dongle pinouts and construction",
            "inventory four original-compatible EVGA VGA feeds without mixing modular cable families",
            "photograph chassis clearance for slots 2, 4, and 6 plus both cooling assemblies",
            "derive a rail assignment and pass staged wall-power, OCP, temperature, ECC, and throttling checks",
        ],
        "decision": (
            "reject_dated_transcribed_bom_as_purchase_authority;"
            "retain_two_p100_quad_nvme_architecture_as_conditional"
        ),
        "gates_passed": True,
        "purchase_authorized": False,
        "A": 0,
        "accepted_tokens": 0,
        "performance_claim": "none; installability and procurement prerequisite only",
        "limitations": [
            "slot support does not prove chassis, blower, or cable clearance",
            "nameplate power values do not predict simultaneous measured load",
            "the market input is a dated transcription and contains an unresolved SSD identity conflict",
            "no tax, complete delivered shipping, device health, sustained bandwidth, CUDA runtime, or endpoint result",
        ],
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1000.0,
        "safety": safety.evidence(),
    }
    atomic_write_new(output_path, canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pw0151", required=True, type=Path)
    parser.add_argument("--pw0154", required=True, type=Path)
    parser.add_argument("--manual", required=True, type=Path)
    parser.add_argument("--p100-brief", required=True, type=Path)
    parser.add_argument("--market", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(
            arguments.pw0151,
            arguments.pw0154,
            arguments.manual,
            arguments.p100_brief,
            arguments.market,
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
