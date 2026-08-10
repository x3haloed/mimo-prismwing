#!/usr/bin/env python3
"""Run PW-0169's active Intel A770 Limited Edition BOM preflight."""

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


TARGET_SHA256 = "91fe6e0441bb0a0e1ab0852db60fb575d131b61ff069002c9c333f9b776e4950"
PW0151_SHA256 = "d6919e47f0f4495ccac2ad56ebcfe6662b3309aebd3296c6b546a50836829cb1"
PW0167_SHA256 = "0ff6f2cb1017cb6589b8c5705e7adda349fc2637721e3ddc8c695f051dff2c01"
PSU_PHOTO_SHA256 = "3c398ea5c2a12b71908c5b9adcf16d58fc6e26e867cd7c38c550f42bea367b42"
PRODUCT_SHA256 = "b4691de4514c938e8c0d386a6d1fa6583b96479b4c11ad4aed2726ac1527eccd"
DIMENSIONS_SHA256 = "c7656a01a4aa734b6488309d430eaaf61ad6b48df353d3aeca7a6a357a9eece5"
POWER_SHA256 = "369554f262e5409f7795823b1904ef6767a7af69f7a5c41fba2d445a668450b1"
MARKET_SHA256 = "dd2551749fd8c508d76deea4ea7810ac7ca76a5c181c59df09f9d47e7070d080"

CARD_TBP_WATTS = 225
EPYC_TDP_WATTS = 170
PSU_12V_WATTS = 732
INCREMENTAL_CAP_USD = 500.0


def normalized_html(path: Path) -> str:
    raw = html.unescape(path.read_text(errors="strict"))
    return " ".join(re.sub(r"<[^>]+>", " ", raw).split())


def validate_dimensions(source: dict) -> None:
    expected = {
        "article_id": "000092554",
        "applies_to": "Intel Arc A770 Limited Edition",
        "dimensions_mm": {
            "length_without_io_bracket": 268.6,
            "length_with_io_bracket": 279.9,
            "width_with_shroud": 98.4,
            "width_including_pcie_connector": 111.16,
            "width_including_pcie_connector_and_io_bracket": 126.36,
            "height_with_shroud": 40.81,
            "height_with_shroud_and_io_bracket": 42.0,
        },
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0169 official dimension mismatch: {key}")


def validate_power(source: dict) -> None:
    expected = {
        "article_id": "000092523",
        "product": "Intel Arc A770 Graphics (8 GB/16 GB) Limited Edition",
        "tbp_watts": CARD_TBP_WATTS,
        "power_connectors": "1x8-pin + 1x6-pin",
        "both_external_connectors_required": True,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0169 official power mismatch: {key}")


def validate_market(source: dict) -> None:
    expected = {
        "evidence_class": "dated_semantic_transcription_after_direct_ebay_html_fetch_403",
        "item_id": "168591709192",
        "mpn": "21P01J00BA",
        "condition": "used",
        "item_price_usd": 300.0,
        "observed_shipping_usd": 11.71,
        "seller_location": "Kenmore, Washington, United States",
        "active_buy_it_now": True,
        "complete_delivered_cost_proven": False,
        "purchase_authorized": False,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0169 active listing mismatch: {key}")


def power_ledger() -> dict:
    subtotal = CARD_TBP_WATTS + EPYC_TDP_WATTS
    return {
        "exact_card_tbp_watts": CARD_TBP_WATTS,
        "epyc_tdp_watts": EPYC_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": subtotal,
        "psu_combined_12v_watts": PSU_12V_WATTS,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": PSU_12V_WATTS - subtotal,
        "exact_card_auxiliary_connectors": "1x8-pin + 1x6-pin",
        "both_external_connectors_required": True,
        "candidate_separate_rail_outputs": ["VGA1 (+12V2)", "VGA3 (+12V4)"],
        "original_compatible_cables_present": None,
        "connector_pinout_verified": False,
        "measured_wall_power_watts": None,
        "electrical_installation_proven": False,
    }


def cost_ledger(market: dict) -> dict:
    observed = market["item_price_usd"] + market["observed_shipping_usd"]
    return {
        "incremental_hardware_cap_usd": INCREMENTAL_CAP_USD,
        "observed_item_price_usd": market["item_price_usd"],
        "observed_shipping_usd": market["observed_shipping_usd"],
        "observed_shipping_destination_zip": market["observed_shipping_destination_zip"],
        "observed_item_plus_shipping_usd": observed,
        "headroom_before_actual_destination_delta_tax_and_parts_usd": (
            INCREMENTAL_CAP_USD - observed
        ),
        "actual_destination_shipping_authenticated": False,
        "sales_tax_authenticated": False,
        "required_cable_or_adapter_cost_authenticated": False,
        "active_buy_it_now_observed": True,
        "seller_accepts_returns": False,
        "seller_working_order_claim_is_component_validation": False,
        "complete_delivered_bom_proven": False,
        "purchase_authorized": False,
    }


def physical_ledger(dimensions: dict) -> dict:
    return {
        "card_dimensions_mm": dimensions["dimensions_mm"],
        "available_x16_slots_reported": 3,
        "chassis_model_known": False,
        "measured_length_clearance_mm": None,
        "measured_width_clearance_mm": None,
        "measured_adjacent_slot_clearance_mm": None,
        "airflow_validated": False,
        "physical_installation_proven": False,
    }


def _authenticate(paths: dict[str, Path]) -> tuple[dict[str, str], dict, dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0167": PW0167_SHA256,
        "psu_photo": PSU_PHOTO_SHA256,
        "product": PRODUCT_SHA256,
        "dimensions": DIMENSIONS_SHA256,
        "power": POWER_SHA256,
        "market": MARKET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0169 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    pw0151 = json.loads(paths["pw0151"].read_text())
    pw0167 = json.loads(paths["pw0167"].read_text())
    dimensions = json.loads(paths["dimensions"].read_text())
    power = json.loads(paths["power"].read_text())
    market = json.loads(paths["market"].read_text())
    if "USD $500 total" not in target or "1,000 W" not in target:
        raise ValueError("PW-0169 TARGET authority mismatch")
    if (
        pw0151.get("owned_hardware", {}).get("psu", {}).get("combined_12v_watts")
        != PSU_12V_WATTS
        or pw0167.get("decision", {}).get("source_oriented_bf16")
        != "retain_arithmetic_survivor_only"
    ):
        raise ValueError("PW-0169 prior hardware authority mismatch")
    product = normalized_html(paths["product"])
    for fact in ("Intel® Arc™ A770 Graphics", "16 GB", "225 W", "262"):
        if fact not in product:
            raise ValueError(f"PW-0169 official product fact missing: {fact}")
    validate_dimensions(dimensions)
    validate_power(power)
    validate_market(market)
    return expected, pw0167, dimensions, market


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    source_hashes, pw0167, dimensions, market = _authenticate(paths)
    safety.checkpoint("all_reference_card_and_market_sources_authenticated")
    power = power_ledger()
    cost = cost_ledger(market)
    physical = physical_ledger(dimensions)
    safety.checkpoint("complete_reference_card_preflight_computed")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0169_active_a770_limited_edition_bom_preflight",
        "commit": commit,
        "source_hashes": source_hashes,
        "exact_board": {
            "product": "Intel Arc A770 Limited Edition 16GB",
            "mpn": market["mpn"],
            "condition": market["condition"],
            "seller_working_order_claim": market["seller_note"],
            "dimensions_mm": dimensions["dimensions_mm"],
            "tbp_watts": CARD_TBP_WATTS,
            "power_connectors": "1x8-pin + 1x6-pin",
        },
        "inherited_arithmetic": {
            "pw0167_floor_seconds": pw0167["arithmetic"]["floor_seconds"],
            "pw0167_remaining_1m_ttft_seconds": pw0167["arithmetic"][
                "remaining_1m_ttft_seconds"
            ],
            "status": "arithmetic_survivor_only",
        },
        "power": power,
        "cost": cost,
        "physical": physical,
        "platform_prerequisites": pw0167["platform_prerequisites"],
        "decision": {
            "active_exact_reference_card_found": True,
            "credible_pre_tax_complete_bom_room": True,
            "complete_bom_gate_passed": False,
            "physical_installation_gate_passed": False,
            "platform_gate_passed": False,
            "preferred_over_pw0168_photon_listing": True,
            "branch": "retain_preferred_active_candidate_pending_physical_and_checkout_evidence",
            "next_user_physical_evidence": [
                "photograph the candidate x16 slot and chassis clearance with a ruler",
                "photograph the original EVGA VGA cables and both connector ends",
                "capture a non-purchasing checkout total for item 168591709192",
            ],
            "purchase_authorized": False,
            "runtime_implementation_authorized": False,
        },
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "authenticated official reference-card facts and active domestic listing; not "
            "actual-destination checkout cost, cable inventory, fit, cooling, component test, "
            "supported ReBAR/oneAPI platform, endpoint behavior, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official Intel product source", "semantic source captures", "prior manifests"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "target", "pw0151", "pw0167", "psu_photo", "product", "dimensions", "power",
        "market",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        "target", "pw0151", "pw0167", "psu_photo", "product", "dimensions", "power",
        "market",
    )
    result = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(result), end="")


if __name__ == "__main__":
    main()
