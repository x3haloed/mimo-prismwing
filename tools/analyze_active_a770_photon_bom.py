#!/usr/bin/env python3
"""Run PW-0168's exact-board active A770 BOM and installation preflight."""

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
PRODUCT_HTML_SHA256 = "44830fe78ed6971bca45a19df127175419256486318b89147f67202f291a8e1d"
SPEC_PANEL_SHA256 = "75149fac3b91f3447967121a4ea704b31f7be289611924f442ce2870f7a313e7"
SPEC_TRANSCRIPTION_SHA256 = "bc34e4a4e2bb6c76186d82318f3df92c1ad3d8fc8ebfdeb0f959dc15656921d0"
MARKET_TRANSCRIPTION_SHA256 = "c8f040b06ac9e6d776b5ce0d4333090b4bf7087569c73dee0670c8f2c0773836"

CARD_TBP_WATTS = 285
EPYC_TDP_WATTS = 170
PSU_12V_WATTS = 732
INCREMENTAL_CAP_USD = 500.0


def normalized_html(path: Path) -> str:
    raw = html.unescape(path.read_text(errors="strict"))
    return " ".join(re.sub(r"<[^>]+>", " ", raw).split())


def validate_spec(spec: dict, image_sha256: str) -> None:
    expected = {
        "product": "GUNNIR Intel Arc A770 Photon 16G OC",
        "maximum_gpu_frequency_mhz": 2400,
        "memory_gb": 16,
        "power_connectors": "2x8-pin",
        "pl1_watts": 195,
        "tbp_watts": CARD_TBP_WATTS,
        "dimensions_mm": {"length": 300, "height": 118.5, "thickness": 50},
    }
    if spec.get("source_image_sha256") != image_sha256:
        raise ValueError("PW-0168 spec transcription is not bound to the panel image")
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"PW-0168 exact-board specification mismatch: {key}")


def validate_market(market: dict) -> None:
    expected = {
        "evidence_class": "dated_semantic_transcription_after_direct_ebay_html_fetch_403",
        "item_id": "127017511242",
        "mpn": "A770 16G PHOTON OC W",
        "condition": "new",
        "item_price_usd": 411.0,
        "shipping_usd": 20.0,
        "quantity_available": 4,
        "seller_location": "CN, China",
        "complete_delivered_cost_proven": False,
        "purchase_authorized": False,
    }
    for key, value in expected.items():
        if market.get(key) != value:
            raise ValueError(f"PW-0168 active-market observation mismatch: {key}")


def power_ledger() -> dict:
    subtotal = CARD_TBP_WATTS + EPYC_TDP_WATTS
    return {
        "exact_card_tbp_watts": CARD_TBP_WATTS,
        "epyc_tdp_watts": EPYC_TDP_WATTS,
        "gpu_plus_cpu_nameplate_watts": subtotal,
        "psu_combined_12v_watts": PSU_12V_WATTS,
        "combined_12v_headroom_after_gpu_plus_cpu_watts": PSU_12V_WATTS - subtotal,
        "psu_12v_rail_count": 4,
        "psu_12v_amps_each": 20,
        "exact_card_auxiliary_connectors": "2x8-pin",
        "candidate_separate_rail_outputs": ["VGA1 (+12V2)", "VGA3 (+12V4)"],
        "original_compatible_cables_present": None,
        "connector_pinout_verified": False,
        "measured_wall_power_watts": None,
        "electrical_installation_proven": False,
    }


def cost_ledger(market: dict) -> dict:
    observed = market["item_price_usd"] + market["shipping_usd"]
    return {
        "incremental_hardware_cap_usd": INCREMENTAL_CAP_USD,
        "observed_item_price_usd": market["item_price_usd"],
        "observed_shipping_usd": market["shipping_usd"],
        "observed_item_plus_shipping_usd": observed,
        "headroom_before_sales_tax_and_missing_installation_parts_usd": (
            INCREMENTAL_CAP_USD - observed
        ),
        "import_fees_included_by_listing": True,
        "sales_tax_authenticated": False,
        "required_cable_or_adapter_cost_authenticated": False,
        "active_quantity_observed": market["quantity_available"],
        "active_card_observation": True,
        "complete_delivered_bom_proven": False,
        "purchase_authorized": False,
    }


def physical_ledger(spec: dict) -> dict:
    return {
        "card_dimensions_mm": spec["dimensions_mm"],
        "cooler": spec["cooling"],
        "available_x16_slots_reported": 3,
        "chassis_model_known": False,
        "measured_length_clearance_mm": None,
        "measured_adjacent_slot_clearance_mm": None,
        "airflow_validated": False,
        "physical_installation_proven": False,
    }


def _authenticate(paths: dict[str, Path]) -> tuple[dict[str, str], dict, dict, dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "pw0151": PW0151_SHA256,
        "pw0167": PW0167_SHA256,
        "psu_photo": PSU_PHOTO_SHA256,
        "product_html": PRODUCT_HTML_SHA256,
        "spec_panel": SPEC_PANEL_SHA256,
        "spec_transcription": SPEC_TRANSCRIPTION_SHA256,
        "market_transcription": MARKET_TRANSCRIPTION_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0168 source hash mismatch: {name}")

    target = paths["target"].read_text(errors="strict")
    pw0151 = json.loads(paths["pw0151"].read_text())
    pw0167 = json.loads(paths["pw0167"].read_text())
    spec = json.loads(paths["spec_transcription"].read_text())
    market = json.loads(paths["market_transcription"].read_text())
    if "USD $500 total" not in target or "1,000 W" not in target:
        raise ValueError("PW-0168 TARGET authority mismatch")
    if (
        pw0151.get("owned_hardware", {}).get("psu", {}).get("combined_12v_watts")
        != PSU_12V_WATTS
        or pw0167.get("decision", {}).get("source_oriented_bf16")
        != "retain_arithmetic_survivor_only"
        or pw0167.get("platform_prerequisites", {}).get(
            "owned_h11ssl_native_resizable_bar_supported"
        )
        is not False
    ):
        raise ValueError("PW-0168 prior hardware authority mismatch")
    product = normalized_html(paths["product_html"])
    if (
        "GUNNIR Intel Arc A770 Photon 16G OC" not in product
        or "b50afd47-862c-40a4-8212-44a087a01cd7.jpg" not in product
    ):
        raise ValueError("PW-0168 official product-page identity mismatch")
    validate_spec(spec, expected["spec_panel"])
    validate_market(market)
    return expected, pw0151, pw0167, spec, market


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    source_hashes, _pw0151, pw0167, spec, market = _authenticate(paths)
    safety.checkpoint("all_exact_board_and_market_sources_authenticated")
    power = power_ledger()
    cost = cost_ledger(market)
    physical = physical_ledger(spec)
    safety.checkpoint("complete_installation_preflight_computed")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0168_active_a770_photon_bom_installation_preflight",
        "commit": commit,
        "source_hashes": source_hashes,
        "exact_board": spec,
        "inherited_arithmetic": {
            "pw0167_floor_seconds": pw0167["arithmetic"]["floor_seconds"],
            "pw0167_remaining_1m_ttft_seconds": pw0167["arithmetic"][
                "remaining_1m_ttft_seconds"
            ],
            "exact_board_clock_uplift_granted": False,
            "status": "arithmetic_survivor_only",
        },
        "power": power,
        "cost": cost,
        "physical": physical,
        "platform_prerequisites": pw0167["platform_prerequisites"],
        "decision": {
            "active_exact_card_found": True,
            "complete_bom_gate_passed": False,
            "physical_installation_gate_passed": False,
            "platform_gate_passed": False,
            "branch": "retain_exact_active_candidate_pending_physical_and_checkout_evidence",
            "next_user_physical_evidence": [
                "photograph the candidate x16 slot and chassis clearance with a ruler",
                "photograph the original EVGA VGA cables and both connector ends",
                "capture a non-purchasing checkout total for the actual delivery address",
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
            "authenticated active listing and exact-board identity, dimensions, connectors, "
            "TBP, PSU nameplate arithmetic, and inherited PW-0167 compute/platform envelope; "
            "not checkout cost, cable inventory, fit, cooling, installed oneAPI performance, "
            "endpoint behavior, or TPS"
        ),
        "platform": platform.platform(),
        "complete_wall_ms": (time.perf_counter() - started) * 1_000,
    }
    safety.release_checkpoint(
        "source_payloads_released",
        ["official product HTML and panel", "market transcription", "prior manifests"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = [snapshot.to_dict() for snapshot in safety.snapshots]
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "target", "pw0151", "pw0167", "psu_photo", "product_html", "spec_panel",
        "spec_transcription", "market_transcription",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        "target", "pw0151", "pw0167", "psu_photo", "product_html", "spec_panel",
        "spec_transcription", "market_transcription",
    )
    result = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(result), end="")


if __name__ == "__main__":
    main()
