#!/usr/bin/env python3
"""Adjudicate PW-0172's cheaper four-by-2.5-GB/s A770 storage BOM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
PW0169_SHA256 = "127a898e54f51044bf68bf58f80d071e98b2e10130f2b008a6fe0d313d2d9db3"
PW0170_SHA256 = "c8eba5c4348378177d0d297b8eb4713fd9be71aa2f5a7c2790895c35859af5af"
PW0171_SHA256 = "14549b38ee1daee523fd5a76ca9654cdcf7aa6284c651fb36eccac68908b28d3"
DRIVE_SHA256 = "c6538cfe1199f20eb9061763ccf81379a1d26fffe4b992e6afdc5e9f0b6ffc14"
PRODUCT_SHA256 = "33766195fea443024e99cf1a7aadad658e80a94b82df8197fc6fdb52c723aae8"
CARRIER_SHA256 = "a09ee08a0c26d243634778ede3fd48066c83a717a13679ef3e998bcbfc7fd567"

CAP_USD = 500.0
DRIVE_COUNT = 4
CHECKPOINT_BYTES = 315_714_053_402


def validate_drive(source: dict) -> None:
    expected = {
        "evidence_class": "dated_semantic_transcription_of_direct_active_ebay_listing",
        "item_id": "204182011052",
        "active_buy_it_now": True,
        "condition": "used",
        "title_part_number": "MZVLB256HAHQ-000L7",
        "model": "MZ-VLB2560",
        "observed_available_quantity": 6,
        "item_price_usd": 28.99,
        "observed_shipping_usd": 0.0,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0172 drive listing mismatch: {key}")


def validate_product(source: dict) -> None:
    expected = {
        "evidence_class": "dated_semantic_transcription_of_direct_retail_product_page",
        "product": "Samsung PM981",
        "model": "MZVLB256HAHQ-00000",
        "capacity_gb": 256,
        "form_factor": "M.2 2280",
        "interface": "PCI Express Gen3 x4",
        "sequential_read_mb_per_second": 2800,
        "status": "product_specification_not_manufacturer_or_installed_measurement",
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0172 product source mismatch: {key}")


def validate_carrier(source: dict) -> None:
    if (
        source.get("item_id") != "277337205211"
        or source.get("active_buy_it_now") is not True
        or source.get("ports") != 4
        or source.get("bifurcation_required") is not True
        or source.get("item_price_usd") != 39.99
        or source.get("observed_shipping_usd") != 0.0
    ):
        raise ValueError("PW-0172 carrier listing mismatch")


def cost_ledger(card: dict, drive: dict, carrier: dict) -> dict:
    drive_subtotal = DRIVE_COUNT * drive["item_price_usd"]
    card_delivered = card["observed_item_plus_shipping_usd"]
    carrier_delivered = carrier["item_price_usd"] + carrier["observed_shipping_usd"]
    pre_tax = card_delivered + drive_subtotal + drive["observed_shipping_usd"] + carrier_delivered
    remaining = CAP_USD - pre_tax
    taxable_item_subtotal = (
        card["observed_item_price_usd"] + drive_subtotal + carrier["item_price_usd"]
    )
    return {
        "incremental_hardware_cap_usd": CAP_USD,
        "card_item_plus_observed_shipping_usd": card_delivered,
        "drive_count": DRIVE_COUNT,
        "drive_item_plus_observed_shipping_subtotal_usd": (
            drive_subtotal + drive["observed_shipping_usd"]
        ),
        "carrier_item_plus_observed_shipping_usd": carrier_delivered,
        "pre_tax_card_storage_carrier_total_usd": pre_tax,
        "remaining_for_tax_cables_and_cooling_usd": remaining,
        "taxable_item_subtotal_usd": taxable_item_subtotal,
        "break_even_sales_tax_rate_if_cables_and_cooling_are_free": remaining
        / taxable_item_subtotal,
        "actual_destination_tax_usd": None,
        "gpu_cable_cost_usd": None,
        "additional_cooling_cost_usd": None,
        "complete_delivered_bom_proven": False,
    }


def capacity_and_speed_ledger(drive: dict, product: dict) -> dict:
    aggregate = DRIVE_COUNT * product["capacity_gb"] * 1_000_000_000
    return {
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "aggregate_drive_decimal_bytes": aggregate,
        "capacity_headroom_bytes": aggregate - CHECKPOINT_BYTES,
        "retailer_product_read_nameplate_bytes_per_second_per_drive": (
            product["sequential_read_mb_per_second"] * 1_000_000
        ),
        "inherited_conservative_required_bytes_per_second_per_drive": 2_500_000_000,
        "nameplate_margin_fraction_over_required": (
            product["sequential_read_mb_per_second"] / 2500.0 - 1.0
        ),
        "listing_suffix_matches_retail_base_part": (
            drive["title_part_number"].split("-", 1)[0]
            == product["model"].split("-", 1)[0]
        ),
        "manufacturer_speed_authority": False,
        "sustained_concurrent_read_measured": False,
        "platform_bifurcation_measured": False,
    }


def select_four_lane_2_5_scenario(pw0170: dict) -> dict:
    matches = [
        row
        for row in pw0170.get("storage_scenarios", [])
        if row.get("lanes") == 4
        and row.get("granted_nameplate_bytes_per_second_per_lane") == 2_500_000_000.0
    ]
    if len(matches) != 1:
        raise ValueError("PW-0172 inherited 2.5-GB/s scenario mismatch")
    scenario = matches[0]
    if (
        scenario.get("targets", {}).get("34.3", {}).get("minimum_integer_A") != 77
        or scenario.get("targets", {}).get("50.0", {}).get("minimum_integer_A") != 113
    ):
        raise ValueError("PW-0172 inherited acceptance mismatch")
    return scenario


def _authenticate(paths: dict[str, Path]) -> tuple[dict[str, str], dict, dict, dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "pw0169": PW0169_SHA256,
        "pw0170": PW0170_SHA256,
        "pw0171": PW0171_SHA256,
        "drive": DRIVE_SHA256,
        "product": PRODUCT_SHA256,
        "carrier": CARRIER_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0172 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    pw0169 = json.loads(paths["pw0169"].read_text())
    pw0170 = json.loads(paths["pw0170"].read_text())
    pw0171 = json.loads(paths["pw0171"].read_text())
    drive = json.loads(paths["drive"].read_text())
    product = json.loads(paths["product"].read_text())
    carrier = json.loads(paths["carrier"].read_text())
    if "USD $500 total" not in target or "1,000 W" not in target:
        raise ValueError("PW-0172 TARGET authority mismatch")
    if pw0171.get("decision", "").startswith("reject_current_active") is not True:
        raise ValueError("PW-0172 predecessor decision mismatch")
    validate_drive(drive)
    validate_product(product)
    validate_carrier(carrier)
    return expected, pw0169["cost"], pw0170, drive, product, carrier


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    hashes, card, pw0170, drive, product, carrier = _authenticate(paths)
    safety.checkpoint("all_predecessor_market_and_product_sources_authenticated")
    cost = cost_ledger(card, drive, carrier)
    capacity = capacity_and_speed_ledger(drive, product)
    scenario = select_four_lane_2_5_scenario(pw0170)
    safety.checkpoint("conditional_slower_storage_bom_computed")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0172_a770_slower_four_lane_storage_bom",
        "commit": commit,
        "source_hashes": hashes,
        "cost": cost,
        "capacity_and_speed": capacity,
        "inherited_four_lane_2_5_GBps_scenario": scenario,
        "decision": (
            "retain_pre_tax_bom_only_pending_checkout_cables_cooling_installed_storage_and_"
            "base_aligned_q137_A113_proposer;purchase_not_authorized"
        ),
        "purchase_authorized": False,
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "dated active-listing pre-tax ledger and retailer speed nameplate only; no "
            "manufacturer speed authority, checkout, hardware, sustained storage, A770 runtime, "
            "proposer, accepted-token timing, or endpoint TPS"
        ),
    }
    safety.release_checkpoint(
        "source_reports_released", ["predecessor reports", "market transcriptions"]
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = safety.evidence()
    manifest["complete_wall_ms"] = (time.perf_counter() - started) * 1000.0
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("target", "pw0169", "pw0170", "pw0171", "drive", "product", "carrier"):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ("target", "pw0169", "pw0170", "pw0171", "drive", "product", "carrier")
    report = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(report), end="")


if __name__ == "__main__":
    main()
