#!/usr/bin/env python3
"""Adjudicate PW-0171's dated A770 four-lane storage BOM."""

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
DRIVE_SHA256 = "e56a97801ba76b0e06ad1e2ffd1777949386134ea5a03f11700f4956448f1afb"
CARRIER_SHA256 = "a09ee08a0c26d243634778ede3fd48066c83a717a13679ef3e998bcbfc7fd567"
SAMSUNG_SHA256 = "279cf31b28da0a695a368227d65d7dc9300b66165949c8fd55d2ecb5c7d8c58f"

CAP_USD = 500.0
DRIVE_COUNT = 4
CHECKPOINT_BYTES = 315_714_053_402


def validate_drive(source: dict) -> None:
    expected = {
        "evidence_class": "dated_semantic_transcription_of_direct_active_ebay_listing",
        "item_id": "136939844540",
        "active_buy_it_now": True,
        "condition": "used",
        "model": "Samsung PM981a",
        "storage_capacity_gb": 256,
        "observed_available_quantity": 4,
        "item_price_usd": 39.99,
        "minimum_observed_order_shipping_usd": 8.15,
        "tax_calculated_at_checkout": True,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0171 drive listing mismatch: {key}")


def validate_carrier(source: dict) -> None:
    expected = {
        "evidence_class": "dated_semantic_transcription_of_direct_active_ebay_listing",
        "item_id": "277337205211",
        "active_buy_it_now": True,
        "condition": "new",
        "ports": 4,
        "bifurcation_required": True,
        "observed_available_quantity": 7,
        "item_price_usd": 39.99,
        "observed_shipping_usd": 0.0,
        "tax_calculated_at_checkout": True,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0171 carrier listing mismatch: {key}")


def validate_samsung(source: dict) -> None:
    expected = {
        "evidence_class": "dated_semantic_transcription_of_official_manufacturer_product_page",
        "product": "Samsung PM981a",
        "model": "MZVLB256HBHQ-00$00/07",
        "capacity_gb": 256,
        "interface": "PCIe 3.0 x4",
        "form_factor": "M.2",
        "sequential_read_128kb_mb_per_second": 3500,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"PW-0171 Samsung authority mismatch: {key}")


def cost_ledger(card_cost: float, drive: dict, carrier: dict) -> dict:
    drive_items = DRIVE_COUNT * drive["item_price_usd"]
    minimum_storage = (
        drive_items
        + drive["minimum_observed_order_shipping_usd"]
        + carrier["item_price_usd"]
        + carrier["observed_shipping_usd"]
    )
    minimum_total = card_cost + minimum_storage
    return {
        "incremental_hardware_cap_usd": CAP_USD,
        "card_item_plus_observed_shipping_usd": card_cost,
        "drive_count": DRIVE_COUNT,
        "drive_item_subtotal_usd": drive_items,
        "minimum_observed_drive_order_shipping_usd": drive[
            "minimum_observed_order_shipping_usd"
        ],
        "carrier_item_plus_observed_shipping_usd": (
            carrier["item_price_usd"] + carrier["observed_shipping_usd"]
        ),
        "minimum_storage_subtotal_usd": minimum_storage,
        "minimum_card_plus_storage_total_usd": minimum_total,
        "minimum_over_cap_before_tax_cables_and_cooling_usd": max(0.0, minimum_total - CAP_USD),
        "sales_tax_usd": None,
        "gpu_cable_cost_usd": None,
        "additional_cooling_cost_usd": None,
        "complete_delivered_bom_proven": False,
    }


def capacity_ledger(drive: dict, samsung: dict) -> dict:
    aggregate = DRIVE_COUNT * drive["storage_capacity_gb"] * 1_000_000_000
    return {
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "aggregate_drive_decimal_bytes": aggregate,
        "capacity_headroom_bytes": aggregate - CHECKPOINT_BYTES,
        "drive_count": DRIVE_COUNT,
        "independent_pcie_lanes_if_platform_bifurcation_functions": DRIVE_COUNT,
        "manufacturer_nameplate_read_mb_per_second_per_drive": samsung[
            "sequential_read_128kb_mb_per_second"
        ],
        "aggregate_nameplate_read_bytes_per_second": (
            DRIVE_COUNT * samsung["sequential_read_128kb_mb_per_second"] * 1_000_000
        ),
        "sustained_concurrent_read_measured": False,
        "platform_bifurcation_measured": False,
    }


def _authenticate(paths: dict[str, Path]) -> tuple[dict[str, str], dict, dict, dict, dict, dict]:
    expected = {
        "target": TARGET_SHA256,
        "pw0169": PW0169_SHA256,
        "pw0170": PW0170_SHA256,
        "drive": DRIVE_SHA256,
        "carrier": CARRIER_SHA256,
        "samsung": SAMSUNG_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"PW-0171 source hash mismatch: {name}")
    target = paths["target"].read_text(errors="strict")
    pw0169 = json.loads(paths["pw0169"].read_text())
    pw0170 = json.loads(paths["pw0170"].read_text())
    drive = json.loads(paths["drive"].read_text())
    carrier = json.loads(paths["carrier"].read_text())
    samsung = json.loads(paths["samsung"].read_text())
    if "USD $500 total" not in target or "1,000 W" not in target:
        raise ValueError("PW-0171 TARGET authority mismatch")
    if pw0169.get("cost", {}).get("observed_item_plus_shipping_usd") != 311.71:
        raise ValueError("PW-0171 card cost authority mismatch")
    survivor = pw0170.get("strongest_nameplate_survivor", {})
    if survivor.get("lanes") != 4 or survivor.get("bytes_per_second_per_lane") != 3_500_000_000.0:
        raise ValueError("PW-0171 four-lane authority mismatch")
    validate_drive(drive)
    validate_carrier(carrier)
    validate_samsung(samsung)
    return expected, pw0169, pw0170, drive, carrier, samsung


def run(paths: dict[str, Path], output: Path, commit: str) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    authenticate_implementation_commit(commit)
    started = time.perf_counter()
    safety = HostSafetyMonitor()
    hashes, pw0169, pw0170, drive, carrier, samsung = _authenticate(paths)
    safety.checkpoint("all_cost_capacity_and_device_sources_authenticated")
    cost = cost_ledger(pw0169["cost"]["observed_item_plus_shipping_usd"], drive, carrier)
    capacity = capacity_ledger(drive, samsung)
    safety.checkpoint("minimum_complete_storage_bom_computed")
    manifest = {
        "schema_version": 1,
        "evidence_class": "pw0171_a770_four_lane_storage_bom",
        "commit": commit,
        "source_hashes": hashes,
        "storage_candidate": {
            "drives": f"{DRIVE_COUNT}x Samsung PM981a 256GB used",
            "drive_listing_item_id": drive["item_id"],
            "carrier": carrier["product"],
            "carrier_listing_item_id": carrier["item_id"],
        },
        "capacity_and_nameplate": capacity,
        "cost": cost,
        "inherited_required_acceptance": pw0170["strongest_nameplate_survivor"],
        "decision": (
            "reject_current_active_a770_four_lane_storage_bom_over_500_before_tax_cables_and_cooling;"
            "retain_mechanism_pending_cheaper_complete_bom"
        ),
        "purchase_authorized": False,
        "accepted_tokens": 0,
        "A": 0,
        "U": None,
        "performance_claim": None,
        "endpoint_tps": None,
        "limitations": (
            "dated active-listing lower-bound cost and manufacturer storage nameplate only; "
            "not checkout, purchased hardware, measured concurrent storage, installed A770, proposer, "
            "accepted-token timing, or endpoint TPS"
        ),
    }
    safety.release_checkpoint(
        "source_reports_released",
        ["parsed source manifests", "listing transcriptions"],
    )
    safety.checkpoint("final_service_health")
    manifest["safety"] = safety.evidence()
    manifest["complete_wall_ms"] = (time.perf_counter() - started) * 1000.0
    atomic_write_new(output, canonical_json(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("target", "pw0169", "pw0170", "drive", "carrier", "samsung"):
        parser.add_argument(name, type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ("target", "pw0169", "pw0170", "drive", "carrier", "samsung")
    report = run({name: getattr(args, name) for name in names}, args.output, args.commit)
    print(canonical_json(report), end="")


if __name__ == "__main__":
    main()
