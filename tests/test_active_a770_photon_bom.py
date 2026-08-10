import unittest

from tools.analyze_active_a770_photon_bom import (
    cost_ledger,
    physical_ledger,
    power_ledger,
    validate_market,
    validate_spec,
)


SPEC_SHA = "75149fac3b91f3447967121a4ea704b31f7be289611924f442ce2870f7a313e7"


def valid_spec() -> dict:
    return {
        "source_image_sha256": SPEC_SHA,
        "product": "GUNNIR Intel Arc A770 Photon 16G OC",
        "maximum_gpu_frequency_mhz": 2400,
        "memory_gb": 16,
        "power_connectors": "2x8-pin",
        "pl1_watts": 195,
        "tbp_watts": 285,
        "dimensions_mm": {"length": 300, "height": 118.5, "thickness": 50},
        "cooling": "3x90mm",
    }


def valid_market() -> dict:
    return {
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


class ActiveA770PhotonBomTests(unittest.TestCase):
    def test_exact_board_power_replaces_reference_card_nameplate(self) -> None:
        power = power_ledger()
        self.assertEqual(power["exact_card_tbp_watts"], 285)
        self.assertEqual(power["gpu_plus_cpu_nameplate_watts"], 455)
        self.assertEqual(power["combined_12v_headroom_after_gpu_plus_cpu_watts"], 277)
        self.assertFalse(power["electrical_installation_proven"])

    def test_active_listing_is_under_cap_before_unproved_costs(self) -> None:
        market = valid_market()
        validate_market(market)
        cost = cost_ledger(market)
        self.assertEqual(cost["observed_item_plus_shipping_usd"], 431.0)
        self.assertEqual(
            cost["headroom_before_sales_tax_and_missing_installation_parts_usd"], 69.0
        )
        self.assertFalse(cost["complete_delivered_bom_proven"])

    def test_spec_is_bound_to_official_panel_and_fit_remains_unknown(self) -> None:
        spec = valid_spec()
        validate_spec(spec, SPEC_SHA)
        physical = physical_ledger(spec)
        self.assertEqual(physical["card_dimensions_mm"]["length"], 300)
        self.assertFalse(physical["physical_installation_proven"])
        spec["source_image_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not bound"):
            validate_spec(spec, SPEC_SHA)

    def test_three_eight_pin_marketplace_noise_fails_closed(self) -> None:
        spec = valid_spec()
        spec["power_connectors"] = "3x8-pin"
        with self.assertRaisesRegex(ValueError, "power_connectors"):
            validate_spec(spec, SPEC_SHA)


if __name__ == "__main__":
    unittest.main()
