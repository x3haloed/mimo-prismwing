import unittest

from tools.analyze_a770_slower_storage_bom import (
    capacity_and_speed_ledger,
    cost_ledger,
    select_four_lane_2_5_scenario,
    validate_drive,
    validate_product,
)


def drive() -> dict:
    return {
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


def product() -> dict:
    return {
        "evidence_class": "dated_semantic_transcription_of_direct_retail_product_page",
        "product": "Samsung PM981",
        "model": "MZVLB256HAHQ-00000",
        "capacity_gb": 256,
        "form_factor": "M.2 2280",
        "interface": "PCI Express Gen3 x4",
        "sequential_read_mb_per_second": 2800,
        "status": "product_specification_not_manufacturer_or_installed_measurement",
    }


class A770SlowerStorageBomTests(unittest.TestCase):
    def test_pre_tax_bom_fits_but_checkout_budget_is_tiny(self) -> None:
        validate_drive(drive())
        card = {"observed_item_plus_shipping_usd": 311.71, "observed_item_price_usd": 300.0}
        carrier = {"item_price_usd": 39.99, "observed_shipping_usd": 0.0}
        ledger = cost_ledger(card, drive(), carrier)
        self.assertAlmostEqual(ledger["pre_tax_card_storage_carrier_total_usd"], 467.66)
        self.assertAlmostEqual(ledger["remaining_for_tax_cables_and_cooling_usd"], 32.34)
        self.assertAlmostEqual(
            ledger["break_even_sales_tax_rate_if_cables_and_cooling_are_free"],
            0.070928829915561,
        )
        self.assertFalse(ledger["complete_delivered_bom_proven"])

    def test_capacity_passes_but_speed_is_not_authoritative(self) -> None:
        validate_product(product())
        ledger = capacity_and_speed_ledger(drive(), product())
        self.assertGreater(ledger["capacity_headroom_bytes"], 700_000_000_000)
        self.assertAlmostEqual(ledger["nameplate_margin_fraction_over_required"], 0.12)
        self.assertTrue(ledger["listing_suffix_matches_retail_base_part"])
        self.assertFalse(ledger["manufacturer_speed_authority"])
        self.assertFalse(ledger["sustained_concurrent_read_measured"])

    def test_inherited_scenario_requires_A113_for_fifty(self) -> None:
        scenario = {
            "lanes": 4,
            "granted_nameplate_bytes_per_second_per_lane": 2_500_000_000.0,
            "targets": {
                "34.3": {"minimum_integer_A": 77},
                "50.0": {"minimum_integer_A": 113},
            },
        }
        self.assertIs(select_four_lane_2_5_scenario({"storage_scenarios": [scenario]}), scenario)

    def test_quantity_short_listing_fails_closed(self) -> None:
        source = drive()
        source["observed_available_quantity"] = 3
        with self.assertRaisesRegex(ValueError, "observed_available_quantity"):
            validate_drive(source)


if __name__ == "__main__":
    unittest.main()
