import unittest

from tools.analyze_a770_four_lane_storage_bom import (
    capacity_ledger,
    cost_ledger,
    validate_carrier,
    validate_drive,
    validate_samsung,
)


def drive() -> dict:
    return {
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


def carrier() -> dict:
    return {
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


def samsung() -> dict:
    return {
        "evidence_class": "dated_semantic_transcription_of_official_manufacturer_product_page",
        "product": "Samsung PM981a",
        "model": "MZVLB256HBHQ-00$00/07",
        "capacity_gb": 256,
        "interface": "PCIe 3.0 x4",
        "form_factor": "M.2",
        "sequential_read_128kb_mb_per_second": 3500,
    }


class A770FourLaneStorageBomTests(unittest.TestCase):
    def test_active_quantity_four_bom_is_already_over_cap(self) -> None:
        validate_drive(drive())
        validate_carrier(carrier())
        ledger = cost_ledger(311.71, drive(), carrier())
        self.assertAlmostEqual(ledger["minimum_storage_subtotal_usd"], 208.10)
        self.assertAlmostEqual(ledger["minimum_card_plus_storage_total_usd"], 519.81)
        self.assertAlmostEqual(
            ledger["minimum_over_cap_before_tax_cables_and_cooling_usd"], 19.81
        )
        self.assertFalse(ledger["complete_delivered_bom_proven"])

    def test_capacity_and_nameplate_are_sufficient_but_unmeasured(self) -> None:
        validate_samsung(samsung())
        ledger = capacity_ledger(drive(), samsung())
        self.assertEqual(ledger["aggregate_drive_decimal_bytes"], 1_024_000_000_000)
        self.assertGreater(ledger["capacity_headroom_bytes"], 700_000_000_000)
        self.assertEqual(ledger["aggregate_nameplate_read_bytes_per_second"], 14_000_000_000)
        self.assertFalse(ledger["sustained_concurrent_read_measured"])

    def test_missing_fourth_drive_fails_closed(self) -> None:
        source = drive()
        source["observed_available_quantity"] = 3
        with self.assertRaisesRegex(ValueError, "observed_available_quantity"):
            validate_drive(source)

    def test_non_bifurcating_carrier_fails_closed(self) -> None:
        source = carrier()
        source["bifurcation_required"] = False
        with self.assertRaisesRegex(ValueError, "bifurcation_required"):
            validate_carrier(source)


if __name__ == "__main__":
    unittest.main()
