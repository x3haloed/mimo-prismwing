import unittest

from tools.analyze_active_a770_le_bom import (
    cost_ledger,
    physical_ledger,
    power_ledger,
    validate_dimensions,
    validate_market,
    validate_listing_images,
    validate_power,
)


def valid_dimensions() -> dict:
    return {
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


def valid_power() -> dict:
    return {
        "article_id": "000092523",
        "product": "Intel Arc A770 Graphics (8 GB/16 GB) Limited Edition",
        "tbp_watts": 225,
        "power_connectors": "1x8-pin + 1x6-pin",
        "both_external_connectors_required": True,
    }


def valid_market() -> dict:
    return {
        "evidence_class": "dated_semantic_transcription_after_direct_ebay_html_fetch_403",
        "item_id": "168591709192",
        "mpn": "21P01J00BA",
        "condition": "used",
        "seller_note": "working",
        "item_price_usd": 300.0,
        "observed_shipping_usd": 11.71,
        "observed_shipping_destination_zip": "27709",
        "seller_location": "Kenmore, Washington, United States",
        "active_buy_it_now": True,
        "complete_delivered_cost_proven": False,
        "purchase_authorized": False,
    }


class ActiveA770LimitedEditionBomTests(unittest.TestCase):
    def test_reference_card_power_matches_pw0167_and_keeps_margin(self) -> None:
        validate_power(valid_power())
        power = power_ledger()
        self.assertEqual(power["gpu_plus_cpu_nameplate_watts"], 395)
        self.assertEqual(power["combined_12v_headroom_after_gpu_plus_cpu_watts"], 337)
        self.assertFalse(power["electrical_installation_proven"])

    def test_domestic_listing_has_substantial_but_incomplete_room(self) -> None:
        market = valid_market()
        validate_market(market)
        cost = cost_ledger(market)
        self.assertEqual(cost["observed_item_plus_shipping_usd"], 311.71)
        self.assertAlmostEqual(
            cost["headroom_before_actual_destination_delta_tax_and_parts_usd"], 188.29
        )
        self.assertFalse(cost["complete_delivered_bom_proven"])

    def test_official_dimensions_are_smaller_than_photon_but_fit_unproved(self) -> None:
        dimensions = valid_dimensions()
        validate_dimensions(dimensions)
        physical = physical_ledger(dimensions)
        self.assertEqual(physical["card_dimensions_mm"]["length_with_io_bracket"], 279.9)
        self.assertEqual(physical["card_dimensions_mm"]["height_with_shroud_and_io_bracket"], 42.0)
        self.assertFalse(physical["physical_installation_proven"])

    def test_wrong_connector_metadata_fails_closed(self) -> None:
        power = valid_power()
        power["power_connectors"] = "3x8-pin"
        with self.assertRaisesRegex(ValueError, "power_connectors"):
            validate_power(power)

    def test_listing_images_bind_box_and_card_but_not_function(self) -> None:
        hashes = (
            "da65a27c20d54a6e00d2058b71c518521d14ee8e8f7841d178e7a814dc9319a3",
            "dd9b6a26ea5e8f06aaacd40c314ce46a6bb16026a5e8a21f3a7a6ad191789330",
            "9658830c4bbea8c094318ec770c4ba712d7220874147bc02b483d84ac3ad2f99",
            "1711a34b9557a00579906f8762f31840e17c511879391a7deae1f94352517f50",
        )
        source = {
            "evidence_class": (
                "human_verified_semantic_transcription_of_authenticated_active_listing_images"
            ),
            "listing_url": "https://www.ebay.com/itm/168591709192",
            "images": [
                {"sha256": hashes[0], "observation": "box"},
                {
                    "sha256": hashes[1],
                    "observation": "product code 21P01J00BA and 8-pin plus one 6-pin",
                },
                {"sha256": hashes[2], "observation": "card"},
                {"sha256": hashes[3], "observation": "ARC A770 Limited Edition"},
            ],
            "actual_component_serial_authenticated": False,
            "component_function_authenticated": False,
            "purchase_authorized": False,
        }
        validate_listing_images(source, hashes)
        source["component_function_authenticated"] = True
        with self.assertRaisesRegex(ValueError, "listing-image"):
            validate_listing_images(source, hashes)


if __name__ == "__main__":
    unittest.main()
