import unittest

from tools.analyze_owned_epyc_installable_bom import (
    parse_market_observations,
    power_ledger,
    topology_ledger,
)


class OwnedEpycInstallableBomTests(unittest.TestCase):
    def test_topology_preserves_four_independent_nvme_lanes(self):
        row = topology_ledger()
        self.assertEqual(row["x16_slots"], [2, 4, 6])
        self.assertIn("x4x4x4x4", row["x16_bifurcation_options"])
        self.assertTrue(row["logical_lane_topology_supported"])
        self.assertFalse(row["physical_chassis_clearance_proven"])

    def test_psu_is_tighter_than_formal_cap_and_has_only_62_watt_margin(self):
        row = power_ledger()
        self.assertEqual(row["combined_12v_headroom_after_gpu_plus_cpu_watts"], 62)
        self.assertTrue(row["installed_psu_is_tighter_than_project_cap"])
        self.assertTrue(row["auxiliary_input_can_consume_entire_labeled_rail"])
        self.assertFalse(row["full_power_electrical_install_proven"])

    def test_market_ledger_fails_closed_on_missing_delivery_and_identity(self):
        payload = {
            "evidence_class": "dated_web_observation_transcription",
            "observation_date": "2026-08-09",
            "items": [
                {
                    "id": item_id,
                    "quantity": quantity,
                    "unit_price_usd_at_quantity": unit_price,
                    "shipping_usd": None,
                    "identity_issue": "conflict" if item_id in {"nvme_drives", "p100_power_dongles"} else None,
                }
                for item_id, quantity, unit_price in (
                    ("p100_cards", 2, 74.37),
                    ("nvme_drives", 4, 29.00),
                    ("quad_nvme_carrier", 1, 78.50),
                    ("p100_power_dongles", 2, 6.08),
                    ("p100_cooling_kits", 2, 23.99),
                )
            ],
        }
        _rows, ledger = parse_market_observations(payload)
        self.assertAlmostEqual(ledger["named_component_subtotal_usd"], 403.38)
        self.assertAlmostEqual(
            ledger["unallocated_before_tax_shipping_and_missing_parts_usd"], 96.62
        )
        self.assertFalse(ledger["all_destination_shipping_known"])
        self.assertFalse(ledger["storage_identity_bound"])
        self.assertFalse(ledger["dongle_pinout_and_construction_authenticated"])
        self.assertFalse(ledger["complete_delivered_bom_under_cap"])


if __name__ == "__main__":
    unittest.main()
