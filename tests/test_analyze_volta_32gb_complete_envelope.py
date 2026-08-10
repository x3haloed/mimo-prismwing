import unittest

from tools.analyze_volta_32gb_complete_envelope import (
    MANDATORY_TOTAL_FLOPS,
    arithmetic_scenarios,
    capacity_ledger,
    cost_ledger,
    power_ledger,
)


class Volta32GBCompleteEnvelopeTest(unittest.TestCase):
    def test_standard_v100_fails_and_v100s_survives_favorable_arithmetic(self):
        rows = {row["id"]: row for row in arithmetic_scenarios()}
        standard = rows["v100_pcie_32gb"]
        faster = rows["v100s_pcie_32gb"]
        self.assertEqual(standard["modes"][1]["mandatory_matrix_plus_attention_flops"], MANDATORY_TOTAL_FLOPS)
        self.assertAlmostEqual(standard["modes"][1]["floor_seconds"], 1899.602900275386)
        self.assertFalse(standard["ordinary_dense_candidate_survives_favorable_arithmetic"])
        self.assertAlmostEqual(faster["modes"][1]["floor_seconds"], 1638.0744886433756)
        self.assertTrue(faster["ordinary_dense_candidate_survives_favorable_arithmetic"])

    def test_exact_capacity_control_does_not_fit(self):
        row = capacity_ledger()
        self.assertEqual(row["full_source_resident_control_bytes"], 38_221_107_536)
        self.assertEqual(row["full_source_resident_control_over_hbm_bytes"], 6_221_107_536)
        self.assertFalse(row["full_source_resident_control_fits"])
        self.assertEqual(row["optimistic_complete_expert_slots"], 261)
        self.assertEqual(row["optimistic_tail_bytes"], 23_564_288)

    def test_card_alone_rejects_both_captured_procurement_branches(self):
        market = {
            "evidence_class": "dated_semantic_listing_transcription_not_purchase_authority",
            "listings": [
                {
                    "id": "v100_pcie_32gb", "active": True, "condition": "used", "memory_gb": 32,
                    "seller_positive_percent": 99.2, "tax_known": False,
                    "observed_price_usd": 679.0, "observed_shipping_usd": 0.0,
                },
                {
                    "id": "v100s_pcie_32gb", "active": True, "condition": "used", "memory_gb": 32,
                    "seller_positive_percent": 99.8, "tax_known": False,
                    "observed_price_usd": 1049.99, "observed_shipping_usd": 5.0,
                },
            ],
        }
        rows = {row["id"]: row for row in cost_ledger(market)}
        self.assertEqual(rows["v100_pcie_32gb"]["card_alone_over_complete_cap_usd"], 179.0)
        self.assertAlmostEqual(rows["v100s_pcie_32gb"]["card_alone_over_complete_cap_usd"], 554.99)
        self.assertFalse(rows["v100_pcie_32gb"]["captured_procurement_branch_survives"])
        self.assertFalse(rows["v100s_pcie_32gb"]["captured_procurement_branch_survives"])

    def test_single_card_power_margin_is_not_installation_proof(self):
        row = power_ledger()
        self.assertEqual(row["combined_12v_headroom_after_gpu_plus_cpu_watts"], 312)
        self.assertFalse(row["single_card_nameplate_is_installation_proof"])


if __name__ == "__main__":
    unittest.main()
