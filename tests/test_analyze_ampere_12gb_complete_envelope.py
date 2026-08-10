import unittest

from tools.analyze_ampere_12gb_complete_envelope import (
    arithmetic_floor,
    cost_ledger,
    optimistic_hbm_expert_capacity,
    storage_lane_scenarios,
)


def pinned_geometry():
    return {
        "hybrid_layer_pattern": [0 if layer in {0, 5, 11, 17, 23, 29, 35, 41, 47} else 1 for layer in range(48)],
        "num_attention_heads": 64,
        "head_dim": 192,
        "v_head_dim": 128,
        "sliding_window_size": 128,
    }


def market_fixture():
    return {
        "evidence_class": "dated_search_and_listing_transcription_not_purchase_authority",
        "components": [
            {"id": "active_rtx3080_12gb", "active": True, "seller_feedback_count": 3, "delivered_before_tax_usd": 446.72, "tax_usd": None},
            {"id": "sold_rtx3080_12gb", "active": False, "sold_price_usd": 362.50},
            {"id": "nvme_256gb", "quantity": 3, "subtotal_usd": 83.52, "rendered_shipping_usd": 14.01, "identity_bound": False, "sustained_read_bound": False},
            {"id": "single_nvme_adapters", "quantity": 3, "subtotal_usd": 30.75, "shipping_usd": 0.0},
        ],
    }


class Ampere12gbCompleteEnvelopeTests(unittest.TestCase):
    def test_one_million_arithmetic_narrowly_survives_favorable_peak(self):
        row = arithmetic_floor(pinned_geometry(), 1_000_000)
        self.assertEqual(row["mandatory_matrix_flops"], 29_641_146_368_000_000)
        self.assertEqual(row["mandatory_attention_flops"], 184_524_643_656_007_680)
        self.assertAlmostEqual(row["matrix_plus_attention_floor_seconds"], 1741.18528474803)
        self.assertLess(row["matrix_plus_attention_floor_seconds"], 1800)

    def test_twelve_gb_holds_only_375_optimistic_experts_at_8k(self):
        row = optimistic_hbm_expert_capacity()
        self.assertEqual(row["complete_expert_slots"], 375)
        self.assertLess(row["unallocated_tail_bytes"], 25_171_968)

    def test_first_4096_positions_already_require_three_lanes(self):
        arithmetic = arithmetic_floor(pinned_geometry(), 8_000)
        rows = storage_lane_scenarios(4585, arithmetic["matrix_plus_attention_floor_seconds"])
        self.assertFalse(rows[1]["passes_15_second_gate"])
        self.assertTrue(rows[2]["passes_15_second_gate"])
        self.assertAlmostEqual(rows[1]["serial_8k_ttft_floor_seconds"], 17.176131974380674)

    def test_active_bom_is_over_cap_before_tax(self):
        row = cost_ledger(market_fixture(), 3)
        self.assertAlmostEqual(row["minimum_storage_and_adapters_before_tax_usd"], 128.28)
        self.assertAlmostEqual(row["active_named_subtotal_before_tax_usd"], 575.0)
        self.assertAlmostEqual(row["maximum_delivered_card_price_before_unknown_tax_to_reopen_usd"], 371.72)
        self.assertFalse(row["captured_active_bom_under_cap"])

    def test_market_fails_closed_on_zero_feedback(self):
        market = market_fixture()
        market["components"][0]["seller_feedback_count"] = 0
        with self.assertRaisesRegex(ValueError, "market authority mismatch"):
            cost_ledger(market, 3)


if __name__ == "__main__":
    unittest.main()
