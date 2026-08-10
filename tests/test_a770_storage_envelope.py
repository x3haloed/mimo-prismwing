import unittest

from tools.analyze_a770_storage_envelope import (
    EXPERT_BYTES,
    a770_capacity,
    conditional_match_probability,
    storage_scenarios,
)


class A770StorageEnvelopeTests(unittest.TestCase):
    def test_capacity_uses_complete_expert_slots(self) -> None:
        result = a770_capacity(12_814_555_472, 209_879_040)
        self.assertEqual(result["available_for_complete_experts_bytes"], 634_572_464)
        self.assertEqual(result["complete_expert_slots"], 25)
        self.assertEqual(result["expert_cache_bytes"], 25 * EXPERT_BYTES)
        self.assertEqual(result["unallocated_tail_bytes"], 5_273_264)

    def test_storage_scenarios_keep_serial_nameplate_accounting(self) -> None:
        scenarios = storage_scenarios(22_100_987_904)
        fast = next(
            row for row in scenarios
            if row["lanes"] == 4
            and row["granted_nameplate_bytes_per_second_per_lane"] == 3.5e9
        )
        self.assertAlmostEqual(fast["serial_storage_plus_compute_seconds"], 1.6094660638476266)
        self.assertEqual(fast["targets"]["34.3"]["minimum_integer_A"], 56)
        self.assertEqual(fast["targets"]["50.0"]["minimum_integer_A"], 81)
        slow_three = next(
            row for row in scenarios
            if row["lanes"] == 3
            and row["granted_nameplate_bytes_per_second_per_lane"] == 2.5e9
        )
        self.assertFalse(slow_three["targets"]["50.0"]["possible_with_A_at_most_q"])

    def test_diagnostic_probability_is_not_an_acceptance_claim(self) -> None:
        self.assertAlmostEqual(conditional_match_probability(81), 0.9914745647009651)
        with self.assertRaises(ValueError):
            conditional_match_probability(138)

    def test_capacity_fails_if_mandatory_state_exceeds_hbm(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot hold mandatory"):
            a770_capacity(15_000_000_000, 1_000_000_000)


if __name__ == "__main__":
    unittest.main()
