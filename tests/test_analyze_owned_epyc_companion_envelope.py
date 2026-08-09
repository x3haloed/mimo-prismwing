import unittest

from tools.analyze_owned_epyc_companion_envelope import (
    MANDATORY_OPERATIONS,
    accelerator_ceiling,
    route_storage_rows,
)


class OwnedEpycCompanionEnvelopeTests(unittest.TestCase):
    def test_only_two_p100_direct_fp32_clears_prefill_floor(self):
        one_p100 = accelerator_ceiling("one", 1, 9.3e12)
        two_p100 = accelerator_ceiling("two", 2, 9.3e12)
        one_v100 = accelerator_ceiling("v100", 1, 14e12)
        self.assertFalse(one_p100["passes_impossible_15_second_prefill_floor"])
        self.assertTrue(two_p100["passes_impossible_15_second_prefill_floor"])
        self.assertFalse(one_v100["passes_impossible_15_second_prefill_floor"])

    def test_storage_rows_bind_bytes_compute_and_required_acceptance(self):
        unique = [8] * 47
        two_p100_peak = 16 * 2.9e9 * 16 + 2 * 9.3e12
        compute = 137 * MANDATORY_OPERATIONS / two_p100_peak
        result = route_storage_rows(137, unique, 25_171_968, compute)
        self.assertEqual(result["layer_expert_records"], 376)
        self.assertEqual(result["mean_normalized_union_u"], 1.0)
        scenarios = result["scenarios"]
        self.assertEqual(len(scenarios), 8)
        slow = next(
            row
            for row in scenarios
            if row["lanes"] == 1
            and row["granted_nameplate_bytes_per_second_per_lane"] == 2.5e9
        )
        fast = next(
            row
            for row in scenarios
            if row["lanes"] == 4
            and row["granted_nameplate_bytes_per_second_per_lane"] == 3.5e9
        )
        self.assertGreater(
            slow["serial_expert_plus_matrix_floor_seconds"],
            fast["serial_expert_plus_matrix_floor_seconds"],
        )
        self.assertFalse(slow["targets"]["50.0"]["possible_with_A_at_most_q"])

    def test_invalid_route_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "route window identity"):
            route_storage_rows(137, [8] * 46, 25_171_968, 1.0)


if __name__ == "__main__":
    unittest.main()
