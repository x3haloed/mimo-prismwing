import unittest

from tools.analyze_native_mtp_cost_upper_bound import perfect_schedule_bound


class NativeMtpCostUpperBoundTests(unittest.TestCase):
    def test_perfect_oracle_selects_best_q_without_using_proposals(self):
        rows = [
            {"control_A": 7, "q4_units": 10, "q8_units": 20},
            {"control_A": 4, "q4_units": 8, "q8_units": 12},
        ]
        result = perfect_schedule_bound(rows)
        self.assertEqual(result["control_A"], 11)
        self.assertEqual(result["control_expert_units"], 32)
        self.assertEqual(result["fixed_q"]["4"]["perfect_max_A"], 6)
        self.assertEqual(result["fixed_q"]["8"]["perfect_max_A"], 14)
        self.assertEqual(
            len(result["perfect_per_window_q_oracle"]["q_choices"]), 2
        )

    def test_empty_bound_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "requires windows"):
            perfect_schedule_bound([])


if __name__ == "__main__":
    unittest.main()
