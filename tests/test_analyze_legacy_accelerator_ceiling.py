import unittest

from tools.analyze_legacy_accelerator_ceiling import summarize_windows


class LegacyAcceleratorCeilingAnalysisTests(unittest.TestCase):
    def test_window_summary_preserves_capacity_and_extrema(self):
        windows = [
            {
                "impossible_perfect_acceptance_tps": 70.0,
                "source_expert_transfer_bytes": 20,
                "maximum_layer_expert_records": 28,
                "arena_residency_bytes": {"3": 300},
                "three_arenas_fit_24_decimal_gb": True,
            },
            {
                "impossible_perfect_acceptance_tps": 75.0,
                "source_expert_transfer_bytes": 18,
                "maximum_layer_expert_records": 31,
                "arena_residency_bytes": {"3": 330},
                "three_arenas_fit_24_decimal_gb": True,
            },
        ]
        result = summarize_windows(windows)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["minimum_impossible_tps"], 70.0)
        self.assertEqual(result["maximum_impossible_tps"], 75.0)
        self.assertEqual(result["maximum_layer_expert_records"], 31)
        self.assertTrue(result["all_three_arena_capacity_gates_pass"])

    def test_empty_window_summary_fails_closed(self):
        with self.assertRaises(ValueError):
            summarize_windows([])


if __name__ == "__main__":
    unittest.main()
