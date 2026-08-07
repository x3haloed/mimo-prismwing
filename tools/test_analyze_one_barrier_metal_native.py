from __future__ import annotations

import unittest

from tools.analyze_one_barrier_metal_native import median, select


class OneBarrierAnalysisTests(unittest.TestCase):
    def test_median_requires_three_positive_trials(self) -> None:
        self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)
        with self.assertRaises(ValueError):
            median([1.0, 2.0])
        with self.assertRaises(ValueError):
            median([1.0, 0.0, 2.0])

    def test_select_requires_three_named_state_values(self) -> None:
        trials = [
            {"cache_state": "cold", "wall": 3.0},
            {"cache_state": "cold", "wall": 2.0},
            {"cache_state": "cold", "wall": 1.0},
            {"cache_state": "warm", "wall": 0.5},
        ]
        self.assertEqual(select(trials, "cold", "wall"), [3.0, 2.0, 1.0])
        with self.assertRaises(ValueError):
            select(trials, "warm", "wall")


if __name__ == "__main__":
    unittest.main()
