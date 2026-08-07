import unittest

from tools.analyze_wide_route_economics import (
    calibrated_frequency_cache,
    normalized_window_union,
    route_windows,
)


class WideRouteEconomicsTests(unittest.TestCase):
    def test_normalized_union_uses_eight_experts_as_one_unit(self):
        layer = [list(range(8)), list(range(4, 12))]
        union, counts = normalized_window_union([layer] * 47, 0, 2)
        self.assertEqual(counts, [12] * 47)
        self.assertEqual(union, 1.5)

    def test_sliding_width_economics_are_optimistic_acceptance_over_union(self):
        rows = [list(range(8)) for _ in range(137)]
        windows = route_windows([rows] * 47)
        self.assertEqual(windows["94"]["window_count"], 44)
        self.assertEqual(
            windows["94"]["A_over_U_at_impossible_perfect_acceptance"]["maximum"],
            94.0,
        )
        self.assertEqual(windows["137"]["mean_U"]["median"], 1.0)

    def test_calibrated_frequency_cache_never_looks_at_holdout(self):
        calibration = [(1, 1), (1, 1), (1, 2), (1, 3)]
        holdout = [(1, 1), (1, 2), (1, 2), (1, 4)]
        result = calibrated_frequency_cache(calibration, holdout, 2)
        self.assertEqual(result["hits"], 3)
        self.assertEqual(result["misses"], 1)


if __name__ == "__main__":
    unittest.main()
