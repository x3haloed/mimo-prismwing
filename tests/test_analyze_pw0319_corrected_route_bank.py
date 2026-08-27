import unittest

from tools.analyze_pw0319_corrected_route_bank import (
    RouteRow,
    coverage_snapshot,
    greedy_order,
)


class Pw0319CoveragePlannerTests(unittest.TestCase):
    def rows(self):
        return [
            RouteRow("a", 0, 0, 1, ((1, 0), (1, 1), (1, 2), (1, 3)), (0.4, 0.3, 0.2, 0.1)),
            RouteRow("b", 1, 0, 1, ((1, 0), (1, 1), (1, 3), (1, 2)), (0.4, 0.3, 0.2, 0.1)),
        ]

    def test_greedy_order_starts_with_three_identities_per_layer(self):
        order = greedy_order(self.rows(), layers=(1,), experts_per_layer=4, maximum_budget=4)
        self.assertEqual(order[:3], [(1, 0), (1, 1), (1, 2)])
        self.assertEqual(set(order), {(1, 0), (1, 1), (1, 2), (1, 3)})

    def test_coverage_snapshot_accounts_for_hits_mass_and_fallbacks(self):
        snapshot = coverage_snapshot(self.rows(), {(1, 0), (1, 1), (1, 2)}, 3)
        self.assertEqual(snapshot["rows_with_at_least_three"], 2)
        self.assertEqual(snapshot["source_fallback_count_distribution"]["5"], 2)
        self.assertAlmostEqual(snapshot["coverage_fraction"], 1.0)
        self.assertAlmostEqual(snapshot["selected_route_weight_fraction"], 0.85)

    def test_budget_below_three_per_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            greedy_order(self.rows(), layers=(1, 2), experts_per_layer=4, maximum_budget=5)


if __name__ == "__main__":
    unittest.main()
