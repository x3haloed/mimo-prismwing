import unittest

from tools.analyze_cold_storage_speculation_bound import (
    accepted_tps_ceiling,
    minimum_width,
    required_accepted_over_union,
)


class ColdStorageSpeculationBoundTests(unittest.TestCase):
    def test_bound_dimensions_and_width_rounding(self):
        self.assertEqual(accepted_tps_ceiling(10.0, 2.0), 5.0)
        self.assertEqual(required_accepted_over_union(50.0, 2.0), 100.0)
        self.assertEqual(minimum_width(50.0, 2.0), 100)
        self.assertEqual(minimum_width(50.0, 2.0, union=1.5, acceptance_rate=0.75), 200)

    def test_bound_rejects_nonphysical_inputs(self):
        with self.assertRaises(ValueError):
            accepted_tps_ceiling(0.0, 1.0)
        with self.assertRaises(ValueError):
            required_accepted_over_union(1.0, 0.0)
        with self.assertRaises(ValueError):
            minimum_width(1.0, 1.0, union=0.5)
        with self.assertRaises(ValueError):
            minimum_width(1.0, 1.0, acceptance_rate=1.1)


if __name__ == "__main__":
    unittest.main()
