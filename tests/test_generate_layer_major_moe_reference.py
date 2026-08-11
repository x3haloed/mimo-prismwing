import unittest

import numpy as np

from tools.generate_layer_major_moe_reference import ROWS, TOP_K, build_schedule


class LayerMajorMoeReferenceTests(unittest.TestCase):
    def test_schedule_preserves_every_route_placement(self):
        selected = np.tile(np.arange(TOP_K, dtype=np.int64), (ROWS, 1))
        weights = np.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=np.float32)
        schedule = build_schedule(selected, weights)
        self.assertEqual(len(schedule), TOP_K)
        self.assertEqual(sum(len(row["positions"]) for row in schedule.values()), ROWS * TOP_K)

    def test_schedule_rejects_duplicate_route(self):
        selected = np.zeros((ROWS, TOP_K), dtype=np.int64)
        weights = np.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=np.float32)
        with self.assertRaises(ValueError):
            build_schedule(selected, weights)


if __name__ == "__main__":
    unittest.main()
