import unittest

import numpy as np

import torch

from tools.generate_layer_major_moe_reference import (
    ROWS,
    TOP_K,
    bf16_widen,
    build_schedule,
    dynamic_input,
)


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

    def test_dynamic_input_uses_independent_group_128_scales(self):
        values = torch.zeros((1, 256), dtype=torch.float32)
        values[0, 0] = 448.0
        values[0, 128] = 224.0
        values[0, 129] = 0.5
        quantized = dynamic_input(values)
        self.assertEqual(float(quantized[0, 0]), 448.0)
        self.assertEqual(float(quantized[0, 128]), 224.0)
        self.assertNotEqual(float(quantized[0, 129]), 0.0)

    def test_bf16_widen_returns_f32_with_bf16_values(self):
        values = torch.tensor([1.00390625], dtype=torch.float32)
        widened = bf16_widen(values)
        self.assertEqual(widened.dtype, torch.float32)
        self.assertEqual(float(widened[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
