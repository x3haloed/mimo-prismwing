import unittest

import numpy as np
import torch

from tools.mlx_f32_moe_benchmark import dequantize, error_metrics


class ExactF32MoeTests(unittest.TestCase):
    def test_source_fp8_block_scales_expand_exactly(self):
        weight = torch.ones((256, 256), dtype=torch.float8_e4m3fn)
        weight[128:, :128] = -2.0
        scale = torch.tensor([[0.5, 0.25], [0.125, 2.0]], dtype=torch.float32)
        actual = dequantize({"w": weight, "w_scale_inv": scale}, "w")
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (256, 256))
        self.assertEqual(float(actual[0, 0]), 0.5)
        self.assertEqual(float(actual[0, 128]), 0.25)
        self.assertEqual(float(actual[128, 0]), -0.25)
        self.assertEqual(float(actual[128, 128]), 2.0)

    def test_error_metrics_are_independent_f64(self):
        reference = np.array([[3.0, 4.0]], dtype=np.float32)
        actual = np.array([[3.0, 5.0]], dtype=np.float32)
        relative_l2, maximum_absolute_error = error_metrics(actual, reference)
        self.assertAlmostEqual(relative_l2, 0.2)
        self.assertEqual(maximum_absolute_error, 1.0)


if __name__ == "__main__":
    unittest.main()
