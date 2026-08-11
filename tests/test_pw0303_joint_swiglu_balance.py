import unittest

import numpy as np
import torch

from tools.run_pw0303_joint_swiglu_balance import (
    affine6,
    balance_scales,
    dynamic_input,
)


class PW0161JointSwiGLUBalanceTests(unittest.TestCase):
    def test_alpha_zero_is_identity(self):
        rng = np.random.default_rng(1)
        up = rng.normal(size=(256, 128)).astype(np.float32)
        down = rng.normal(size=(64, 256)).astype(np.float32)
        self.assertTrue(np.array_equal(balance_scales(up, down, 0.0), np.ones(256)))

    def test_real_arithmetic_symmetry(self):
        rng = np.random.default_rng(2)
        hidden = rng.normal(size=(7, 256))
        down = rng.normal(size=(32, 256))
        scales = np.exp(rng.normal(size=256))
        expected = hidden @ down.T
        actual = (hidden / scales) @ (down * scales).T
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-13)

    def test_dynamic_fp8_is_finite_and_shape_preserving(self):
        values = torch.linspace(-2, 2, 512, dtype=torch.bfloat16).reshape(2, 256)
        result = dynamic_input(values)
        self.assertEqual(tuple(result.shape), (2, 256))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_affine6_installs_f16_values(self):
        rng = np.random.default_rng(3)
        weight = rng.normal(size=(4, 256)).astype(np.float32)
        result = affine6(weight)
        self.assertEqual(result.shape, weight.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.array_equal(result, result.astype(np.float16).astype(np.float32)))


if __name__ == "__main__":
    unittest.main()
