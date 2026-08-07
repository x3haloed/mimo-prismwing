import unittest

import numpy as np

from tools.run_fixed_residual_hadamard_rotation_control import (
    left_rotate_down,
    normalized_fwht,
    right_rotate,
    right_unrotate,
    rotation_signs,
    unquantized_rotation_parity,
)


class FixedResidualRotationTests(unittest.TestCase):
    def test_normalized_fwht_and_roundtrip(self):
        values = np.arange(16, dtype=np.float64).reshape(2, 8)
        signs = rotation_signs(8)
        np.testing.assert_allclose(right_unrotate(right_rotate(values, signs), signs), values, atol=1e-12)
        transformed = normalized_fwht(np.eye(8))
        np.testing.assert_allclose(transformed @ transformed.T, np.eye(8), atol=1e-12)

    def test_rotation_signs_are_deterministic(self):
        self.assertTrue(np.array_equal(rotation_signs(16), rotation_signs(16)))
        self.assertEqual(set(rotation_signs(16).tolist()), {-1.0, 1.0})
        with self.assertRaises(ValueError):
            rotation_signs(12)

    def test_left_and_right_rotation_preserve_swiglu_expert(self):
        generator = np.random.default_rng(7)
        weights = {
            "gate": generator.normal(size=(4, 8)),
            "up": generator.normal(size=(4, 8)),
            "down": generator.normal(size=(8, 4)),
        }
        inputs = generator.normal(size=(3, 8))
        parity = unquantized_rotation_parity(inputs, weights, rotation_signs(8))
        self.assertLess(parity["forward_relative_l2"], 1e-12)
        self.assertLess(parity["roundtrip_relative_l2"], 1e-12)
        self.assertEqual(left_rotate_down(weights["down"], rotation_signs(8)).shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
