import unittest

import numpy as np

from tools.run_global_hessian_gptq_rescue import (
    calibration_positions,
    global_hessian_gptq_fixed_grid,
    projected_workspace_bytes,
)
from tools.run_group_local_gptq_three_expert_control import validate_grid_membership


class GlobalHessianGptqTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        weight = np.linspace(-0.7, 0.8, 256, dtype=np.float32).reshape(1, 256)
        activations = np.stack(
            [
                np.linspace(-1.0, 1.0, 256),
                np.sin(np.linspace(0.0, 8.0, 256)),
                np.cos(np.linspace(0.0, 5.0, 256)),
                np.linspace(1.0, -0.25, 256) ** 3,
            ]
        ).astype(np.float32)
        scales = np.array([[0.1, 0.05]], dtype=np.float32)
        biases = np.array([[-0.8, -0.4]], dtype=np.float32)
        return weight, activations, scales, biases

    def test_cross_block_propagation_and_original_group_grid(self):
        weight, activations, scales, biases = self._fixture()
        candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
            weight, activations, scales, biases, block_size=64
        )
        validate_grid_membership(codes, candidate, scales, biases)
        self.assertEqual(diagnostics["block_count"], 4)
        self.assertGreater(diagnostics["cross_block_update_l2"], 0.0)
        self.assertTrue(np.all(codes <= 15))

    def test_blocked_and_unblocked_assignments_match(self):
        weight, activations, scales, biases = self._fixture()
        blocked, blocked_codes, _ = global_hessian_gptq_fixed_grid(
            weight, activations, scales, biases, block_size=64
        )
        unblocked, unblocked_codes, _ = global_hessian_gptq_fixed_grid(
            weight, activations, scales, biases, block_size=256
        )
        np.testing.assert_array_equal(blocked_codes, unblocked_codes)
        np.testing.assert_array_equal(blocked, unblocked)

    def test_dead_activation_columns_remain_well_defined(self):
        weight = np.linspace(-0.5, 0.5, 128, dtype=np.float32).reshape(1, 128)
        activations = np.zeros((3, 128), dtype=np.float32)
        scales = np.array([[0.1]], dtype=np.float32)
        biases = np.array([[-0.5]], dtype=np.float32)
        candidate, codes, diagnostics = global_hessian_gptq_fixed_grid(
            weight, activations, scales, biases
        )
        validate_grid_membership(codes, candidate, scales, biases)
        self.assertEqual(diagnostics["dead_activation_columns"], 128)

    def test_partition_isolation(self):
        train, validation = calibration_positions([0, 111, 112, 167])
        self.assertEqual(train, [0, 1])
        self.assertEqual(validation, [2, 3])
        with self.assertRaises(ValueError):
            calibration_positions([168])

    def test_projected_workspace_is_conservative_and_shape_checked(self):
        weight = np.zeros((8, 256), dtype=np.float16)
        self.assertGreater(projected_workspace_bytes(weight), 6 * 256 * 256 * 4)
        with self.assertRaises(ValueError):
            projected_workspace_bytes(np.zeros(8))


if __name__ == "__main__":
    unittest.main()
