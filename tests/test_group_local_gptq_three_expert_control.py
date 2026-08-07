import unittest

import numpy as np

from tools.run_group_local_gptq_three_expert_control import (
    gptq_fixed_grid,
    physical_ledger,
    quantize_fixed_grid,
    train_positions,
    validate_grid_membership,
)


class GroupLocalGptqTests(unittest.TestCase):
    def test_fixed_grid_rounding_and_membership(self):
        values = np.array([[-1.0, -0.4, 0.6, 2.0]], dtype=np.float32)
        scales = np.array([[0.2]], dtype=np.float32)
        biases = np.array([[-1.0]], dtype=np.float32)
        codes, dequantized = quantize_fixed_grid(values, scales, biases)
        self.assertEqual(codes.tolist(), [[0, 3, 8, 15]])
        validate_grid_membership(codes, dequantized, scales, biases)

    def test_error_propagation_stays_on_grid(self):
        weight = np.array([[0.3, 0.7, -0.4, 1.1]], dtype=np.float32)
        activations = np.array([[1.0, 2.0, 0.5, -1.0], [0.3, -0.4, 2.0, 1.0]])
        scales = np.array([[0.1]], dtype=np.float32)
        biases = np.array([[-0.4]], dtype=np.float32)
        candidate, codes, diagnostics = gptq_fixed_grid(
            weight, activations, scales, biases, 0.01, "activation"
        )
        validate_grid_membership(codes, candidate, scales, biases)
        self.assertTrue(np.isfinite(candidate).all())
        self.assertEqual(diagnostics["dead_activation_groups"], 0)

    def test_zero_activation_group_is_damped(self):
        weight = np.ones((1, 4), dtype=np.float32)
        activations = np.zeros((2, 4), dtype=np.float32)
        scales = np.array([[0.1]], dtype=np.float32)
        biases = np.array([[0.0]], dtype=np.float32)
        candidate, codes, diagnostics = gptq_fixed_grid(
            weight, activations, scales, biases, 0.001, "natural"
        )
        validate_grid_membership(codes, candidate, scales, biases)
        self.assertEqual(diagnostics["dead_activation_groups"], 1)

    def test_partition_rejects_validation_as_train(self):
        self.assertEqual(train_positions([0, 111, 112, 167]), [0, 111])
        with self.assertRaises(ValueError):
            train_positions([168])

    def test_physical_ledger_is_unchanged_int4(self):
        ledger = physical_ledger()
        self.assertEqual(ledger["packed_bytes_per_expert"], 13_369_344)
        self.assertEqual(ledger["additional_runtime_macs"], 0)


if __name__ == "__main__":
    unittest.main()
