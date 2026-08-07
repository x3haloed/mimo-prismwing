import unittest

import numpy as np

from tools.run_int4_output_affine_repair_oracle import (
    apply_output_repair,
    fit_output_repair,
    partition_local_indices,
)


class Int4OutputAffineRepairOracleTests(unittest.TestCase):
    def test_single_placement_bias_repair_is_exact_with_f16_values(self):
        candidate = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
        source = np.array([[1.25, -1.5, 0.0]], dtype=np.float32)
        scale, bias = fit_output_repair(candidate, source, "affine")
        actual = apply_output_repair(candidate, scale, bias)
        self.assertTrue(np.array_equal(actual, source))

    def test_multi_placement_affine_fit_recovers_scale_and_bias(self):
        candidate = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float32)
        source = candidate * np.array([1.5, -0.5], dtype=np.float32) + np.array([0.25, 2.0])
        scale, bias = fit_output_repair(candidate, source, "affine")
        actual = apply_output_repair(candidate, scale, bias)
        self.assertTrue(np.array_equal(actual, source))

    def test_constant_columns_fall_back_to_unit_scale_and_mean_bias(self):
        candidate = np.ones((3, 2), dtype=np.float32)
        source = np.array([[2.0, 3.0], [2.0, 5.0], [2.0, 7.0]], dtype=np.float32)
        scale, bias = fit_output_repair(candidate, source, "affine")
        self.assertTrue(np.array_equal(scale, np.ones(2, dtype=np.float16)))
        actual = apply_output_repair(candidate, scale, bias)
        self.assertTrue(np.array_equal(actual[:, 0], source[:, 0]))
        self.assertTrue(np.array_equal(actual[:, 1], np.full(3, 5.0, dtype=np.float32)))

    def test_invalid_shapes_and_modes_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_output_repair(np.ones((1, 2)), np.ones((2, 2)), "affine")
        with self.assertRaises(ValueError):
            fit_output_repair(np.ones((1, 2)), np.ones((1, 2)), "matrix")
        with self.assertRaises(ValueError):
            apply_output_repair(
                np.ones((1, 2), dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float16),
            )

    def test_partition_mapping_preserves_prior_dispatch_topology(self):
        train, validation = partition_local_indices([0, 111, 112, 167])
        self.assertEqual(train, [0, 1])
        self.assertEqual(validation, [2, 3])
        with self.assertRaises(ValueError):
            partition_local_indices([168])


if __name__ == "__main__":
    unittest.main()
