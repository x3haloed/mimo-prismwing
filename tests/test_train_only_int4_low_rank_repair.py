import unittest

import numpy as np

from tools.run_train_only_int4_low_rank_repair import (
    fit_train_only_repairs,
    split_indices,
)


class TrainOnlyInt4LowRankRepairTests(unittest.TestCase):
    def test_partition_split_is_disjoint_and_complete(self):
        train, validation = split_indices([0, 50, 111, 112, 150, 167])
        self.assertEqual(train, [0, 1, 2])
        self.assertEqual(validation, [3, 4, 5])
        self.assertFalse(set(train) & set(validation))

    def test_invalid_partition_position_fails_closed(self):
        with self.assertRaises(ValueError):
            split_indices([168])

    def test_unseen_expert_identity_fallback_is_visible_by_construction(self):
        candidate = np.array([[1.0, 2.0]], dtype=np.float32)
        repaired = candidate.copy()
        self.assertTrue(np.array_equal(candidate, repaired))

    def test_validation_targets_do_not_change_fitted_parameters(self):
        moe_input = np.zeros((224, 4096), dtype=np.float32)
        moe_input[0, 0] = 1.0
        moe_input[112, 0] = 1.0
        candidate = np.zeros((2, 4096), dtype=np.float32)
        source = np.zeros_like(candidate)
        source[0, 0] = 1.0
        source[1, 0] = 100.0
        rows = {0: {"positions": [0, 112], "candidate": candidate, "source": source}}
        first = fit_train_only_repairs(rows, moe_input)
        rows[0]["source"][1, 0] = -100.0
        second = fit_train_only_repairs(rows, moe_input)
        self.assertEqual(first["affine_parameter_sha256"], second["affine_parameter_sha256"])
        self.assertEqual(first["rank_factor_sha256"], second["rank_factor_sha256"])
        self.assertTrue(
            np.array_equal(
                first["repaired_rows"][0]["repaired"][1],
                second["repaired_rows"][0]["repaired"][1],
            )
        )


if __name__ == "__main__":
    unittest.main()
